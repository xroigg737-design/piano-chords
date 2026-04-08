#!/usr/bin/env python3
"""Flask server for Piano Chord Annotator web interface."""

import os
import sys
import tempfile
import uuid

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

import fitz

from analyzer import analyze_page, estimate_barlines
from config import PREFER_ALGORITHMIC, MIN_CHORD_CONFIDENCE
from note_extractor import detect_is_vector_music, extract_notes
from chord_identifier import analyze_page_chords, identify_chord, _key_sharps_from_signature, _detect_harmonic_rhythm, explain_measure_chord, classify_note_as_chord_tone, is_chromatic_note, detect_inversion, roman_numeral
from musicxml_parser import parse_musicxml
from musicxml_pdf_writer import generate_chord_chart_pdf
from pdf_writer import (
    render_page_to_png,
    detect_barlines,
    build_layout_from_estimates,
    annotate_page,
    save_annotated_pdf,
)

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

UPLOAD_DIR = tempfile.mkdtemp(prefix="piano_chords_")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


def parse_pages(pages_str: str, total: int) -> list[int]:
    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total, int(end))
            for i in range(start, end + 1):
                indices.add(i - 1)
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.add(idx)
    return sorted(indices)


def _handle_musicxml(file, form):
    """Process a MusicXML file: identify chords and generate a PDF chord chart."""
    import zipfile

    notation = form.get("notation", "latin")

    job_id = uuid.uuid4().hex[:12]
    fname_lower = file.filename.lower()

    if fname_lower.endswith(".mxl"):
        # .mxl is a ZIP containing a .xml MusicXML file
        mxl_path = os.path.join(UPLOAD_DIR, f"{job_id}_input.mxl")
        file.save(mxl_path)
        try:
            with zipfile.ZipFile(mxl_path, "r") as zf:
                xml_names = [n for n in zf.namelist()
                             if n.endswith(".xml") and not n.startswith("META-INF")]
                if not xml_names:
                    return jsonify({"error": "No s'ha trobat cap .xml dins del .mxl"}), 400
                input_path = os.path.join(UPLOAD_DIR, f"{job_id}_input.xml")
                with open(input_path, "wb") as out:
                    out.write(zf.read(xml_names[0]))
        except zipfile.BadZipFile:
            return jsonify({"error": "El fitxer .mxl no és un ZIP vàlid"}), 400
        finally:
            try:
                os.remove(mxl_path)
            except OSError:
                pass
    else:
        ext = ".musicxml" if fname_lower.endswith(".musicxml") else ".xml"
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_input{ext}")
        file.save(input_path)

    try:
        mxml_data = parse_musicxml(input_path)
    except Exception as e:
        return jsonify({"error": f"Error processant MusicXML: {e}"}), 400

    # Identify chords and collect notes for each measure
    try:
        measures_data = []
        for measure in mxml_data.measures:
            measure_chords = _detect_harmonic_rhythm(measure, mxml_data.key_sharps)
            chord_symbol = measure_chords[0]["chord"] if measure_chords else ""

            # Classify ALL notes: chord tone, passing, or chromatic
            # Separate by staff (treble = right hand, bass = left hand)
            rh_notes = []  # right hand (treble staff)
            lh_notes = []  # left hand (bass staff)
            seen_rh = set()
            seen_lh = set()

            for ng in measure.note_groups:
                for n in sorted(ng.notes, key=lambda x: x.midi_pitch):
                    is_ct = classify_note_as_chord_tone(n, chord_symbol, mxml_data.key_sharps)
                    is_chrom = is_chromatic_note(n, mxml_data.key_sharps)
                    # Type: "chord", "passing", or "chromatic"
                    if is_chrom and not is_ct:
                        note_type = "chromatic"
                    elif is_ct:
                        note_type = "chord"
                    else:
                        note_type = "passing"

                    entry = {"name": n.full_name, "type": note_type}

                    if n.staff == "bass":
                        if n.full_name not in seen_lh:
                            seen_lh.add(n.full_name)
                            lh_notes.append(entry)
                    else:
                        if n.full_name not in seen_rh:
                            seen_rh.add(n.full_name)
                            rh_notes.append(entry)

            # Detect inversion and Roman numeral
            bass_pc = None
            if measure.note_groups and measure.note_groups[0].notes:
                bass_note = min(measure.note_groups[0].notes, key=lambda n: n.midi_pitch)
                bass_pc = bass_note.pitch_class

            inversion = detect_inversion(chord_symbol, bass_pc, mxml_data.key_sharps)
            degree = roman_numeral(chord_symbol, mxml_data.key_sharps, mxml_data.key_mode)

            # Generate explanation
            explanation = explain_measure_chord(measure, mxml_data.key_sharps, notation=notation)

            measures_data.append({
                "measure_index": measure.measure_index,
                "chords": measure_chords,
                "rh_notes": rh_notes,
                "lh_notes": lh_notes,
                "degree": degree,
                "inversion": inversion,
                "explanation": explanation,
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error analitzant acords: {e}"}), 400

    # Generate PDF chord chart
    base_name = os.path.splitext(file.filename)[0]
    output_path = os.path.join(UPLOAD_DIR, f"{job_id}_acordes.pdf")

    try:
        generate_chord_chart_pdf(
            title=mxml_data.title or base_name,
            measures_data=measures_data,
            key_sharps=mxml_data.key_sharps,
            key_mode=mxml_data.key_mode,
            time_num=mxml_data.time_num,
            time_den=mxml_data.time_den,
            total_measures=mxml_data.total_measures,
            notation=notation,
            output_path=output_path,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generant PDF: {e}"}), 400

    # Clean input
    try:
        os.remove(input_path)
    except OSError:
        pass

    return send_file(
        output_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{base_name}_acordes.pdf",
    )


@app.route("/annotate", methods=["POST"])
def annotate():
    if "file" not in request.files:
        return jsonify({"error": "Cap fitxer enviat"}), 400

    file = request.files["file"]
    fname_lower = file.filename.lower()
    is_musicxml = fname_lower.endswith(".xml") or fname_lower.endswith(".musicxml") or fname_lower.endswith(".mxl")
    is_pdf = fname_lower.endswith(".pdf")

    if not is_pdf and not is_musicxml:
        return jsonify({"error": "Només fitxers PDF o MusicXML"}), 400

    if is_musicxml:
        return _handle_musicxml(file, request.form)

    notation = request.form.get("notation", "latin")
    font_size_str = request.form.get("font_size", "auto")
    pages_str = request.form.get("pages", "")

    font_size = font_size_str
    if font_size != "auto":
        try:
            font_size = float(font_size)
        except ValueError:
            font_size = "auto"

    # Save uploaded file
    job_id = uuid.uuid4().hex[:12]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_input.pdf")
    file.save(input_path)

    try:
        doc = fitz.open(input_path)
    except Exception as e:
        return jsonify({"error": f"Error obrint el PDF: {e}"}), 400

    total_pages = len(doc)

    if pages_str:
        page_indices = parse_pages(pages_str, total_pages)
    else:
        page_indices = list(range(total_pages))

    log_messages = []

    for page_idx in page_indices:
        page = doc[page_idx]
        pnum = page_idx + 1
        log_messages.append({"text": f"Pàgina {pnum}/{total_pages}", "type": "info"})

        layout = detect_barlines(page)
        if layout.has_vector_barlines:
            total_m = sum(s.measure_count for s in layout.systems)
            log_messages.append({"text": f"  {len(layout.systems)} sistemes, {total_m} compassos (vector)", "type": "info"})
        else:
            log_messages.append({"text": "  Sense barres vectorials", "type": "info"})

        is_vector = PREFER_ALGORITHMIC and detect_is_vector_music(page)

        if not layout.has_vector_barlines:
            png_bytes = render_page_to_png(page)
            log_messages.append({"text": "  Estimant barres via Claude...", "type": "info"})
            try:
                estimates = estimate_barlines(png_bytes)
                layout = build_layout_from_estimates(page, estimates)
                total_m = sum(s.measure_count for s in layout.systems)
                log_messages.append({"text": f"  Estimats {len(layout.systems)} sistemes, {total_m} compassos", "type": "info"})
            except Exception as e:
                log_messages.append({"text": f"  Error estimant barres: {e}", "type": "warn"})
                continue

        if not layout.systems:
            log_messages.append({"text": "  Cap sistema detectat, saltant", "type": "warn"})
            continue

        analysis = None

        if is_vector:
            log_messages.append({"text": "  Extracció algorítmica", "type": "info"})
            try:
                note_data = extract_notes(page, layout)
                if note_data.has_music_glyphs:
                    analysis = analyze_page_chords(note_data, layout, page_idx)
                    n_groups = sum(len(m.note_groups) for m in note_data.measures)
                    log_messages.append({"text": f"  {n_groups} grups de notes extrets", "type": "info"})
                else:
                    log_messages.append({"text": "  Insuficients glifs, fallback a Claude Vision", "type": "warn"})
            except Exception as e:
                log_messages.append({"text": f"  Error extracció: {e}", "type": "warn"})

        if analysis is None:
            if not locals().get("png_bytes"):
                png_bytes = render_page_to_png(page)
            log_messages.append({"text": "  Analitzant via Claude Vision...", "type": "info"})
            try:
                analysis = analyze_page(
                    png_bytes,
                    page_number=page_idx,
                    num_systems_hint=len(layout.systems),
                    measures_hint=[s.measure_count for s in layout.systems],
                )
            except Exception as e:
                log_messages.append({"text": f"  Error anàlisi: {e}", "type": "err"})
                continue

        if not analysis.get("has_music", True):
            log_messages.append({"text": f"  No s'ha detectat música", "type": "warn"})
            continue

        annotate_page(page, layout, analysis, notation=notation, font_size=font_size)
        n_chords = sum(
            len(m.get("chords", []))
            for s in analysis.get("systems", [])
            for m in s.get("measures", [])
        )
        log_messages.append({"text": f"  Anotat ({n_chords} acords)", "type": "info"})

    output_path = save_annotated_pdf(doc, input_path)
    doc.close()

    # Clean input
    try:
        os.remove(input_path)
    except OSError:
        pass

    # Return the annotated PDF directly
    return send_file(
        output_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=file.filename.replace(".pdf", "_acordes.pdf"),
    )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5070
    print(f"Annotador d'Acords PDF server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

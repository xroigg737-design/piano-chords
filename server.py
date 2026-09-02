#!/usr/bin/env python3
"""Flask server for Piano Chord Annotator web interface."""

import base64
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


# ---------------------------------------------------------------------------
# Deep harmonic analysis via Claude
# ---------------------------------------------------------------------------

def _format_notes_for_claude(mxml_data, notation="latin"):
    """Format MusicXML data as structured text for Claude analysis."""
    from config import ANGLO_TO_LATIN

    key_names_sharp = ["Do", "Sol", "Re", "La", "Mi", "Si", "Fa#"]
    key_names_flat = ["Do", "Fa", "Sib", "Mib", "Lab", "Reb", "Solb"]
    if mxml_data.key_sharps >= 0:
        key_name = key_names_sharp[min(mxml_data.key_sharps, 6)]
    else:
        key_name = key_names_flat[min(-mxml_data.key_sharps, 6)]
    mode_cat = "major" if mxml_data.key_mode == "major" else "menor"

    sharp_notes_latin = ["Fa", "Do", "Sol", "Re", "La", "Mi", "Si"]
    flat_notes_latin = ["Si", "Mi", "La", "Re", "Sol", "Do", "Fa"]
    if mxml_data.key_sharps > 0:
        altered = [f"{sharp_notes_latin[i]}#" for i in range(min(mxml_data.key_sharps, 7))]
        key_sig_desc = f"{mxml_data.key_sharps} sostingut(s): {', '.join(altered)}"
    elif mxml_data.key_sharps < 0:
        n = min(-mxml_data.key_sharps, 7)
        altered = [f"{flat_notes_latin[i]}b" for i in range(n)]
        key_sig_desc = f"{n} bemoll(s): {', '.join(altered)}"
    else:
        key_sig_desc = "cap alteració (0 sostinguts, 0 bemolls)"

    lines = []
    lines.append(f"TÍTOL: {mxml_data.title or 'Sense títol'}")
    lines.append(f"TONALITAT: {key_name} {mode_cat}")
    lines.append(f"ARMADURA DE CLAU: {key_sig_desc}")
    lines.append(f"COMPÀS: {mxml_data.time_num}/{mxml_data.time_den}")
    lines.append(f"TOTAL COMPASSOS: {mxml_data.total_measures}")
    lines.append("")

    def _note_name(note):
        name = note.pitch_name
        if notation == "latin":
            name = ANGLO_TO_LATIN.get(name, name)
        acc = ""
        if note.accidental == 1:
            acc = "#"
        elif note.accidental == -1:
            acc = "b"
        elif note.accidental == 2:
            acc = "##"
        elif note.accidental == -2:
            acc = "bb"
        return f"{name}{acc}{note.octave}"

    for measure in mxml_data.measures:
        mi = measure.measure_index + 1
        treble_notes = []
        bass_notes = []
        for ng in measure.note_groups:
            beat = round(ng.x, 2)
            for n in sorted(ng.notes, key=lambda x: -x.midi_pitch):
                entry = f"{_note_name(n)}(t={beat})"
                if n.staff == "bass":
                    bass_notes.append(entry)
                else:
                    treble_notes.append(entry)

        lines.append(f"COMPÀS {mi}:")
        lines.append(f"  Mà dreta (clau sol): {', '.join(treble_notes) if treble_notes else '(silenci)'}")
        lines.append(f"  Mà esquerra (clau fa): {', '.join(bass_notes) if bass_notes else '(silenci)'}")

    return "\n".join(lines)


HARMONIC_ANALYSIS_SYSTEM_PROMPT = """\
Ets un professor expert en harmonia musical i anàlisi tonal amb anys d'experiència \
ensenyant a conservatoris. Fas anàlisi harmònica professional, precisa i didàctica.

TASCA: Analitza compàs per compàs la partitura de piano proporcionada. Produeix una \
anàlisi harmònica completa i rigorosa en català.

═══ QUÈ HAS D'ANALITZAR PER CADA COMPÀS ═══

1. **Acord**: Identifica l'acord amb símbol jazz (Do, Rem, Sol7, Faø, etc.) usant \
notació llatina (Do Re Mi Fa Sol La Si).

2. **Numeral romà**: Grau dins la tonalitat actual (I, ii, iii, IV, V, vi, viiº). \
Majúscules = major, minúscules = menor. Afegeix 7 si escau (V7, viio7, ii7, etc.).

3. **Funció tonal**: T (tònica), S (subdominant), D (dominant), DD (dominant de la \
dominant), tp (tònica paral·lela = relatiu menor de la tònica), sp (subdominant \
paral·lela), dp (dominant paral·lela). Usa la nomenclatura funcional de Riemann.

4. **Inversió**: Estat fonamental (EF), 1a inversió (6), 2a inversió (6/4), \
3a inversió d'un acord de 7a (4/2, 6/5, 4/3). Indica el baix real.

5. **Notes no harmòniques**: Identifica notes de pas, retards, brodadures \
(mordents), apoggiatures, escapades, anticipacions. Especifica quines notes \
exactes són i de quin tipus.

6. **Cadències**: Identifica cadències quan es produeixin: \
perfecta (V→I), imperfecta, plagal (IV→I), semicadència (→V), \
rota/enganyosa (V→vi), frigiana, napolitana.

7. **Modulacions**: Detecta canvis de tonalitat. Indica la tonalitat d'origen, \
la nova tonalitat, l'acord pivot (si n'hi ha) i el tipus de modulació \
(diatònica, cromàtica, enarmònica, directa).

8. **Observacions didàctiques**: Per cada compàs o grup de compassos, afegeix \
comentaris que ajudin l'estudiant a entendre: progressions típiques, patrons \
recurrents, relacions entre veus, conducció de veus notable.

═══ FORMAT DE SORTIDA — JSON vàlid ═══

{
  "title": "<títol de l'obra>",
  "key": "<tonalitat principal, ex: Do major>",
  "time_signature": "<compàs, ex: 4/4>",
  "structure": "<descripció breu de la forma: ABA, binària, sonata, etc.>",
  "measures": [
    {
      "measure": <número>,
      "chord_symbol": "<símbol acord en notació llatina>",
      "roman_numeral": "<numeral romà amb qualitat>",
      "tonal_function": "<T/S/D/DD/tp/sp/dp>",
      "inversion": "<EF/6/6_4/6_5/4_3/4_2>",
      "bass_note": "<nota del baix>",
      "non_harmonic_tones": [
        {"note": "<nota>", "type": "<pas/retard/brodadura/appogiatura/escapada/anticipació>"}
      ],
      "cadence": "<null o tipus de cadència>",
      "modulation": "<null o descripció>",
      "observations": "<comentari didàctic breu>"
    }
  ],
  "modulations_summary": [
    {"from_key": "<tonalitat>", "to_key": "<tonalitat>", "at_measure": <n>, "type": "<tipus>", "pivot_chord": "<acord pivot o null>"}
  ],
  "pedagogical_summary": "<resum de 3-5 paràgrafs per a l'estudiant: estructura harmònica general, progressions principals, elements destacables, consells per a la memorització i interpretació>"
}

═══ REGLES IMPORTANTS ═══
- PRIMER DE TOT: identifica l'armadura de clau comptant EXACTAMENT el nombre de \
sostinguts o bemolls. Referència: 0=Do, 1#=Sol, 2#=Re, 3#=La, 4#=Mi, 5#=Si, \
6#=Fa#, 1b=Fa, 2b=Sib, 3b=Mib, 4b=Lab, 5b=Reb, 6b=Solb. Un error en la \
tonalitat invalida TOTA l'anàlisi. Compta cada sostingut/bemoll individualment.
- VERIFICACIÓ DE TONALITAT: Un cop identificada la tonalitat, comprova que els \
acords principals (sobretot al principi i al final) siguin coherents amb ella. \
Si l'últim acord és diferent de la tònica esperada, reconsidera si la tonalitat \
és correcta. Si la peça comença i acaba en un acord menor, podria ser en mode menor \
(relatiu menor de la tonalitat major de l'armadura).
- Si et proporcionen la TONALITAT a les dades d'entrada, utilitza-la com a referència \
fiable (prové del fitxer MusicXML original). Només qüestiona-la si les notes no hi \
encaixen gens.
- Usa SEMPRE notació llatina: Do Re Mi Fa Sol La Si (NO C D E F G A B)
- Sigues rigorós amb la identificació d'inversions — mira sempre el baix real
- No simplifiquis: si un acord té 7a, 9a, etc., indica-ho
- Quan hi hagi ambigüitat, explica les possibles interpretacions
- El resum pedagògic ha de ser útil per a un estudiant de piano intermedi
- Respon NOMÉS amb el JSON, sense text addicional
"""


def _deep_harmonic_analysis(mxml_data, notation="latin"):
    """Send parsed MusicXML data to Claude for deep harmonic analysis."""
    import anthropic
    from analyzer import _extract_json_from_response
    from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

    notes_text = _format_notes_for_claude(mxml_data, notation)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=64000,
        thinking={
            "type": "enabled",
            "budget_tokens": 32000,
        },
        system=HARMONIC_ANALYSIS_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Analitza harmònicament aquesta partitura de piano compàs per compàs.\n\n"
                f"{notes_text}"
            ),
        }],
    ) as stream:
        response = stream.get_final_message()

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text
            break
    if not response_text:
        raise ValueError("Resposta buida de Claude")

    return _extract_json_from_response(response_text)


def _detect_key_from_image(page_images):
    """Pass 1: ask Claude to identify ONLY the key signature from the score image."""
    import anthropic, json
    from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

    content = []
    b64 = base64.b64encode(page_images[0]).decode()
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    })
    content.append({
        "type": "text",
        "text": (
            "Mira NOMÉS l'armadura de clau (key signature) al principi del pentagrama "
            "d'aquesta partitura. NO facis cap altra anàlisi.\n\n"
            "1. Compta un per un els sostinguts (#) o bemolls (b) que hi apareixen "
            "ENTRE la clau (sol/fa) i el compàs.\n"
            "2. Mira si hi ha indicació de tonalitat menor (minor) o si pel context "
            "podria ser menor.\n\n"
            "Referència:\n"
            "- 0 alteracions = Do major / La menor\n"
            "- 1# (Fa#) = Sol major / Mi menor\n"
            "- 2# (Fa#, Do#) = Re major / Si menor\n"
            "- 3# (Fa#, Do#, Sol#) = La major / Fa# menor\n"
            "- 4# (Fa#, Do#, Sol#, Re#) = Mi major / Do# menor\n"
            "- 5# (Fa#, Do#, Sol#, Re#, La#) = Si major / Sol# menor\n"
            "- 1b (Sib) = Fa major / Re menor\n"
            "- 2b (Sib, Mib) = Sib major / Sol menor\n"
            "- 3b (Sib, Mib, Lab) = Mib major / Do menor\n"
            "- 4b (Sib, Mib, Lab, Reb) = Lab major / Fa menor\n"
            "- 5b (Sib, Mib, Lab, Reb, Solb) = Reb major / Sib menor\n\n"
            "Respon NOMÉS amb JSON:\n"
            '{"sharps_or_flats": <número, positiu=sostinguts, negatiu=bemolls>, '
            '"key": "<tonalitat, ex: Sol major>", '
            '"time_signature": "<compàs, ex: 4/4>", '
            '"confidence": "<high/medium/low>"}'
        ),
    })

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        thinking={"type": "enabled", "budget_tokens": 8000},
        messages=[{"role": "user", "content": content}],
    )

    for block in resp.content:
        if block.type == "text":
            text = block.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import re
                m = re.search(r'\{[^}]+\}', text)
                if m:
                    try:
                        return json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
    return None


def _deep_harmonic_analysis_vision(page_images, notation="latin"):
    """Two-pass PDF analysis: detect key first, then full harmonic analysis."""
    import anthropic
    from analyzer import _extract_json_from_response
    from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

    # ── Pass 1: detect key signature ──
    key_info = _detect_key_from_image(page_images)
    key_hint = ""
    if key_info:
        key_hint = (
            f"\n\nDETECCIÓ PRÈVIA DE L'ARMADURA: {key_info.get('key', '?')} "
            f"({key_info.get('sharps_or_flats', 0)} alteracions, "
            f"confiança: {key_info.get('confidence', '?')}). "
            f"Compàs detectat: {key_info.get('time_signature', '?')}. "
            f"Verifica-ho tu mateix mirant la partitura, però utilitza "
            f"aquesta detecció com a referència."
        )

    # ── Pass 2: full harmonic analysis ──
    content = []
    for i, png_bytes in enumerate(page_images):
        b64 = base64.b64encode(png_bytes).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
        if len(page_images) > 1:
            content.append({"type": "text", "text": f"(Pàgina {i + 1})"})

    notation_str = (
        "llatina (Do Re Mi Fa Sol La Si)"
        if notation == "latin"
        else "anglosaxona (C D E F G A B)"
    )
    content.append({
        "type": "text",
        "text": (
            f"Analitza harmònicament aquesta partitura de piano compàs per compàs. "
            f"Usa notació {notation_str}. "
            f"MOLT IMPORTANT: Abans de tot, examina amb molta cura l'armadura de clau "
            f"(key signature) al principi del pentagrama. Compta un per un els sostinguts "
            f"(#) o bemolls (b) que hi apareixen. Referència ràpida: "
            f"1#=Sol, 2#=Re, 3#=La, 4#=Mi, 5#=Si. "
            f"1b=Fa, 2b=Sib, 3b=Mib, 4b=Lab. "
            f"La tonalitat correcta és fonamental per a tota l'anàlisi."
            f"{key_hint}"
        ),
    })

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=64000,
        thinking={"type": "enabled", "budget_tokens": 32000},
        system=HARMONIC_ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        response = stream.get_final_message()

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text
            break
    if not response_text:
        raise ValueError("Resposta buida de Claude")

    return _extract_json_from_response(response_text)


@app.route("/harmonic-analysis", methods=["POST"])
def harmonic_analysis():
    """Deep harmonic analysis of a MusicXML or PDF file via Claude."""
    if "file" not in request.files:
        return jsonify({"error": "Cap fitxer enviat"}), 400

    file = request.files["file"]
    fname_lower = file.filename.lower()
    is_musicxml = (
        fname_lower.endswith(".xml")
        or fname_lower.endswith(".musicxml")
        or fname_lower.endswith(".mxl")
    )
    is_pdf = fname_lower.endswith(".pdf")

    if not is_musicxml and not is_pdf:
        return jsonify({"error": "Només fitxers MusicXML (.xml, .musicxml, .mxl) o PDF"}), 400

    notation = request.form.get("notation", "latin")
    job_id = uuid.uuid4().hex[:12]

    # --- PDF path: render pages → Claude Vision ---
    if is_pdf:
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_input.pdf")
        file.save(input_path)
        try:
            doc = fitz.open(input_path)
            pages_str = request.form.get("pages", "")
            total = len(doc)
            page_indices = parse_pages(pages_str, total) if pages_str else list(range(total))

            MAX_PAGES = 3
            if len(page_indices) > MAX_PAGES:
                page_indices = page_indices[:MAX_PAGES]

            ANALYSIS_DPI = 300
            images = []
            for idx in page_indices:
                images.append(render_page_to_png(doc[idx], dpi=ANALYSIS_DPI))
            doc.close()

            result = _deep_harmonic_analysis_vision(images, notation)
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error en l'anàlisi del PDF: {e}"}), 400
        finally:
            try:
                os.remove(input_path)
            except OSError:
                pass

    # --- MusicXML path ---
    import zipfile

    if fname_lower.endswith(".mxl"):
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

    measures_str = request.form.get("measures", "")
    if measures_str:
        total_m = mxml_data.total_measures
        m_indices = set()
        for part in measures_str.split(","):
            part = part.strip()
            if "-" in part:
                s, e = part.split("-", 1)
                for i in range(max(1, int(s)), min(total_m, int(e)) + 1):
                    m_indices.add(i - 1)
            else:
                idx = int(part) - 1
                if 0 <= idx < total_m:
                    m_indices.add(idx)
        if m_indices:
            mxml_data.measures = [m for m in mxml_data.measures if m.measure_index in m_indices]
            mxml_data.total_measures = len(mxml_data.measures)

    try:
        result = _deep_harmonic_analysis(mxml_data, notation)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error en l'anàlisi harmònica: {e}"}), 400
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass

    return jsonify(result)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5070
    print(f"Annotador d'Acords PDF server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

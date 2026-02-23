#!/usr/bin/env python3
"""Piano Chord Annotator — CLI entry point.

Usage:
    python main.py partitura.pdf [--notation anglo|latin] [--font-size auto|N] [--pages 1-5]
"""

import argparse
import sys

import fitz  # PyMuPDF

from analyzer import analyze_page, analyze_system, estimate_barlines
from config import PREFER_ALGORITHMIC
from note_extractor import detect_is_vector_music, extract_notes
from chord_identifier import analyze_page_chords
from pdf_writer import (
    render_page_to_png,
    render_system_to_png,
    detect_barlines,
    build_layout_from_estimates,
    annotate_page,
    save_annotated_pdf,
)


def parse_pages(pages_str: str, total: int) -> list[int]:
    """Parse a page range string like '1-5' or '2,4,6' into 0-based indices."""
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


def _log_analysis(analysis: dict) -> None:
    """Print a summary of the chord analysis."""
    for sys_a in analysis.get("systems", []):
        si = sys_a.get("system_index", "?")
        n_measures = len(sys_a.get("measures", []))
        chord_strs = []
        for m in sys_a.get("measures", []):
            c_list = m.get("chords", [])
            c_names = [c.get("chord", "?") for c in c_list]
            chord_strs.append(" ".join(c_names) if c_names else "-")
        print(f"  System {si}: {n_measures} measure(s)")
        print(f"    Chords: {' | '.join(chord_strs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate piano sheet music PDF with chord symbols."
    )
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument(
        "--notation",
        choices=["anglo", "latin"],
        default="anglo",
        help="Chord notation style (default: anglo)",
    )
    parser.add_argument(
        "--font-size",
        default="auto",
        help="Font size for chord labels: 'auto' or a number (default: auto)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Page range to process, e.g. '1-5' or '2,4,6' (default: all)",
    )

    args = parser.parse_args()

    # Parse font size
    font_size = args.font_size
    if font_size != "auto":
        try:
            font_size = float(font_size)
        except ValueError:
            print(f"Error: invalid font-size '{font_size}', using 'auto'")
            font_size = "auto"

    # Open the PDF
    try:
        doc = fitz.open(args.input)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    total_pages = len(doc)
    print(f"Opened '{args.input}' — {total_pages} page(s)")

    # Determine pages to process
    if args.pages:
        page_indices = parse_pages(args.pages, total_pages)
    else:
        page_indices = list(range(total_pages))

    print(f"Processing {len(page_indices)} page(s) with notation={args.notation}\n")

    for page_idx in page_indices:
        page = doc[page_idx]
        page_num_display = page_idx + 1
        print(f"--- Page {page_num_display}/{total_pages} ---")

        # Step 1: Detect barlines from vector graphics
        layout = detect_barlines(page)
        if layout.has_vector_barlines:
            total_measures = sum(s.measure_count for s in layout.systems)
            print(f"  Detected {len(layout.systems)} system(s), {total_measures} measure(s) from vectors")
        else:
            print("  No vector barlines found")

        # Step 2: Check if vector music (contains SMuFL noteheads)
        is_vector = PREFER_ALGORITHMIC and detect_is_vector_music(page)

        if is_vector:
            print("  Vector music detected — using algorithmic extraction")
        else:
            print("  Using Claude Vision analysis")

        # Step 3: If no vector barlines, estimate via Claude (needed for both paths)
        if not layout.has_vector_barlines:
            png_bytes = render_page_to_png(page)
            print("  Estimating barline positions via Claude...")
            try:
                estimates = estimate_barlines(png_bytes)
                layout = build_layout_from_estimates(page, estimates)
                total_measures = sum(s.measure_count for s in layout.systems)
                print(f"  Estimated {len(layout.systems)} system(s), {total_measures} measure(s)")
            except Exception as e:
                print(f"  Warning: barline estimation failed: {e}")
                print("  Skipping page (no layout information)")
                continue

        if not layout.systems:
            print("  No staff systems detected, skipping")
            continue

        # Step 4: Analyze chords — algorithmic or Claude Vision
        analysis = None

        if is_vector:
            # Algorithmic path: extract notes from PDF vectors
            try:
                note_data = extract_notes(page, layout)
                if note_data.has_music_glyphs:
                    analysis = analyze_page_chords(note_data, layout, page_idx)
                    # Log key signature
                    for gs in note_data.grand_staffs:
                        if gs.key_signature:
                            ks_str = ", ".join(f"{k}{'#' if v > 0 else 'b'}" for k, v in gs.key_signature.items())
                            print(f"  Key signature (system {gs.system_index}): {ks_str}")
                        break
                    print(f"  Extracted {sum(len(m.note_groups) for m in note_data.measures)} note groups")
                else:
                    print("  Not enough music glyphs — falling back to Claude Vision")
            except Exception as e:
                print(f"  Algorithmic extraction failed: {e}")
                print("  Falling back to Claude Vision")

        # Fallback to Claude Vision if algorithmic path didn't produce results
        if analysis is None:
            png_bytes = render_page_to_png(page) if not locals().get("png_bytes") else png_bytes
            num_systems = len(layout.systems)
            measures_per_system = [s.measure_count for s in layout.systems]
            print(f"  Analyzing chords via Claude...")
            try:
                analysis = analyze_page(
                    png_bytes,
                    page_number=page_idx,
                    num_systems_hint=num_systems,
                    measures_hint=measures_per_system,
                )
            except Exception as e:
                print(f"  Error analyzing page: {e}")
                continue

        # Step 5: Check if page has music
        if not analysis.get("has_music", True):
            print(f"  No music detected (notes: {analysis.get('notes', '')})")
            continue

        # Log analysis summary
        _log_analysis(analysis)

        # Warn on measure count mismatch
        for sys_layout in layout.systems:
            idx = layout.systems.index(sys_layout)
            analysis_systems = analysis.get("systems", [])
            if idx < len(analysis_systems):
                a_measures = len(analysis_systems[idx].get("measures", []))
                l_measures = sys_layout.measure_count
                if a_measures != l_measures:
                    print(f"  Warning: system {idx} layout has {l_measures} measures "
                          f"but analysis found {a_measures}")

        # Step 6: Annotate
        annotate_page(page, layout, analysis, notation=args.notation, font_size=font_size)
        print(f"  Annotated")

    # Step 7: Save
    output_path = save_annotated_pdf(doc, args.input)
    doc.close()
    print(f"\nSaved annotated PDF: {output_path}")


if __name__ == "__main__":
    main()

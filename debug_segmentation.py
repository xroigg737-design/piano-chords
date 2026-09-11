#!/usr/bin/env python3
"""Verifica visualment la segmentació de sistemes i compassos d'un PDF escanejat.

Ús:  python3 debug_segmentation.py partitura.pdf [carpeta_sortida]

Genera, per cada pàgina, un PNG amb els sistemes emmarcats i les barres de
compàs marcades, i escriu per consola quants compassos s'han detectat.
"""

import os
import sys

import fitz

from staff_segmenter import segment_systems, system_chunks


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    pdf_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_measures = 0
    for pno in range(len(doc)):
        page = doc[pno]
        systems = segment_systems(page)
        print(f"Pàgina {pno + 1}: {len(systems)} sistemes")
        for s in systems:
            chunks = system_chunks(s)
            total_measures += s.measure_count
            print(
                f"   sistema {s.index + 1}: y {s.y_top:.0f}-{s.y_bottom:.0f}pt, "
                f"{s.measure_count} compassos, {len(chunks)} retalls"
            )
            page.draw_rect(
                fitz.Rect(s.x_left, s.y_top, s.x_right, s.y_bottom),
                color=(1, 0, 0), width=0.8,
            )
            for x in s.barlines:
                page.draw_line(
                    fitz.Point(x, s.y_top - 4), fitz.Point(x, s.y_bottom + 4),
                    color=(0, 0.5, 1), width=1.2,
                )
            for x0, x1, first, n in chunks:
                page.draw_rect(
                    fitz.Rect(x0, s.y_top - 6, x1, s.y_bottom + 6),
                    color=(0, 0.7, 0), width=0.6,
                )
        out = os.path.join(out_dir, f"seg_p{pno + 1}.png")
        page.get_pixmap(dpi=120).save(out)
        print(f"   -> {out}")
    print(f"Total compassos detectats: {total_measures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

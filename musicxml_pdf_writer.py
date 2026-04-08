"""Generate a chord analysis PDF table from MusicXML results.

Creates a landscape A4 PDF with a table:
  C. | Acord | Grau | Ma esquerra | Ma dreta | Tipus | Baix
  + an inline legend box after the title.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from config import CHORD_COLOR_RH, ANGLO_TO_LATIN

CHORD_FONT = "hebo"  # Helvetica Bold


# Layout constants (points) -- A4 landscape
PAGE_W, PAGE_H = 841.89, 595.28
MARGIN_TOP = 50
MARGIN_BOTTOM = 35
MARGIN_LEFT = 40
MARGIN_RIGHT = 40
CONTENT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT

# Column widths (fractions of CONTENT_W)
# C. | Acord | Grau | Ma esquerra | Ma dreta | Tipus | Baix
COL_FRACS = [0.042, 0.105, 0.058, 0.195, 0.195, 0.22, 0.185]

# Font sizes
TITLE_SIZE = 17
KEY_SIZE = 13
INFO_SIZE = 9
HEADER_SIZE = 8.0
BODY_SIZE = 8.0
CHORD_SIZE = 9.5
DEGREE_SIZE = 8.0
NOTE_SIZE = 7.5
EXPLAIN_SIZE = 9.0
LEGEND_TITLE_SIZE = 8.5
LEGEND_SIZE = 6.5

# Row heights
HEADER_H = 18
ROW_H = 30

# Colors
COLOR_TITLE = (0.12, 0.12, 0.12)
COLOR_KEY = (0.10, 0.10, 0.10)
COLOR_INFO = (0.45, 0.45, 0.45)
COLOR_HEADER_BG = (0.18, 0.22, 0.35)
COLOR_HEADER_TEXT = (1.0, 1.0, 1.0)
COLOR_ROW_EVEN = (0.965, 0.965, 0.985)
COLOR_ROW_ODD = (1.0, 1.0, 1.0)
COLOR_GRID = (0.75, 0.75, 0.80)
COLOR_NUM = (0.35, 0.35, 0.40)
COLOR_CHORD = CHORD_COLOR_RH
COLOR_NC = (0.55, 0.55, 0.55)

# Note type colors -- chord tones are more vivid/intense
COLOR_NOTE_CHORD = (0.08, 0.28, 0.02)         # very dark green text
COLOR_NOTE_PASSING = (0.50, 0.50, 0.53)        # lighter gray text
COLOR_NOTE_CHROMATIC = (0.45, 0.28, 0.08)      # muted orange text

# Note type background fills
BG_NOTE_CHORD = (0.78, 0.91, 0.70)             # vivid green bg (more intense)
BG_NOTE_PASSING = (0.94, 0.94, 0.95)           # light gray bg (subtle)
BG_NOTE_CHROMATIC = (0.98, 0.93, 0.85)         # light orange bg

# Border colors for pills
BORDER_NOTE_CHORD = (0.30, 0.55, 0.18)         # visible green border
BORDER_NOTE_PASSING = (0.78, 0.78, 0.80)       # subtle gray border
BORDER_NOTE_CHROMATIC = (0.72, 0.52, 0.22)     # orange border

# Degree badge colors
DEGREE_COLORS = {
    "I":    ((0.933, 0.929, 0.996), (0.235, 0.204, 0.537)),
    "II":   ((0.902, 0.945, 0.984), (0.047, 0.267, 0.486)),
    "III":  ((0.902, 0.945, 0.984), (0.047, 0.267, 0.486)),
    "IV":   ((0.882, 0.961, 0.933), (0.031, 0.314, 0.255)),
    "V":    ((0.980, 0.933, 0.855), (0.388, 0.220, 0.024)),
    "VI":   ((0.980, 0.925, 0.906), (0.443, 0.106, 0.075)),
    "VII":  ((0.988, 0.922, 0.922), (0.475, 0.122, 0.122)),
}
DEFAULT_DEGREE_COLORS = ((0.93, 0.93, 0.95), (0.35, 0.35, 0.40))

# Column category colors (for Tipus and Baix text)
COLOR_TIPUS = (0.55, 0.12, 0.55)     # purple for quality
COLOR_BAIX = (0.71, 0.16, 0.16)      # red for bass


def _key_name(key_sharps: int, key_mode: str, notation: str) -> str:
    sharp_major = ["Do", "Sol", "Re", "La", "Mi", "Si", "Fa#", "Do#"]
    flat_major = ["Do", "Fa", "Sib", "Mib", "Lab", "Reb", "Solb", "Dob"]
    sharp_minor = ["La", "Mi", "Si", "Fa#", "Do#", "Sol#", "Re#", "La#"]
    flat_minor = ["La", "Re", "Sol", "Do", "Fa", "Sib", "Mib", "Lab"]

    if key_mode == "minor":
        if key_sharps >= 0:
            name = sharp_minor[min(key_sharps, 7)]
        else:
            name = flat_minor[min(-key_sharps, 7)]
    else:
        if key_sharps >= 0:
            name = sharp_major[min(key_sharps, 7)]
        else:
            name = flat_major[min(-key_sharps, 7)]

    if notation == "anglo":
        anglo_map = {"Do": "C", "Re": "D", "Mi": "E", "Fa": "F",
                     "Sol": "G", "La": "A", "Si": "B", "Sib": "Bb",
                     "Mib": "Eb", "Lab": "Ab", "Reb": "Db", "Solb": "Gb",
                     "Dob": "Cb", "Fa#": "F#", "Do#": "C#", "Sol#": "G#",
                     "Re#": "D#", "La#": "A#"}
        name = anglo_map.get(name, name)

    mode_str = "menor" if key_mode == "minor" else "major"
    if notation == "anglo":
        mode_str = "minor" if key_mode == "minor" else "major"

    return f"{name} {mode_str}"


def _convert_chord(chord: str, notation: str) -> str:
    if notation == "anglo" or not chord or chord == "N.C.":
        return chord
    from pdf_writer import convert_notation
    return convert_notation(chord, notation)


def _convert_note_name(name: str, notation: str) -> str:
    """Convert a note name like 'C#4' to latin notation (with octave)."""
    if not name:
        return name
    for i, ch in enumerate(name):
        if ch.isdigit() or (ch == '-' and i > 0):
            letter_part = name[:i]
            rest = name[i:]
            break
    else:
        letter_part = name
        rest = ""

    if notation == "anglo":
        return f"{letter_part}{rest}"

    base = letter_part[0]
    acc = letter_part[1:] if len(letter_part) > 1 else ""
    latin = ANGLO_TO_LATIN.get(base, base)
    acc_display = acc.replace("#", "\u266f").replace("b", "\u266d")
    return f"{latin}{acc_display}{rest}"


def _chord_full_name(chord: str, notation: str) -> str:
    """Convert chord symbol to full Catalan name like 'Sol Major'."""
    if not chord or chord in ("N.C.", "?"):
        return chord or "-"

    from chord_identifier import _parse_chord_symbol, _spell_root, CHORD_TEMPLATES

    slash = ""
    symbol_part = chord
    if "/" in chord:
        symbol_part, slash_note = chord.rsplit("/", 1)
        slash_display = _convert_note_name(slash_note + "0", notation)[:-1]
        slash = f" / {slash_display}"

    for root_pc in range(12):
        root_name = _spell_root(root_pc, 0)
        for _, quality in CHORD_TEMPLATES:
            if root_name + quality == symbol_part:
                root_display = _convert_note_name(root_name + "0", notation)[:-1]
                quality_names = {
                    "": "Major", "-": "menor", "o": "dim",
                    "+": "aug", "sus": "sus4", "sus2": "sus2",
                    "maj7": "Maj7", "7": "7", "-7": "m7",
                    "\u00f8": "\u00f8", "o7": "dim7",
                    "maj7#5": "Maj7#5", "7#5": "7#5",
                    "-maj7": "mMaj7", "6": "6", "-6": "m6",
                    "7#11": "7#11", "7b9": "7b9", "7b13": "7b13",
                    "7sus": "7sus4",
                }
                q_display = quality_names.get(quality, quality)
                return f"{root_display} {q_display}{slash}".strip()

    return _convert_chord(chord, notation)


def _col_x(col: int) -> float:
    x = MARGIN_LEFT
    for i in range(col):
        x += CONTENT_W * COL_FRACS[i]
    return x


def _col_w(col: int) -> float:
    return CONTENT_W * COL_FRACS[col]


def _text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * 0.48


def _truncate(text: str, max_w: float, font_size: float) -> str:
    char_w = font_size * 0.48
    max_chars = int(max_w / char_w)
    if len(text) <= max_chars:
        return text
    return text[:max(max_chars - 3, 1)] + "..."


def _extract_quality_text(explain_parts: list[tuple[str, str]]) -> str:
    """Extract quality/type description from explanation parts."""
    for text, category in explain_parts:
        if category == "qual":
            return text
    return ""


def _extract_bass_text(explain_parts: list[tuple[str, str]], inversion: str) -> tuple[str, str]:
    """Extract bass note and inversion as separate strings."""
    bass_note = ""
    for text, category in explain_parts:
        if category == "bass":
            bass_note = text.replace("Baix: ", "") if text.startswith("Baix: ") else text
            break
    # Build inversion line
    inv_line = ""
    if inversion and inversion != "fonamental":
        inv_line = inversion
    elif bass_note and "fonamental" in bass_note:
        inv_line = "fonamental"
        # Clean bass_note: remove "(fonamental)" suffix
        bass_note = bass_note.replace(" (fonamental)", "").strip()
    elif inversion == "fonamental":
        inv_line = "fonamental"

    return bass_note, inv_line


def _draw_header(page: fitz.Page, y: float) -> float:
    rect = fitz.Rect(MARGIN_LEFT, y, MARGIN_LEFT + CONTENT_W, y + HEADER_H)
    page.draw_rect(rect, color=None, fill=COLOR_HEADER_BG)

    headers = ["C.", "Acord", "Grau", "Ma esquerra", "Ma dreta", "Tipus", "Baix"]
    text_y = y + HEADER_H - 5.5

    for col, text in enumerate(headers):
        x = _col_x(col) + 4
        page.insert_text(
            fitz.Point(x, text_y),
            text,
            fontname=CHORD_FONT,
            fontsize=HEADER_SIZE,
            color=COLOR_HEADER_TEXT,
        )

    return y + HEADER_H


def _draw_note_pills(
    page: fitz.Page,
    x_start: float,
    text_y: float,
    notes: list[dict],
    max_w: float,
    notation: str,
) -> None:
    """Draw notes as colored pills. Chord tones are vivid, others are subtle."""
    x = x_start
    remaining_w = max_w
    pill_h = 9
    pill_pad = 4
    pill_gap = 2

    for nc in notes:
        name = _convert_note_name(nc["name"], notation)
        note_type = nc.get("type", "chord")

        if note_type == "chromatic":
            fg = COLOR_NOTE_CHROMATIC
            bg = BG_NOTE_CHROMATIC
            border = BORDER_NOTE_CHROMATIC
            border_w = 0.4
        elif note_type == "passing":
            fg = COLOR_NOTE_PASSING
            bg = BG_NOTE_PASSING
            border = BORDER_NOTE_PASSING
            border_w = 0.3
        else:
            # Chord tones: vivid, prominent
            fg = COLOR_NOTE_CHORD
            bg = BG_NOTE_CHORD
            border = BORDER_NOTE_CHORD
            border_w = 0.8

        tw = _text_width(name, NOTE_SIZE) + pill_pad * 2
        if tw + pill_gap > remaining_w:
            break

        pill_rect = fitz.Rect(
            x, text_y - pill_h + 1,
            x + tw, text_y + 2,
        )
        page.draw_rect(pill_rect, color=border, fill=bg, width=border_w)

        page.insert_text(
            fitz.Point(x + pill_pad, text_y - 0.5),
            name,
            fontname=CHORD_FONT,
            fontsize=NOTE_SIZE,
            color=fg,
        )

        x += tw + pill_gap
        remaining_w -= tw + pill_gap


def _draw_degree_badge(
    page: fitz.Page,
    x: float,
    text_y: float,
    degree: str,
) -> None:
    """Draw a Roman numeral degree badge."""
    if not degree:
        return

    base = degree.lstrip("#b")
    for key in ("VII", "VI", "IV", "V", "III", "II", "I"):
        if base.upper().startswith(key):
            bg_color, fg_color = DEGREE_COLORS.get(key, DEFAULT_DEGREE_COLORS)
            break
    else:
        bg_color, fg_color = DEFAULT_DEGREE_COLORS

    tw = _text_width(degree, DEGREE_SIZE) + 10
    badge_h = 12
    badge_rect = fitz.Rect(x, text_y - badge_h + 2, x + tw, text_y + 3)
    page.draw_rect(badge_rect, color=None, fill=bg_color)

    page.insert_text(
        fitz.Point(x + 5, text_y),
        degree,
        fontname=CHORD_FONT,
        fontsize=DEGREE_SIZE,
        color=fg_color,
    )


def _draw_row(
    page: fitz.Page,
    y: float,
    measure_num: int,
    chord_str: str,
    degree: str,
    lh_notes: list[dict],
    rh_notes: list[dict],
    quality_text: str,
    bass_note: str,
    bass_inv: str,
    is_even: bool,
    notation: str = "latin",
) -> float:
    """Draw one table row. Returns Y after row."""
    bg = COLOR_ROW_EVEN if is_even else COLOR_ROW_ODD

    rect = fitz.Rect(MARGIN_LEFT, y, MARGIN_LEFT + CONTENT_W, y + ROW_H)
    page.draw_rect(rect, color=None, fill=bg)

    text_y_top = y + 11

    # Column 0: measure number (centered)
    num_text = str(measure_num)
    num_w = _text_width(num_text, BODY_SIZE)
    x_num = _col_x(0) + (_col_w(0) - num_w) / 2
    page.insert_text(
        fitz.Point(x_num, text_y_top + 3),
        num_text,
        fontname=CHORD_FONT,
        fontsize=BODY_SIZE,
        color=COLOR_NUM,
    )

    # Column 1: chord name (full name)
    chord_display = _chord_full_name(chord_str, notation)
    chord_color = COLOR_NC if chord_str in ("N.C.", "?", "") else COLOR_CHORD
    chord_display_trunc = _truncate(chord_display, _col_w(1) - 8, CHORD_SIZE)
    page.insert_text(
        fitz.Point(_col_x(1) + 4, text_y_top + 3),
        chord_display_trunc,
        fontname=CHORD_FONT,
        fontsize=CHORD_SIZE,
        color=chord_color,
    )

    # Column 2: degree badge
    _draw_degree_badge(page, _col_x(2) + 4, text_y_top + 3, degree)

    # Column 3: left hand notes (bass staff)
    if lh_notes:
        _draw_note_pills(page, _col_x(3) + 3, text_y_top, lh_notes, _col_w(3) - 6, notation)
    else:
        page.insert_text(
            fitz.Point(_col_x(3) + 4, text_y_top),
            "-",
            fontname=CHORD_FONT, fontsize=NOTE_SIZE, color=COLOR_NC,
        )

    # Column 4: right hand notes (treble staff)
    if rh_notes:
        _draw_note_pills(page, _col_x(4) + 3, text_y_top, rh_notes, _col_w(4) - 6, notation)
    else:
        page.insert_text(
            fitz.Point(_col_x(4) + 4, text_y_top),
            "-",
            fontname=CHORD_FONT, fontsize=NOTE_SIZE, color=COLOR_NC,
        )

    # Column 5: Tipus (quality)
    if quality_text:
        qt = _truncate(quality_text, _col_w(5) - 8, EXPLAIN_SIZE)
        page.insert_text(
            fitz.Point(_col_x(5) + 4, text_y_top),
            qt,
            fontname=CHORD_FONT,
            fontsize=EXPLAIN_SIZE,
            color=COLOR_TIPUS,
        )

    # Column 6: Baix (bass note on line 1, inversion on line 2)
    if bass_note:
        bn = _truncate(bass_note, _col_w(6) - 8, EXPLAIN_SIZE)
        page.insert_text(
            fitz.Point(_col_x(6) + 4, text_y_top),
            bn,
            fontname=CHORD_FONT,
            fontsize=EXPLAIN_SIZE,
            color=COLOR_BAIX,
        )
    if bass_inv:
        inv_color = (0.10, 0.30, 0.68) if "inversio" in bass_inv else (0.40, 0.40, 0.45)
        bi = _truncate(bass_inv, _col_w(6) - 8, 7.5)
        page.insert_text(
            fitz.Point(_col_x(6) + 4, text_y_top + 12),
            bi,
            fontname="helv",
            fontsize=7.5,
            color=inv_color,
        )

    # Vertical grid lines
    for col in range(1, len(COL_FRACS)):
        x = _col_x(col)
        page.draw_line(
            fitz.Point(x, y), fitz.Point(x, y + ROW_H),
            color=COLOR_GRID, width=0.3,
        )

    return y + ROW_H


def _draw_inline_legend(page: fitz.Page, y: float) -> float:
    """Draw a compact legend box inline on page 1. Returns Y after the box."""
    box_x = MARGIN_LEFT
    box_w = CONTENT_W
    pad_x = 10
    pad_y = 7
    row_h = 13
    desc_color = (0.30, 0.30, 0.35)
    lbl_size = 7.0
    desc_size = 6.5
    section_title_size = 7.5

    # Two-column layout: left half = notes + columns, right half = degrees
    mid_x = box_x + box_w * 0.55

    # Height: 4 rows on left side (notes header + 3 items), right side fits degrees
    box_h = pad_y + row_h * 5 + pad_y

    # Draw box
    box_rect = fitz.Rect(box_x, y, box_x + box_w, y + box_h)
    page.draw_rect(box_rect, color=(0.78, 0.78, 0.85), fill=(0.975, 0.975, 0.99), width=0.4)

    # Vertical divider
    page.draw_line(
        fitz.Point(mid_x - 8, y + pad_y),
        fitz.Point(mid_x - 8, y + box_h - pad_y),
        color=(0.82, 0.82, 0.88), width=0.3,
    )

    cy = y + pad_y

    # ===== LEFT COLUMN: Notes + Columnes =====

    # Section: Notes
    page.insert_text(
        fitz.Point(box_x + pad_x, cy + 7),
        "Notes:",
        fontname=CHORD_FONT, fontsize=section_title_size, color=COLOR_TITLE,
    )
    cy += row_h

    note_items = [
        (BG_NOTE_CHORD, BORDER_NOTE_CHORD, COLOR_NOTE_CHORD,
         "Acord", "Nota que forma part de l'acord (fonamental, 3a, 5a, 7a...)"),
        (BG_NOTE_PASSING, BORDER_NOTE_PASSING, COLOR_NOTE_PASSING,
         "Pas", "Nota diatonica fora de l'acord (nota de pas, bordadura, retard...)"),
        (BG_NOTE_CHROMATIC, BORDER_NOTE_CHROMATIC, COLOR_NOTE_CHROMATIC,
         "Crom.", "Nota fora de l'escala (alteracio cromatica, dominanta secundaria...)"),
    ]

    for bg, border, fg, label, desc in note_items:
        nx = box_x + pad_x
        # Pill swatch
        sw = _text_width(label, lbl_size) + 8
        pill_rect = fitz.Rect(nx, cy, nx + sw, cy + 9)
        page.draw_rect(pill_rect, color=border, fill=bg, width=0.5)
        page.insert_text(
            fitz.Point(nx + 4, cy + 7),
            label,
            fontname=CHORD_FONT, fontsize=lbl_size, color=fg,
        )
        nx += sw + 5
        page.insert_text(
            fitz.Point(nx, cy + 7),
            desc,
            fontname="helv", fontsize=desc_size, color=desc_color,
        )
        cy += row_h

    # Section: Columnes (Tipus + Baix on one line)
    page.insert_text(
        fitz.Point(box_x + pad_x, cy + 7),
        "Columnes:",
        fontname=CHORD_FONT, fontsize=section_title_size, color=COLOR_TITLE,
    )
    cx = box_x + pad_x + _text_width("Columnes:", section_title_size) + 6

    page.insert_text(fitz.Point(cx, cy + 7), "Tipus", fontname=CHORD_FONT, fontsize=lbl_size, color=COLOR_TIPUS)
    cx += _text_width("Tipus", lbl_size) + 3
    page.insert_text(fitz.Point(cx, cy + 7), "= qualitat de l'acord", fontname="helv", fontsize=desc_size, color=desc_color)
    cx += _text_width("= qualitat de l'acord", desc_size) + 10

    page.insert_text(fitz.Point(cx, cy + 7), "Baix", fontname=CHORD_FONT, fontsize=lbl_size, color=COLOR_BAIX)
    cx += _text_width("Baix", lbl_size) + 3
    page.insert_text(fitz.Point(cx, cy + 7), "= nota greu + inversio (fonamental, 1a, 2a...)", fontname="helv", fontsize=desc_size, color=desc_color)

    # ===== RIGHT COLUMN: Graus =====

    ry = y + pad_y
    page.insert_text(
        fitz.Point(mid_x, ry + 7),
        "Graus tonals:",
        fontname=CHORD_FONT, fontsize=section_title_size, color=COLOR_TITLE,
    )
    ry += row_h

    degree_items = [
        ("I", "Tonica"),
        ("II", "Supertonica"),
        ("III", "Mitjana (o Modal)"),
        ("IV", "Subdominant"),
        ("V", "Dominant"),
        ("VI", "Superdominant"),
        ("VII", "Sensible"),
    ]

    # Render in two columns (4 left, 3 right)
    col1 = degree_items[:4]
    col2 = degree_items[4:]

    for i, (deg, desc) in enumerate(col1):
        dx = mid_x
        _draw_degree_badge(page, dx, ry + 7, deg)
        tw = _text_width(deg, DEGREE_SIZE) + 14
        page.insert_text(
            fitz.Point(dx + tw, ry + 7),
            f"= {desc}",
            fontname="helv", fontsize=desc_size, color=desc_color,
        )
        ry += row_h

    ry2 = y + pad_y + row_h  # start after title
    right_col_x = mid_x + (box_w * 0.45) / 2
    for deg, desc in col2:
        dx = right_col_x
        _draw_degree_badge(page, dx, ry2 + 7, deg)
        tw = _text_width(deg, DEGREE_SIZE) + 14
        page.insert_text(
            fitz.Point(dx + tw, ry2 + 7),
            f"= {desc}",
            fontname="helv", fontsize=desc_size, color=desc_color,
        )
        ry2 += row_h

    return y + box_h


def generate_chord_chart_pdf(
    title: str,
    measures_data: list[dict],
    key_sharps: int,
    key_mode: str,
    time_num: int,
    time_den: int,
    total_measures: int,
    notation: str = "latin",
    output_path: str = "output.pdf",
) -> str:
    """Generate a landscape PDF with chord analysis table."""
    doc = fitz.open()

    # Pre-process measure data
    rows = []
    for m in measures_data:
        mi = m["measure_index"]

        chords_list = m.get("chords", [])
        if chords_list:
            chord_str = chords_list[0].get("chord", "")
        else:
            chord_str = ""

        rh_notes = m.get("rh_notes", [])
        lh_notes = m.get("lh_notes", [])

        if not rh_notes and not lh_notes and "notes_classified" in m:
            rh_notes = [{"name": n["name"], "type": "chord" if n.get("is_chord_tone", True) else "passing"}
                        for n in m["notes_classified"]]

        degree = m.get("degree", "")
        inversion = m.get("inversion", "")

        explanation = m.get("explanation", [])
        if isinstance(explanation, str):
            explanation = [(explanation, "nc")] if explanation else []

        # Extract quality and bass from explanation parts
        quality_text = _extract_quality_text(explanation)
        bass_note, bass_inv = _extract_bass_text(explanation, inversion)

        rows.append((mi + 1, chord_str, degree, lh_notes, rh_notes, quality_text, bass_note, bass_inv))

    # Skip empty measures (all rests)
    rows = [r for r in rows if r[3] or r[4]]

    # Paginate
    page = None
    y = 0.0
    row_idx = 0
    last_table_bottom_y = 0.0

    for measure_num, chord_str, degree, lh_notes, rh_notes, quality_text, bass_note, bass_inv in rows:
        title_block = 0
        if page is None:
            title_block = 130  # title + key + legend box

        needed = title_block + HEADER_H + ROW_H if page is None else ROW_H

        if page is None or y + needed > PAGE_H - MARGIN_BOTTOM:
            if page is not None:
                page.draw_line(
                    fitz.Point(MARGIN_LEFT, last_table_bottom_y),
                    fitz.Point(MARGIN_LEFT + CONTENT_W, last_table_bottom_y),
                    color=COLOR_GRID, width=0.5,
                )

            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN_TOP

            if doc.page_count == 1:
                if title:
                    page.insert_text(
                        fitz.Point(MARGIN_LEFT, y),
                        title,
                        fontname=CHORD_FONT,
                        fontsize=TITLE_SIZE,
                        color=COLOR_TITLE,
                    )
                    y += TITLE_SIZE + 5

                key_str = _key_name(key_sharps, key_mode, notation)
                key_text_w = _text_width(key_str, KEY_SIZE)
                page.insert_text(
                    fitz.Point(MARGIN_LEFT, y),
                    key_str,
                    fontname=CHORD_FONT,
                    fontsize=KEY_SIZE,
                    color=COLOR_KEY,
                )

                info = f"     {time_num}/{time_den}    {total_measures} compassos"
                page.insert_text(
                    fitz.Point(MARGIN_LEFT + key_text_w + 4, y),
                    info,
                    fontname=CHORD_FONT,
                    fontsize=INFO_SIZE,
                    color=COLOR_INFO,
                )
                y += KEY_SIZE + 8

                # Draw inline legend
                y = _draw_inline_legend(page, y)
                y += 6

            y = _draw_header(page, y)

        is_even = row_idx % 2 == 0
        y = _draw_row(
            page, y, measure_num, chord_str, degree,
            lh_notes, rh_notes, quality_text, bass_note, bass_inv,
            is_even, notation=notation,
        )
        last_table_bottom_y = y
        row_idx += 1

    # Bottom border on last page
    if page is not None and rows:
        page.draw_line(
            fitz.Point(MARGIN_LEFT, last_table_bottom_y),
            fitz.Point(MARGIN_LEFT + CONTENT_W, last_table_bottom_y),
            color=COLOR_GRID, width=0.5,
        )

    if not rows:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text(
            fitz.Point(MARGIN_LEFT, MARGIN_TOP + 30),
            "No s'han trobat compassos amb notes al fitxer MusicXML.",
            fontname=CHORD_FONT, fontsize=12,
            color=(0.5, 0.2, 0.2),
        )

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return output_path

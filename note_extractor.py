"""Extract musical notes from vector PDF pages (MuseScore / SMuFL fonts).

Reads glyph positions from the PDF text layer and computes exact pitches
by relating each notehead's Y coordinate to the detected staff lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import fitz  # PyMuPDF

from config import (
    STAFF_LINE_MIN_LENGTH,
    STAFF_LINE_MAX_DY,
    STAFF_SPACING_MIN,
    STAFF_SPACING_MAX,
    STAFF_SPACING_TOLERANCE,
    NOTE_X_GROUP_TOLERANCE,
    ACCIDENTAL_MAX_DX,
    ACCIDENTAL_MAX_DY,
    MIN_NOTEHEADS_FOR_MUSIC,
)
from pdf_writer import PageLayout, StaffSystem


# ---------------------------------------------------------------------------
# SMuFL glyph constants
# ---------------------------------------------------------------------------

NOTEHEAD_CODEPOINTS = {
    0xE0A4: "filled",  # noteheadBlack
    0xE0A3: "half",    # noteheadHalf
    0xE0A2: "whole",   # noteheadWhole
}

ACCIDENTAL_CODEPOINTS = {
    0xE260: -1,  # flat
    0xE261: 0,   # natural (cancels key sig)
    0xE262: 1,   # sharp
    0xE263: 2,   # double sharp
    0xE264: -2,  # double flat
}

CLEF_CODEPOINTS = {
    0xE050: "treble",  # gClef
    0xE062: "bass",    # fClef
}

# Key-signature accidental codepoints (same values, used for filtering)
KEY_SIG_ACCIDENTAL_CODEPOINTS = {0xE260, 0xE261, 0xE262}

NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
MUSIC_FONTS = {"Leland", "LelandText", "MScore", "BravuraText", "Bravura", "Doremi"}

# Doremi font glyph mappings (non-SMuFL music font)
DOREMI_NOTEHEAD_CODEPOINTS = {
    0x00CF: "filled",  # Ï — black notehead
    0x00FA: "half",    # ú — half notehead
    0x00CE: "whole",   # Î — whole notehead
}
DOREMI_ACCIDENTAL_CODEPOINTS = {
    0x0023: 1,   # # — sharp
    0x0062: -1,  # b — flat (lowercase b in Doremi font)
}
DOREMI_CLEF_CODEPOINTS = {
    0x0047: "treble",  # G — treble clef
    0x0046: "bass",    # F — bass clef
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StaffLines:
    """Five staff lines for a single staff (treble or bass)."""
    y_positions: list[float]  # 5 values, top to bottom
    clef: str = "treble"

    @property
    def half_space(self) -> float:
        return (self.y_positions[4] - self.y_positions[0]) / 8

    @property
    def top_line_y(self) -> float:
        return self.y_positions[0]

    @property
    def bottom_line_y(self) -> float:
        return self.y_positions[4]


@dataclass
class GrandStaff:
    """A treble + bass staff pair forming one system."""
    system_index: int
    treble: StaffLines
    bass: StaffLines
    key_signature: dict[str, int] = field(default_factory=dict)

    @property
    def gap_midpoint(self) -> float:
        return (self.treble.bottom_line_y + self.bass.top_line_y) / 2


@dataclass
class ExtractedNote:
    x: float
    y: float
    pitch_name: str       # 'C'..'B'
    octave: int
    accidental: int       # -2..+2
    staff: str            # 'treble' or 'bass'
    system_index: int
    notehead_type: str    # 'filled', 'half', 'whole'

    @property
    def midi_pitch(self) -> int:
        semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        return (self.octave + 1) * 12 + semitones[self.pitch_name] + self.accidental

    @property
    def pitch_class(self) -> int:
        return self.midi_pitch % 12

    @property
    def full_name(self) -> str:
        acc_str = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
        return f"{self.pitch_name}{acc_str[self.accidental]}{self.octave}"


@dataclass
class NoteGroup:
    """Notes sounding simultaneously (same X position)."""
    x: float
    notes: list[ExtractedNote]
    measure_index: int
    system_index: int


@dataclass
class MeasureNotes:
    measure_index: int
    system_index: int
    note_groups: list[NoteGroup]

    @property
    def all_pitch_classes(self) -> set[int]:
        return {n.pitch_class for ng in self.note_groups for n in ng.notes}

    @property
    def bass_note(self) -> ExtractedNote | None:
        if not self.note_groups:
            return None
        first_group = self.note_groups[0]
        if not first_group.notes:
            return None
        return min(first_group.notes, key=lambda n: n.midi_pitch)


@dataclass
class PageNoteData:
    page_number: int
    grand_staffs: list[GrandStaff]
    measures: list[MeasureNotes]
    has_music_glyphs: bool


# ---------------------------------------------------------------------------
# Step 1: Detect staff lines
# ---------------------------------------------------------------------------

def _detect_staff_lines(page: fitz.Page) -> list[list[float]]:
    """Find groups of 5 horizontal lines with consistent ~5pt spacing.

    Uses a spacing-based search that tolerates noise lines (beams, ledger lines)
    between staff lines.

    Returns a list of staff groups, each being 5 Y values (top to bottom).
    """
    drawings = page.get_drawings()
    y_values: list[float] = []

    for d in drawings:
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                dy = abs(p1.y - p2.y)
                dx = abs(p1.x - p2.x)
                if dy < STAFF_LINE_MAX_DY and dx > STAFF_LINE_MIN_LENGTH:
                    y_values.append((p1.y + p2.y) / 2)

    if not y_values:
        return []

    # Deduplicate Y values (within 0.5pt)
    y_values.sort()
    unique_ys: list[float] = [y_values[0]]
    for y in y_values[1:]:
        if y - unique_ys[-1] > 0.5:
            unique_ys.append(y)
        else:
            unique_ys[-1] = (unique_ys[-1] + y) / 2

    # For each candidate starting Y, find 4 more Y values at ~1x, 2x, 3x, 4x spacing.
    # This tolerates noise lines (beams, ledger lines) between staff lines.
    staff_groups: list[list[float]] = []
    used: set[int] = set()

    def _find_nearest(target_y: float, tolerance: float) -> int | None:
        """Find index of nearest unused Y within tolerance."""
        best_idx = None
        best_dist = tolerance
        for idx, y in enumerate(unique_ys):
            if idx in used:
                continue
            dist = abs(y - target_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    for i, y0 in enumerate(unique_ys):
        if i in used:
            continue
        # Build candidate spacings: fixed common values + adaptive from nearby Y
        spacing_candidates = [4.97, 5.0, 4.5, 5.5, 4.0, 4.125, 4.25]
        # Also try spacing derived from the next unused Y
        for j in range(i + 1, min(i + 6, len(unique_ys))):
            if j not in used:
                derived = unique_ys[j] - y0
                if STAFF_SPACING_MIN - 0.2 <= derived <= STAFF_SPACING_MAX + 0.2:
                    spacing_candidates.append(derived)
                break
        for spacing_candidate in spacing_candidates:
            if not (STAFF_SPACING_MIN - 0.2 <= spacing_candidate <= STAFF_SPACING_MAX + 0.2):
                continue
            indices = [i]
            ok = True
            for k in range(1, 5):
                target = y0 + k * spacing_candidate
                idx = _find_nearest(target, STAFF_SPACING_TOLERANCE + 0.3)
                if idx is None or idx in set(indices):
                    ok = False
                    break
                indices.append(idx)
            if ok and len(indices) == 5:
                group = [unique_ys[idx] for idx in indices]
                # Verify spacings
                spacings = [group[j + 1] - group[j] for j in range(4)]
                variance = max(spacings) - min(spacings)
                avg_sp = sum(spacings) / 4
                if (STAFF_SPACING_MIN - 0.2 <= avg_sp <= STAFF_SPACING_MAX + 0.2
                        and variance < STAFF_SPACING_TOLERANCE + 0.3):
                    staff_groups.append(group)
                    for idx in indices:
                        used.add(idx)
                    break

    # Sort groups by top Y
    staff_groups.sort(key=lambda g: g[0])
    return staff_groups


# ---------------------------------------------------------------------------
# Step 2: Detect clefs
# ---------------------------------------------------------------------------

def _is_doremi_font(font_name: str) -> bool:
    return "Doremi" in font_name


def _normalize_codepoint(cp: int, font_name: str) -> int:
    """Translate font-specific codepoints to SMuFL equivalents."""
    if not _is_doremi_font(font_name):
        return cp
    # Map Doremi noteheads to SMuFL
    if cp in DOREMI_NOTEHEAD_CODEPOINTS:
        ntype = DOREMI_NOTEHEAD_CODEPOINTS[cp]
        for smufl_cp, stype in NOTEHEAD_CODEPOINTS.items():
            if stype == ntype:
                return smufl_cp
    # Map Doremi clefs to SMuFL
    if cp in DOREMI_CLEF_CODEPOINTS:
        ctype = DOREMI_CLEF_CODEPOINTS[cp]
        for smufl_cp, stype in CLEF_CODEPOINTS.items():
            if stype == ctype:
                return smufl_cp
    # Map Doremi accidentals to SMuFL
    if cp in DOREMI_ACCIDENTAL_CODEPOINTS:
        aval = DOREMI_ACCIDENTAL_CODEPOINTS[cp]
        for smufl_cp, sval in ACCIDENTAL_CODEPOINTS.items():
            if sval == aval:
                return smufl_cp
    return cp


def _iter_music_chars(page: fitz.Page):
    """Yield (x, y, codepoint, font_name) for all music-font characters.

    Codepoints from non-SMuFL fonts (e.g. Doremi) are normalised to their
    SMuFL equivalents so that downstream code needs only one set of constants.
    """
    rawdict = page.get_text("rawdict")
    for block in rawdict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                if not any(mf in font for mf in MUSIC_FONTS):
                    continue
                for ch in span.get("chars", []):
                    cp = ord(ch["c"]) if len(ch["c"]) == 1 else 0
                    cp = _normalize_codepoint(cp, font)
                    origin = ch["origin"]
                    yield origin[0], origin[1], cp, font


def _detect_clefs(page: fitz.Page, staff_groups: list[list[float]]) -> dict[int, str]:
    """Map staff_group index -> clef type ('treble' or 'bass').

    Scans for clef glyphs and matches each to the nearest staff group.
    """
    clef_map: dict[int, str] = {}

    for x, y, cp, _font in _iter_music_chars(page):
        if cp not in CLEF_CODEPOINTS:
            continue
        clef_type = CLEF_CODEPOINTS[cp]
        # Find nearest staff group
        best_idx = -1
        best_dist = float("inf")
        for idx, group in enumerate(staff_groups):
            top, bottom = group[0], group[4]
            if top - 10 <= y <= bottom + 10:
                dist = abs(y - (top + bottom) / 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx >= 0:
            clef_map[best_idx] = clef_type

    return clef_map


# ---------------------------------------------------------------------------
# Step 3: Pair staves into grand staffs
# ---------------------------------------------------------------------------

def _pair_staves(
    staff_groups: list[list[float]],
    clef_map: dict[int, str],
    layout: PageLayout,
) -> list[GrandStaff]:
    """Pair treble + bass staves into GrandStaffs using layout systems."""
    grand_staffs: list[GrandStaff] = []

    for sys_layout in layout.systems:
        y_top = sys_layout.y_top
        y_bottom = sys_layout.y_bottom

        treble_idx = None
        bass_idx = None

        for idx, group in enumerate(staff_groups):
            group_top = group[0]
            group_bottom = group[4]
            # Treble staff: its top line should be near system y_top
            if abs(group_top - y_top) < 15:
                treble_idx = idx
            # Bass staff: its top line should be near system y_bottom
            # (y_bottom in StaffSystem = bass staff top line)
            elif abs(group_top - y_bottom) < 15:
                bass_idx = idx

        if treble_idx is None or bass_idx is None:
            # Fallback: find by proximity
            candidates = []
            for idx, group in enumerate(staff_groups):
                mid = (group[0] + group[4]) / 2
                if y_top - 20 <= mid <= y_bottom + 40:
                    candidates.append(idx)
            if len(candidates) >= 2:
                candidates.sort(key=lambda i: staff_groups[i][0])
                treble_idx = candidates[0]
                bass_idx = candidates[1]

        if treble_idx is not None and bass_idx is not None:
            treble_clef = clef_map.get(treble_idx, "treble")
            bass_clef = clef_map.get(bass_idx, "bass")
            grand_staffs.append(GrandStaff(
                system_index=sys_layout.system_index,
                treble=StaffLines(y_positions=staff_groups[treble_idx], clef=treble_clef),
                bass=StaffLines(y_positions=staff_groups[bass_idx], clef=bass_clef),
            ))

    return grand_staffs


# ---------------------------------------------------------------------------
# Step 4: Detect key signature
# ---------------------------------------------------------------------------

# Standard key signature positions (half-spaces from bottom line)
# Sharp order: F C G D A E B
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
# Flat order: B E A D G C F
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


def _y_to_diatonic(y: float, staff: StaffLines) -> tuple[str, int]:
    """Convert a Y coordinate to (note_name, octave) on the given staff."""
    half_spaces = round((staff.bottom_line_y - y) / staff.half_space)
    if staff.clef == "treble":
        # Bottom line of treble clef = E4
        abs_step = 2 + half_spaces  # E=2 in NOTE_NAMES
        octave_base = 4
    else:
        # Bottom line of bass clef = G2
        abs_step = 4 + half_spaces  # G=4 in NOTE_NAMES
        octave_base = 2

    note_name = NOTE_NAMES[abs_step % 7]
    octave = octave_base + abs_step // 7
    return note_name, octave


def _detect_key_signature(
    page: fitz.Page,
    grand_staff: GrandStaff,
    barlines: list,
) -> dict[str, int]:
    """Detect key signature accidentals between clef and first measure barline.

    Returns dict mapping note name to accidental value, e.g. {'F': 1} for G major.
    """
    key_sig: dict[str, int] = {}

    # Find the treble clef x for this system
    clef_x = 0.0
    for x, y, cp, _font in _iter_music_chars(page):
        if cp in CLEF_CODEPOINTS:
            staff = grand_staff.treble
            if staff.top_line_y - 10 <= y <= staff.bottom_line_y + 10:
                clef_x = max(clef_x, x)

    if clef_x == 0:
        return key_sig

    # Find the first barline that's AFTER the clef (skip system start barline)
    first_bl_after_clef = 0.0
    for bl in barlines:
        if bl.x > clef_x + 5:
            first_bl_after_clef = bl.x
            break

    if first_bl_after_clef == 0:
        return key_sig

    # Key sig accidentals are clustered right after the clef, typically within
    # 30pt.  Cap the range to avoid picking up in-measure accidentals.
    right_boundary = min(first_bl_after_clef, clef_x + 40)

    # Search for accidentals between clef and first measure barline
    for x, y, cp, _font in _iter_music_chars(page):
        if cp not in ACCIDENTAL_CODEPOINTS:
            continue
        if not (clef_x < x < right_boundary - 5):
            continue
        # Must be on the treble staff
        staff = grand_staff.treble
        if not (staff.top_line_y - 15 <= y <= staff.bottom_line_y + 15):
            continue
        note_name, _octave = _y_to_diatonic(y, staff)
        acc_val = ACCIDENTAL_CODEPOINTS[cp]
        if acc_val != 0:  # naturals in key sig are unusual
            key_sig[note_name] = acc_val

    return key_sig


# ---------------------------------------------------------------------------
# Step 5: Extract raw noteheads
# ---------------------------------------------------------------------------

def _extract_raw_noteheads(page: fitz.Page) -> list[tuple[float, float, str]]:
    """Extract notehead positions: list of (x, y, type)."""
    noteheads: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float]] = set()

    for x, y, cp, _font in _iter_music_chars(page):
        if cp not in NOTEHEAD_CODEPOINTS:
            continue
        # Deduplicate within 1pt
        key = (round(x * 2) / 2, round(y * 2) / 2)
        if key in seen:
            continue
        seen.add(key)
        noteheads.append((x, y, NOTEHEAD_CODEPOINTS[cp]))

    return noteheads


# ---------------------------------------------------------------------------
# Step 6: Extract raw accidentals
# ---------------------------------------------------------------------------

def _extract_raw_accidentals(page: fitz.Page) -> list[tuple[float, float, int]]:
    """Extract accidental positions: list of (x, y, accidental_value)."""
    accidentals: list[tuple[float, float, int]] = []
    seen: set[tuple[float, float]] = set()

    for x, y, cp, _font in _iter_music_chars(page):
        if cp not in ACCIDENTAL_CODEPOINTS:
            continue
        key = (round(x * 2) / 2, round(y * 2) / 2)
        if key in seen:
            continue
        seen.add(key)
        accidentals.append((x, y, ACCIDENTAL_CODEPOINTS[cp]))

    return accidentals


# ---------------------------------------------------------------------------
# Step 7: Assign notes to staves and compute pitch
# ---------------------------------------------------------------------------

def _assign_notes_to_staves(
    noteheads: list[tuple[float, float, str]],
    grand_staffs: list[GrandStaff],
) -> list[ExtractedNote]:
    """Assign each notehead to a staff and compute its diatonic pitch."""
    notes: list[ExtractedNote] = []

    for nx, ny, ntype in noteheads:
        best_gs: GrandStaff | None = None
        best_dist = float("inf")

        for gs in grand_staffs:
            top = gs.treble.top_line_y - 30
            bottom = gs.bass.bottom_line_y + 30
            if top <= ny <= bottom:
                mid = (gs.treble.top_line_y + gs.bass.bottom_line_y) / 2
                dist = abs(ny - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_gs = gs

        if best_gs is None:
            continue

        # Determine treble vs bass
        if ny <= best_gs.gap_midpoint:
            staff = best_gs.treble
            staff_name = "treble"
        else:
            staff = best_gs.bass
            staff_name = "bass"

        note_name, octave = _y_to_diatonic(ny, staff)

        notes.append(ExtractedNote(
            x=nx,
            y=ny,
            pitch_name=note_name,
            octave=octave,
            accidental=0,  # resolved later
            staff=staff_name,
            system_index=best_gs.system_index,
            notehead_type=ntype,
        ))

    return notes


# ---------------------------------------------------------------------------
# Step 8: Resolve accidentals
# ---------------------------------------------------------------------------

def _resolve_accidentals(
    notes: list[ExtractedNote],
    raw_accidentals: list[tuple[float, float, int]],
    grand_staffs: list[GrandStaff],
    layout: PageLayout,
) -> None:
    """Apply key signature, then explicit accidentals with in-measure propagation.

    Modifies notes in-place.
    """
    # Build key signature map per system
    gs_map: dict[int, GrandStaff] = {gs.system_index: gs for gs in grand_staffs}

    # Pass 1: Apply key signature
    for note in notes:
        gs = gs_map.get(note.system_index)
        if gs and note.pitch_name in gs.key_signature:
            note.accidental = gs.key_signature[note.pitch_name]

    # Pass 2: Match explicit accidentals to notes
    # Sort accidentals by x for efficient matching
    acc_sorted = sorted(raw_accidentals, key=lambda a: a[0])

    # Filter out key-signature accidentals (those before the first barline)
    first_barline_xs: dict[int, float] = {}
    for sys in layout.systems:
        if sys.barlines:
            first_barline_xs[sys.system_index] = sys.barlines[0].x

    # Only keep accidentals that are after the first barline (in-measure accidentals)
    measure_accidentals = []
    for ax, ay, aval in acc_sorted:
        # Find which system this accidental belongs to
        for gs in grand_staffs:
            top = gs.treble.top_line_y - 30
            bottom = gs.bass.bottom_line_y + 30
            if top <= ay <= bottom:
                fbx = first_barline_xs.get(gs.system_index, 0)
                if ax > fbx - 5:  # after first barline (with small tolerance)
                    measure_accidentals.append((ax, ay, aval, gs.system_index))
                break

    # Match each accidental to the nearest note to its right
    for ax, ay, aval, sys_idx in measure_accidentals:
        best_note: ExtractedNote | None = None
        best_dx = float("inf")

        for note in notes:
            if note.system_index != sys_idx:
                continue
            dx = note.x - ax  # note should be to the right
            dy = abs(note.y - ay)
            if 0 < dx < ACCIDENTAL_MAX_DX and dy < ACCIDENTAL_MAX_DY and dx < best_dx:
                best_dx = dx
                best_note = note

        if best_note is None:
            continue

        best_note.accidental = aval

        # Forward propagation: apply to all subsequent notes with same
        # pitch_name + octave in the same measure
        # Find measure boundaries for this note
        sys_layout = None
        for sl in layout.systems:
            if sl.system_index == sys_idx:
                sys_layout = sl
                break

        if sys_layout is None:
            continue

        # Find which measure the accidental is in
        measure_left = 0.0
        measure_right = float("inf")
        for i in range(len(sys_layout.barlines) - 1):
            bl = sys_layout.barlines[i].x
            br = sys_layout.barlines[i + 1].x
            if bl <= best_note.x <= br:
                measure_left = bl
                measure_right = br
                break

        # Propagate to subsequent notes in same measure
        for note in notes:
            if (note.system_index == sys_idx
                    and note.pitch_name == best_note.pitch_name
                    and note.octave == best_note.octave
                    and note.x > best_note.x
                    and measure_left <= note.x <= measure_right
                    and note is not best_note):
                note.accidental = aval


# ---------------------------------------------------------------------------
# Step 9: Group notes into measures
# ---------------------------------------------------------------------------

def _group_notes_into_measures(
    notes: list[ExtractedNote],
    layout: PageLayout,
) -> list[MeasureNotes]:
    """Group notes by measure (using barline X) and sub-group by X for chords."""
    measures: list[MeasureNotes] = []

    for sys in layout.systems:
        sys_notes = [n for n in notes if n.system_index == sys.system_index]

        for mi in range(sys.measure_count):
            left_x = sys.barlines[mi].x
            right_x = sys.barlines[mi + 1].x

            # Notes within this measure
            m_notes = [n for n in sys_notes if left_x - 2 <= n.x <= right_x + 2]
            if not m_notes:
                measures.append(MeasureNotes(
                    measure_index=mi,
                    system_index=sys.system_index,
                    note_groups=[],
                ))
                continue

            # Sub-group by X position (simultaneous notes = chord)
            m_notes.sort(key=lambda n: n.x)
            groups: list[NoteGroup] = []
            current_group_notes: list[ExtractedNote] = [m_notes[0]]
            current_x = m_notes[0].x

            for n in m_notes[1:]:
                if abs(n.x - current_x) <= NOTE_X_GROUP_TOLERANCE:
                    current_group_notes.append(n)
                else:
                    avg_x = sum(nn.x for nn in current_group_notes) / len(current_group_notes)
                    groups.append(NoteGroup(
                        x=avg_x,
                        notes=current_group_notes,
                        measure_index=mi,
                        system_index=sys.system_index,
                    ))
                    current_group_notes = [n]
                    current_x = n.x

            avg_x = sum(nn.x for nn in current_group_notes) / len(current_group_notes)
            groups.append(NoteGroup(
                x=avg_x,
                notes=current_group_notes,
                measure_index=mi,
                system_index=sys.system_index,
            ))

            measures.append(MeasureNotes(
                measure_index=mi,
                system_index=sys.system_index,
                note_groups=groups,
            ))

    return measures


# ---------------------------------------------------------------------------
# Step 10: Detect if page is vector music
# ---------------------------------------------------------------------------

def detect_is_vector_music(page: fitz.Page) -> bool:
    """Heuristic: does this page contain music font glyphs (vector score)?"""
    count = 0
    for _x, _y, cp, _font in _iter_music_chars(page):
        if cp in NOTEHEAD_CODEPOINTS:
            count += 1
            if count >= MIN_NOTEHEADS_FOR_MUSIC:
                return True
    return False


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_notes(page: fitz.Page, layout: PageLayout) -> PageNoteData:
    """Full note extraction pipeline for one page.

    Returns a PageNoteData with grand staffs and measure-grouped notes.
    """
    # Step 1: Staff lines
    staff_groups = _detect_staff_lines(page)
    if len(staff_groups) < 2:
        return PageNoteData(
            page_number=page.number,
            grand_staffs=[],
            measures=[],
            has_music_glyphs=False,
        )

    # Step 2: Clefs
    clef_map = _detect_clefs(page, staff_groups)

    # Step 3: Pair into grand staffs
    grand_staffs = _pair_staves(staff_groups, clef_map, layout)
    if not grand_staffs:
        return PageNoteData(
            page_number=page.number,
            grand_staffs=[],
            measures=[],
            has_music_glyphs=False,
        )

    # Step 4: Key signatures
    for gs in grand_staffs:
        sys_layout = None
        for sl in layout.systems:
            if sl.system_index == gs.system_index:
                sys_layout = sl
                break
        if sys_layout and sys_layout.barlines:
            gs.key_signature = _detect_key_signature(page, gs, sys_layout.barlines)

    # Step 5 & 6: Raw noteheads and accidentals
    noteheads = _extract_raw_noteheads(page)
    raw_accidentals = _extract_raw_accidentals(page)

    # Step 7: Assign to staves and compute pitch
    notes = _assign_notes_to_staves(noteheads, grand_staffs)

    # Step 8: Resolve accidentals
    _resolve_accidentals(notes, raw_accidentals, grand_staffs, layout)

    # Step 9: Group into measures
    measures = _group_notes_into_measures(notes, layout)

    return PageNoteData(
        page_number=page.number,
        grand_staffs=grand_staffs,
        measures=measures,
        has_music_glyphs=len(noteheads) >= MIN_NOTEHEADS_FOR_MUSIC,
    )

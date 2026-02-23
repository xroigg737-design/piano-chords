"""Identify chords from sets of pitch classes extracted by note_extractor.

Provides template matching, harmonic rhythm detection, enharmonic spelling,
and a bridge function that produces output compatible with annotate_page().
"""

from __future__ import annotations

from note_extractor import (
    ExtractedNote,
    MeasureNotes,
    NoteGroup,
    PageNoteData,
)
from pdf_writer import PageLayout
from config import MIN_CHORD_CONFIDENCE, HARMONIC_RHYTHM_SPLIT


# ---------------------------------------------------------------------------
# Pitch-class constants
# ---------------------------------------------------------------------------

C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B = range(12)

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


# ---------------------------------------------------------------------------
# Chord templates  (intervals relative to root)
# ---------------------------------------------------------------------------

CHORD_TEMPLATES: list[tuple[set[int], str]] = [
    # Triads
    ({0, 4, 7},         ""),        # major
    ({0, 3, 7},         "-"),       # minor
    ({0, 3, 6},         "o"),       # diminished triad
    ({0, 4, 8},         "+"),       # augmented
    ({0, 5, 7},         "sus"),     # sus4
    ({0, 2, 7},         "sus2"),    # sus2
    # Seventh chords
    ({0, 4, 7, 11},     "maj7"),    # major 7
    ({0, 4, 7, 10},     "7"),       # dominant 7
    ({0, 3, 7, 10},     "-7"),      # minor 7
    ({0, 3, 6, 10},     "\u00f8"),  # half-diminished (ø)
    ({0, 3, 6, 9},      "o7"),      # diminished 7
    ({0, 4, 8, 11},     "maj7#5"),  # augmented major 7
    ({0, 4, 8, 10},     "7#5"),     # augmented dominant 7
    ({0, 3, 7, 11},     "-maj7"),   # minor-major 7
    # Sixth chords
    ({0, 4, 7, 9},      "6"),       # major 6
    ({0, 3, 7, 9},      "-6"),      # minor 6
    # Extended / altered
    ({0, 4, 7, 10, 6},  "7#11"),    # dominant 7 #11
    ({0, 4, 7, 10, 1},  "7b9"),     # dominant 7 b9
    ({0, 4, 7, 10, 8},  "7b13"),    # dominant 7 b13
    ({0, 5, 7, 10},     "7sus"),    # dominant 7 sus4
]


# ---------------------------------------------------------------------------
# Enharmonic spelling
# ---------------------------------------------------------------------------

def _spell_root(pitch_class: int, key_sharps: int) -> str:
    """Spell a pitch class as a root name, respecting the key context.

    key_sharps > 0 = sharp key, < 0 = flat key.
    """
    # Preferred spellings for ambiguous pitch classes
    preferred: dict[int, str] = {
        0: "C",
        2: "D",
        4: "E",
        5: "F",
        7: "G",
        9: "A",
        11: "B",
    }
    # Ambiguous ones depend on key
    if 1 not in preferred:
        preferred[1] = "C#" if key_sharps >= 0 else "Db"
    if 3 not in preferred:
        preferred[3] = "Eb" if key_sharps <= 2 else "D#"
    if 6 not in preferred:
        preferred[6] = "F#" if key_sharps >= -1 else "Gb"
    if 8 not in preferred:
        preferred[8] = "Ab" if key_sharps <= 3 else "G#"
    if 10 not in preferred:
        preferred[10] = "Bb"  # almost always Bb

    return preferred.get(pitch_class, SHARP_NAMES[pitch_class])


def _key_sharps_from_signature(key_sig: dict[str, int]) -> int:
    """Convert a key signature dict to a sharp count (negative = flats)."""
    if not key_sig:
        return 0
    vals = list(key_sig.values())
    if vals[0] > 0:
        return len(key_sig)   # sharps
    else:
        return -len(key_sig)  # flats


def _diatonic_quality(root_pc: int, key_sharps: int) -> str | None:
    """Return the diatonic 7th chord quality for a root in the given key.

    Used as a tiebreaker: when two chord qualities match equally,
    the diatonic one gets a small preference bonus.
    """
    # Build major scale for this key signature
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    sharp_order = [5, 0, 7, 2, 9, 4, 11]
    flat_order = [11, 4, 9, 2, 7, 0, 5]

    scale_pcs = list(major_scale)
    if key_sharps > 0:
        for i in range(min(key_sharps, 7)):
            if sharp_order[i] in scale_pcs:
                idx = scale_pcs.index(sharp_order[i])
                scale_pcs[idx] = (sharp_order[i] + 1) % 12
    elif key_sharps < 0:
        for i in range(min(-key_sharps, 7)):
            if flat_order[i] in scale_pcs:
                idx = scale_pcs.index(flat_order[i])
                scale_pcs[idx] = (flat_order[i] - 1) % 12

    scale_set = set(scale_pcs)

    if root_pc not in scale_set:
        return None  # chromatic root — no diatonic preference

    sorted_scale = sorted(scale_pcs)
    root_idx = sorted_scale.index(root_pc)

    def _scale_degree(steps: int) -> int:
        return sorted_scale[(root_idx + steps) % 7]

    third = (_scale_degree(2) - root_pc) % 12
    fifth = (_scale_degree(4) - root_pc) % 12
    seventh = (_scale_degree(6) - root_pc) % 12
    intervals = {0, third, fifth, seventh}

    for tmpl_intervals, quality in CHORD_TEMPLATES:
        if tmpl_intervals == intervals:
            return quality

    return None


# ---------------------------------------------------------------------------
# Core chord identification
# ---------------------------------------------------------------------------

def _match_template(
    interval_set: set[int],
    template_intervals: set[int],
    quality: str,
) -> float:
    """Score how well interval_set matches a chord template."""
    matched = len(interval_set & template_intervals)
    missing = len(template_intervals - interval_set)
    extra = len(interval_set - template_intervals)

    if matched == 0:
        return 0.0

    score = matched / len(template_intervals)
    score -= missing * 0.15
    score -= extra * 0.05

    # Bonus for 7th chord templates when well-matched.
    # Check for "7" in quality OR ø (half-dim, a 7th chord without "7" in name)
    # Exclude 6th chords which also have 4 intervals but aren't 7th chords.
    if len(template_intervals) >= 4 and matched >= 3:
        if "7" in quality or quality == "\u00f8":
            score += 0.1

    # Bonus for exact match
    if missing == 0:
        score += 0.2

    return score


def identify_chord(
    pitch_classes: set[int],
    bass_pitch_class: int | None = None,
    key_sharps: int = 0,
) -> tuple[str, float]:
    """Identify a chord from a set of pitch classes.

    Returns (chord_symbol, confidence).
    """
    if len(pitch_classes) < 2:
        return ("N.C.", 0.0)

    # Candidate roots: bass first, then all pitch classes
    candidates: list[int] = []
    if bass_pitch_class is not None:
        candidates.append(bass_pitch_class)
    for pc in sorted(pitch_classes):
        if pc not in candidates:
            candidates.append(pc)

    best_symbol = "?"
    best_score = -1.0
    best_is_bass_root = False

    # Pre-compute diatonic qualities for roots (used as tiebreaker)
    _diatonic_cache: dict[int, str | None] = {}

    for root in candidates:
        interval_set = {(pc - root) % 12 for pc in pitch_classes}

        # Compute diatonic quality for this root once
        if root not in _diatonic_cache:
            _diatonic_cache[root] = _diatonic_quality(root, key_sharps)

        for template_intervals, quality in CHORD_TEMPLATES:
            score = _match_template(interval_set, template_intervals, quality)

            # Strong bonus if bass = root (the bass defines the chord in most cases)
            is_bass_root = (root == bass_pitch_class)
            if is_bass_root:
                score += 0.35
            elif bass_pitch_class is not None:
                # Penalty for non-bass roots when bass is clear
                score -= 0.15

            # Small diatonic preference: when the chord quality matches what the
            # key signature predicts for this root, add a tiny bonus.  This
            # breaks ties between e.g. F#ø and F#-7 in E minor (where F#ø is
            # diatonic) without overriding clear evidence from the notes.
            if _diatonic_cache[root] == quality:
                score += 0.02

            # Prefer this match if clearly higher score, or near-equal score
            # with bass=root (tie-break in favor of bass-rooted chords)
            is_better = score > best_score + 1e-9
            is_tied_but_bass = (abs(score - best_score) < 1e-9
                                and is_bass_root and not best_is_bass_root)
            if is_better or is_tied_but_bass:
                best_score = score
                root_name = _spell_root(root, key_sharps)
                best_symbol = root_name + quality
                best_is_bass_root = is_bass_root

    # If bass is not the root and confidence is decent, consider slash chord
    if (bass_pitch_class is not None
            and not best_is_bass_root
            and best_score >= MIN_CHORD_CONFIDENCE):
        bass_name = _spell_root(bass_pitch_class, key_sharps)
        # Try without the bass note to find a cleaner chord on top
        upper_pcs = pitch_classes - {bass_pitch_class}
        if len(upper_pcs) >= 2:
            upper_candidates = sorted(upper_pcs)
            for root in upper_candidates:
                interval_set = {(pc - root) % 12 for pc in upper_pcs}
                for template_intervals, quality in CHORD_TEMPLATES:
                    score = _match_template(interval_set, template_intervals, quality)
                    score += 0.05  # small bonus for cleaner upper structure
                    if score > best_score:
                        best_score = score
                        root_name = _spell_root(root, key_sharps)
                        best_symbol = f"{root_name}{quality}/{bass_name}"

    return (best_symbol, best_score)


# ---------------------------------------------------------------------------
# Harmonic rhythm detection
# ---------------------------------------------------------------------------

def _analyze_note_group_chord(
    note_groups: list[NoteGroup],
    key_sharps: int,
) -> tuple[str, float]:
    """Identify chord from a list of note groups.

    Two-pass approach:
    1. Bass staff notes define the base chord (root + quality).
    2. Treble melody notes on the downbeat may upgrade to an extension
       (e.g., 7th chord → 7#11) but cannot change the root or base quality.
    """
    if not note_groups:
        return ("N.C.", 0.0)

    # Separate bass-staff and treble-staff pitch classes
    bass_staff_pcs: dict[int, int] = {}  # pc -> group count
    treble_staff_pcs: dict[int, int] = {}
    all_pc_count: dict[int, int] = {}

    for ng in note_groups:
        group_bass_pcs = set()
        group_treble_pcs = set()
        for n in ng.notes:
            if n.staff == "bass":
                group_bass_pcs.add(n.pitch_class)
            else:
                group_treble_pcs.add(n.pitch_class)
        for pc in group_bass_pcs:
            bass_staff_pcs[pc] = bass_staff_pcs.get(pc, 0) + 1
        for pc in group_treble_pcs:
            treble_staff_pcs[pc] = treble_staff_pcs.get(pc, 0) + 1
        for pc in group_bass_pcs | group_treble_pcs:
            all_pc_count[pc] = all_pc_count.get(pc, 0) + 1

    n_groups = len(note_groups)

    # Step 1: Build core PCs from bass staff
    if bass_staff_pcs and n_groups > 3:
        core_pcs = {pc for pc, cnt in bass_staff_pcs.items() if cnt >= 2}
        # Also include frequent treble PCs (sustained chord tones)
        treble_threshold = max(2, n_groups * 0.25)
        for pc, cnt in treble_staff_pcs.items():
            if cnt >= treble_threshold:
                core_pcs.add(pc)
    elif n_groups > 3:
        # No bass staff notes — use frequency filter on all notes
        threshold = max(2, n_groups * 0.25)
        core_pcs = {pc for pc, cnt in all_pc_count.items() if cnt >= threshold}
    else:
        core_pcs = set(all_pc_count.keys())

    # Ensure at least 2 PCs
    if len(core_pcs) < 2:
        core_pcs = {pc for pc, _ in sorted(all_pc_count.items(), key=lambda x: -x[1])[:3]}

    # Bass = lowest note in first group
    bass_pc = None
    if note_groups[0].notes:
        bass_note = min(note_groups[0].notes, key=lambda n: n.midi_pitch)
        bass_pc = bass_note.pitch_class

    # Step 1: Identify base chord from core PCs
    base_symbol, base_conf = identify_chord(core_pcs, bass_pc, key_sharps)

    # Step 2: Try adding downbeat treble melody note as extension
    downbeat_treble_pcs = {n.pitch_class for n in note_groups[0].notes
                           if n.staff == "treble"}
    new_pcs = downbeat_treble_pcs - core_pcs
    if new_pcs and base_conf >= MIN_CHORD_CONFIDENCE:
        extended_pcs = core_pcs | new_pcs
        ext_symbol, ext_conf = identify_chord(extended_pcs, bass_pc, key_sharps)
        # Accept extension only if it improves or maintains quality
        # and keeps the same root (doesn't change the chord fundamentally)
        if ext_conf >= base_conf and ext_symbol[0] == base_symbol[0]:
            return (ext_symbol, ext_conf)

    return (base_symbol, base_conf)


def _detect_harmonic_rhythm(
    measure: MeasureNotes,
    key_sharps: int,
) -> list[dict]:
    """Detect if a measure has one chord or a chord change mid-measure.

    Always evaluates both full-measure and split-measure analyses, choosing
    whichever produces cleaner results.

    Returns list of chord dicts: [{"chord": "...", "beat_position": "..."}]
    """
    if not measure.note_groups:
        return []

    # Try single chord for whole measure
    full_symbol, full_conf = _analyze_note_group_chord(measure.note_groups, key_sharps)

    if not HARMONIC_RHYTHM_SPLIT or len(measure.note_groups) <= 2:
        if full_conf >= MIN_CHORD_CONFIDENCE:
            return [{"chord": full_symbol, "beat_position": "full"}]
        return []

    # Always try splitting at midpoint
    xs = [ng.x for ng in measure.note_groups]
    mid_x = (min(xs) + max(xs)) / 2

    first_half = [ng for ng in measure.note_groups if ng.x <= mid_x]
    second_half = [ng for ng in measure.note_groups if ng.x > mid_x]

    if not first_half or not second_half:
        if full_conf >= MIN_CHORD_CONFIDENCE:
            return [{"chord": full_symbol, "beat_position": "full"}]
        return []

    sym1, conf1 = _analyze_note_group_chord(first_half, key_sharps)
    sym2, conf2 = _analyze_note_group_chord(second_half, key_sharps)

    # Check if note content changes between halves (pitch class difference)
    pcs_first = {n.pitch_class for ng in first_half for n in ng.notes}
    pcs_second = {n.pitch_class for ng in second_half for n in ng.notes}
    pcs_changed = pcs_first != pcs_second

    # Check if bass note changes
    bass1 = min(first_half[0].notes, key=lambda n: n.midi_pitch) if first_half[0].notes else None
    bass2 = min(second_half[0].notes, key=lambda n: n.midi_pitch) if second_half[0].notes else None
    bass_changes = (bass1 and bass2 and bass1.pitch_class != bass2.pitch_class)

    # Decide: split or full?
    avg_split_conf = (conf1 + conf2) / 2
    min_split_conf = min(conf1, conf2)
    prefer_split = False

    if sym1 != sym2 and pcs_changed:
        if bass_changes:
            # Bass note changes → strong evidence of harmonic change
            # Accept split if both halves produce reasonable chords
            if min_split_conf >= MIN_CHORD_CONFIDENCE:
                prefer_split = True
        else:
            # Same bass, different upper notes — only split if split is
            # clearly better than full-measure analysis (not just 80%)
            if avg_split_conf >= full_conf * 0.95 and min_split_conf >= MIN_CHORD_CONFIDENCE:
                prefer_split = True

    if sym1 == sym2:
        # Same chord in both halves → definitely don't split
        prefer_split = False

    if prefer_split:
        result = []
        if conf1 >= MIN_CHORD_CONFIDENCE:
            result.append({"chord": sym1, "beat_position": "downbeat"})
        if conf2 >= MIN_CHORD_CONFIDENCE:
            result.append({"chord": sym2, "beat_position": "beat3"})
        if result:
            return result

    # Fall back to single chord
    if full_conf >= MIN_CHORD_CONFIDENCE:
        return [{"chord": full_symbol, "beat_position": "full"}]
    return []


# ---------------------------------------------------------------------------
# Bridge function: produce annotate_page()-compatible dict
# ---------------------------------------------------------------------------

def analyze_page_chords(
    note_data: PageNoteData,
    layout: PageLayout,
    page_number: int,
) -> dict:
    """Convert extracted note data into the dict format expected by annotate_page().

    Returns:
        {
            "page_number": int,
            "has_music": bool,
            "systems": [
                {
                    "system_index": int,
                    "measures": [
                        {
                            "measure_index": int,
                            "chords": [{"chord": "...", "beat_position": "..."}]
                        }
                    ]
                }
            ]
        }
    """
    if not note_data.has_music_glyphs:
        return {
            "page_number": page_number,
            "has_music": False,
            "systems": [],
        }

    # Compute key_sharps from first grand staff's key signature
    key_sharps = 0
    if note_data.grand_staffs:
        key_sharps = _key_sharps_from_signature(note_data.grand_staffs[0].key_signature)

    # Group measures by system
    measures_by_system: dict[int, list[MeasureNotes]] = {}
    for m in note_data.measures:
        measures_by_system.setdefault(m.system_index, []).append(m)

    systems_out = []
    for sys_layout in layout.systems:
        si = sys_layout.system_index
        sys_measures = measures_by_system.get(si, [])
        sys_measures.sort(key=lambda m: m.measure_index)

        measures_out = []
        for m in sys_measures:
            chords = _detect_harmonic_rhythm(m, key_sharps)
            measures_out.append({
                "measure_index": m.measure_index,
                "chords": chords,
            })

        systems_out.append({
            "system_index": si,
            "measures": measures_out,
        })

    return {
        "page_number": page_number,
        "has_music": True,
        "systems": systems_out,
    }

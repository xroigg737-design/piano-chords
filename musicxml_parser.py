"""Parse MusicXML files and extract note data for chord identification.

Converts MusicXML into the same data structures used by note_extractor,
so that chord_identifier can process them identically.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from note_extractor import ExtractedNote, NoteGroup, MeasureNotes


# Pitch name → semitone offset (same as ExtractedNote.midi_pitch)
_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


@dataclass
class MusicXMLData:
    """Parsed MusicXML data ready for chord identification."""
    title: str
    measures: list[MeasureNotes]
    key_sharps: int  # positive = sharps, negative = flats
    key_mode: str  # "major" or "minor"
    time_num: int  # numerator (e.g. 4)
    time_den: int  # denominator (e.g. 4)
    total_measures: int


def _pick_piano_part(parts: list[ET.Element], ns: str) -> ET.Element:
    """Select the part most likely to be the piano.

    Heuristic: pick the part with the most <note> elements that have <pitch>.
    Skip parts that are all rests.
    """
    best_part = parts[0]
    best_count = 0
    for part in parts:
        count = 0
        for note in part.iter(f"{ns}note"):
            if note.find(f"{ns}pitch") is not None:
                count += 1
        if count > best_count:
            best_count = count
            best_part = part
    return best_part


def _parse_measure_notes(
    measure_el: ET.Element,
    ns: str,
    divisions: int,
) -> list[ExtractedNote]:
    """Parse all notes from a single <measure>, handling voices and backup/forward.

    Tracks beat position per voice so that <backup>/<forward> don't corrupt
    the beat timeline of other voices.
    """

    def _find(el, tag):
        return el.find(f"{ns}{tag}")

    def _findtext(el, tag, default=""):
        result = el.findtext(f"{ns}{tag}", default)
        return result if result else default

    # Track beat position per voice independently.
    # In MusicXML, <backup> moves a shared cursor back to write another voice.
    # We use a cursor that tracks position, and each voice remembers its own beat.
    cursor = 0.0  # current position in the measure (in beats)
    voice_beats: dict[str, float] = {}
    last_voice: str = "1"
    notes: list[ExtractedNote] = []

    for elem in measure_el:
        tag = elem.tag.replace(ns, "") if ns else elem.tag

        if tag == "note":
            voice = _findtext(elem, "voice", "1")
            dur_el = _find(elem, "duration")
            duration = int(dur_el.text) if dur_el is not None else divisions
            is_chord = _find(elem, "chord") is not None

            # Initialize voice beat from cursor if new voice
            if voice not in voice_beats:
                voice_beats[voice] = cursor
            # If voice changed (after a backup), sync to cursor
            if voice != last_voice:
                voice_beats[voice] = cursor
                last_voice = voice

            current_beat = voice_beats[voice]

            # Rest: just advance the voice beat
            if _find(elem, "rest") is not None:
                if not is_chord:
                    voice_beats[voice] = current_beat + duration / divisions
                    cursor = voice_beats[voice]
                continue

            # Must have pitch
            pitch_el = _find(elem, "pitch")
            if pitch_el is None:
                if not is_chord:
                    voice_beats[voice] = current_beat + duration / divisions
                    cursor = voice_beats[voice]
                continue

            step = _findtext(pitch_el, "step", "C")
            octave = int(_findtext(pitch_el, "octave", "4"))
            alter = int(float(_findtext(pitch_el, "alter", "0")))

            staff_num = int(_findtext(elem, "staff", "1"))
            staff_name = "treble" if staff_num == 1 else "bass"

            note = ExtractedNote(
                x=current_beat,
                y=0.0,
                pitch_name=step,
                octave=octave,
                accidental=alter,
                staff=staff_name,
                system_index=0,
                notehead_type="filled",
            )
            notes.append(note)

            # Only advance beat if not a chord member
            if not is_chord:
                voice_beats[voice] = current_beat + duration / divisions
                cursor = voice_beats[voice]

        elif tag == "forward":
            dur = int(_findtext(elem, "duration", "0"))
            cursor += dur / divisions

        elif tag == "backup":
            dur = int(_findtext(elem, "duration", "0"))
            cursor = max(0.0, cursor - dur / divisions)

    return notes


# Key signature ↔ pitch-class tonic (circle of fifths)
_SHARPS_TO_MAJOR_TONIC = {
    0: 0, 1: 7, 2: 2, 3: 9, 4: 4, 5: 11, 6: 6, 7: 1,
    -1: 5, -2: 10, -3: 3, -4: 8, -5: 1, -6: 6, -7: 11,
}
_MAJOR_TONIC_TO_SHARPS = {v: k for k, v in _SHARPS_TO_MAJOR_TONIC.items()}

# Krumhansl-Kessler key profiles (correlation targets)
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n == 0:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _detect_key_krumhansl(
    measures: list,
) -> tuple[int | None, str]:
    """Detect key from note content using the Krumhansl-Kessler algorithm.

    Correlates the pitch-class distribution of all notes against major and
    minor key profiles for each of the 12 possible tonics.  Returns
    (key_sharps, key_mode) for the best match, or (None, "") if there
    aren't enough notes.
    """
    from collections import Counter

    pc_count: Counter[int] = Counter()
    for m in measures:
        for ng in m.note_groups:
            for n in ng.notes:
                pc_count[n.pitch_class] += 1

    total = sum(pc_count.values())
    if total < 10:
        return None, ""

    dist = [pc_count[i] / total for i in range(12)]

    best_r = -2.0
    best_tonic = 0
    best_mode = "major"

    for tonic in range(12):
        # Rotate distribution so that tonic maps to index 0
        rotated = [dist[(tonic + i) % 12] for i in range(12)]
        r_maj = _pearson(rotated, _KK_MAJOR)
        r_min = _pearson(rotated, _KK_MINOR)

        if r_maj > best_r:
            best_r = r_maj
            best_tonic = tonic
            best_mode = "major"
        if r_min > best_r:
            best_r = r_min
            best_tonic = tonic
            best_mode = "minor"

    # Convert tonic pitch class to key_sharps
    if best_mode == "major":
        key_sharps = _MAJOR_TONIC_TO_SHARPS.get(best_tonic)
    else:
        # Minor tonic → relative major tonic → key_sharps
        rel_major_tonic = (best_tonic + 3) % 12
        key_sharps = _MAJOR_TONIC_TO_SHARPS.get(rel_major_tonic)

    if key_sharps is None:
        return None, ""

    return key_sharps, best_mode


def parse_musicxml(file_path: str) -> MusicXMLData:
    """Parse a MusicXML file and extract note data for chord identification.

    Supports both .xml (plain MusicXML) and .musicxml formats.
    Returns MusicXMLData with measures containing NoteGroups.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Handle namespace (MusicXML 4.0 uses a namespace)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def find(el: ET.Element, tag: str):
        return el.find(f"{ns}{tag}")

    def findall(el: ET.Element, tag: str):
        return el.findall(f"{ns}{tag}")

    def findtext(el: ET.Element, tag: str, default: str = ""):
        result = el.findtext(f"{ns}{tag}", default)
        return result if result else default

    # Extract title
    title = ""
    work_el = find(root, "work")
    if work_el is not None:
        title = findtext(work_el, "work-title")
    if not title:
        movement = findtext(root, "movement-title")
        if movement:
            title = movement

    parts = findall(root, "part")
    if not parts:
        raise ValueError("No s'ha trobat cap part al fitxer MusicXML")

    # Pick the part with the most notes (likely the piano)
    part = _pick_piano_part(parts, ns)

    # Default values
    divisions = 1
    key_sharps = 0
    key_mode = "major"
    time_num = 4
    time_den = 4

    measures_by_index: dict[int, list[ExtractedNote]] = {}
    measure_count = 0

    for measure_el in findall(part, "measure"):
        measure_num_str = measure_el.get("number", str(measure_count + 1))
        try:
            measure_index = int(measure_num_str) - 1
        except ValueError:
            measure_index = measure_count

        # Check for attributes (key, time, divisions)
        attrs = find(measure_el, "attributes")
        if attrs is not None:
            div_el = find(attrs, "divisions")
            if div_el is not None and div_el.text:
                divisions = int(div_el.text)

            key_el = find(attrs, "key")
            if key_el is not None:
                fifths_text = findtext(key_el, "fifths", "0")
                key_sharps = int(fifths_text)
                key_mode = findtext(key_el, "mode", "major")

            time_el = find(attrs, "time")
            if time_el is not None:
                beats_text = findtext(time_el, "beats", "4")
                beat_type_text = findtext(time_el, "beat-type", "4")
                time_num = int(beats_text)
                time_den = int(beat_type_text)

        # Parse notes with proper voice tracking
        notes = _parse_measure_notes(measure_el, ns, divisions)
        measures_by_index[measure_index] = notes
        measure_count = max(measure_count, measure_index + 1)

    # Convert to MeasureNotes with NoteGroups
    # Group notes by beat position (use small tolerance for float comparison)
    BEAT_TOLERANCE = 0.05
    all_measures: list[MeasureNotes] = []

    for mi in sorted(measures_by_index.keys()):
        notes = measures_by_index[mi]
        if not notes:
            all_measures.append(MeasureNotes(
                measure_index=mi,
                system_index=0,
                note_groups=[],
            ))
            continue

        # Deduplicate notes (same pitch at same beat from different voices)
        seen_keys: set[tuple[float, int]] = set()
        deduped: list[ExtractedNote] = []
        for n in notes:
            # Round beat to avoid float drift
            beat_key = round(n.x * 100)
            key = (beat_key, n.midi_pitch)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(n)
        notes = deduped

        if not notes:
            all_measures.append(MeasureNotes(
                measure_index=mi,
                system_index=0,
                note_groups=[],
            ))
            continue

        # Group notes by beat position
        notes.sort(key=lambda n: (n.x, n.midi_pitch))
        groups: list[NoteGroup] = []
        current_group: list[ExtractedNote] = [notes[0]]
        current_x = notes[0].x

        for n in notes[1:]:
            if abs(n.x - current_x) <= BEAT_TOLERANCE:
                current_group.append(n)
            else:
                avg_x = sum(nn.x for nn in current_group) / len(current_group)
                groups.append(NoteGroup(
                    x=avg_x,
                    notes=current_group,
                    measure_index=mi,
                    system_index=0,
                ))
                current_group = [n]
                current_x = n.x

        avg_x = sum(nn.x for nn in current_group) / len(current_group)
        groups.append(NoteGroup(
            x=avg_x,
            notes=current_group,
            measure_index=mi,
            system_index=0,
        ))

        all_measures.append(MeasureNotes(
            measure_index=mi,
            system_index=0,
            note_groups=groups,
        ))

    # Use Krumhansl-Kessler only as fallback when MusicXML key looks like a
    # default (0 sharps / C major) — notation editors export the correct key,
    # so overriding it with a statistical guess was causing wrong tonalities.
    xml_key_is_default = (key_sharps == 0 and key_mode == "major")
    detected_sharps, detected_mode = _detect_key_krumhansl(all_measures)
    if xml_key_is_default and detected_sharps is not None:
        key_sharps = detected_sharps
        key_mode = detected_mode

    return MusicXMLData(
        title=title,
        measures=all_measures,
        key_sharps=key_sharps,
        key_mode=key_mode,
        time_num=time_num,
        time_den=time_den,
        total_measures=measure_count,
    )

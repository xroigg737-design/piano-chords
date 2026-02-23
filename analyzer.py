"""Claude Vision API for chord analysis and barline estimation."""

import base64
import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


# ---------------------------------------------------------------------------
# JSON extraction helper (from VideoMind)
# ---------------------------------------------------------------------------

def _extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from a Claude response, handling markdown code blocks."""
    text = response_text.strip()
    # Try to find JSON in markdown code blocks
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        inside = False
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True
                continue
            if line.strip() == "```" and inside:
                break
            if inside:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines)
    # Fallback: find first { ... } block
    if not text.startswith("{"):
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert jazz harmony analyst producing a lead-sheet analysis of piano scores.

TASK: For each measure, identify the chord symbol(s) by analyzing ALL notes from BOTH \
hands combined. Name each chord precisely using jazz chord symbols.

═══ STEP 1 — READ THE NOTES CAREFULLY ═══
For each measure, list (mentally) every distinct pitch sounding in BOTH staves:
- Bass clef: identify the LOWEST note and the repeated chord voicing above it.
- Treble clef: identify the melody note(s).

═══ STEP 2 — IDENTIFY THE CHORD ROOT ═══
CRITICAL: The chord ROOT is usually determined by stacking the notes in thirds \
from the BASS NOTE upward — NOT from the melody note downward.

When the bass moves chromatically (e.g. E → D# → D → C# → C → B), each new bass \
note typically generates a COMPLETELY DIFFERENT chord. Do NOT default to calling \
everything a variation of the tonic chord.

Examples of correct root identification:
- Bass=E,  upper notes include G,B       → E- (E minor)
- Bass=F#, upper notes include A,C,E     → F#ø (F# half-diminished: F#-A-C-E)
- Bass=F#, upper notes include A,C,D#    → F#o7 (F# dim7: F#-A-C-D#)
- Bass=F,  upper notes include A,C,E,B   → F7#11 (F-A-C-E with #11=B)
- Bass=B,  upper notes include D,F,A     → Bø (B half-diminished: B-D-F-A)
- Bass=B,  upper notes include D#,F#,A   → B7 (B dominant 7: B-D#-F#-A)
- Bass=D,  upper notes include F,A,C     → D-7 (D minor 7: D-F-A-C)
- Bass=D,  upper notes include F,Ab,C    → Do7 (D dim7: D-F-Ab-C)
- Bass=C,  upper notes include E,G#,B    → Cmaj7#5 (C-E-G#-B)

ONLY use slash notation (e.g. E-/D#) when the bass is clearly a non-chord-tone \
pedal or a brief passing note. If the bass note can serve as the root of a \
recognizable chord with the upper notes, name that chord directly.

═══ STEP 3 — DETERMINE CHORD QUALITY ═══
Be precise — do NOT simplify to basic triads:
- Major: C           - Minor: C-  (minus sign, NOT "m")
- Dom 7: C7          - Maj 7: Cmaj7
- Min 7: C-7         - Half-dim: Cø  (= C-7b5)
- Dim 7: Co7         - Augmented: C+
- Aug maj7: Cmaj7#5  - Suspended: Csus, C7sus
- Min 6: C-6         - Altered: C7#11, C7b9, C7b13, etc.
- Slash chord: C/E   (only when necessary, see above)

═══ STEP 4 — HARMONIC RHYTHM ═══
Usually one chord per measure. If harmony changes mid-measure, report each chord \
with its beat position. Most measures in slow pieces have ONE chord.

═══ NOTATION ═══
- Roots: C D E F G A B (# or b for accidentals)
- Minor = minus: E-    - Half-dim = ø: F#ø
- Dim 7 = o7: F#o7     - Maj 7 = maj7: Cmaj7

═══ OUTPUT — valid JSON only ═══
{
  "page_number": <int>,
  "has_music": true,
  "key": "<e.g. E minor>",
  "time_signature": "<e.g. 4/4>",
  "systems": [
    {
      "system_index": <int, 0-based>,
      "measures": [
        {
          "measure_index": <int, 0-based within system>,
          "chords": [
            {"chord": "<symbol>", "beat_position": "full"}
          ]
        }
      ]
    }
  ]
}

beat_position: "full" = whole measure, "downbeat"/"beat3" = chord changes mid-measure.
If no music: {"page_number": <int>, "has_music": false, "systems": []}
"""

SYSTEM_ANALYSIS_PROMPT = """\
You are an expert jazz harmony analyst. You are looking at a SINGLE staff system \
(treble + bass clef pair) cropped from a piano score.

TASK: Identify the chord for each measure in this system.

═══ METHOD — follow these steps precisely ═══

STEP 1 — READ NOTES: For each measure, carefully identify EVERY note in both staves:
  - Account for the KEY SIGNATURE (sharps/flats at the start of the staff).
  - Account for any ACCIDENTALS (sharps, flats, naturals) on individual notes.
  - Bass clef (bottom staff): identify the LOWEST sounding note and all chord tones above.
  - Treble clef (top staff): identify melody notes and any harmony notes.

STEP 2 — DETERMINE ROOT: The chord ROOT is determined by the BASS NOTE (lowest pitch).
  When the bass descends chromatically (e.g. E→D#→D→C#→C→B), each step creates a \
DIFFERENT chord. Name the chord from that bass note — do NOT call everything a \
variation of the tonic.

STEP 3 — NAME THE CHORD: Stack ALL sounding notes (both staves) from bass upward \
and identify the chord quality precisely:

  EXAMPLES:
  Bass=E,  notes: E,G,B           → E-
  Bass=F#, notes: F#,A,C,E        → F#ø (half-diminished)
  Bass=F#, notes: F#,A,C,D#(Eb)   → F#o7 (diminished 7th)
  Bass=F,  notes: F,A,C,E,B       → F7#11
  Bass=B,  notes: B,D,F,A         → Bø (half-diminished)
  Bass=B,  notes: B,D#,F#,A       → B7 (dominant 7th)
  Bass=D,  notes: D,F,A,C         → D-7 (minor 7th)
  Bass=D,  notes: D,F,Ab,C        → Do7 (diminished 7th)
  Bass=C,  notes: C,E,G#,B        → Cmaj7#5 (augmented major 7th)
  Bass=A,  notes: A,C,E,F#        → A-6 (minor 6th)

═══ CHORD SYMBOLS ═══
Major: C      Minor: C- (minus, NOT "m")    Dom7: C7
Maj7: Cmaj7   Min7: C-7    Half-dim: Cø    Dim7: Co7
Aug: C+   Aug maj7: Cmaj7#5    Sus: Csus, C7sus    Min6: C-6
Altered: C7#11, C7b9, C7b13     Slash: C/E (ONLY for true inversions)

═══ OUTPUT — valid JSON only ═══
{
  "system_index": <int>,
  "measures": [
    {
      "measure_index": <int, 0-based>,
      "chords": [
        {"chord": "<symbol>", "beat_position": "full"}
      ]
    }
  ]
}
beat_position: "full" = whole measure. "downbeat"/"beat3" if chord changes mid-measure.
"""


BARLINE_ESTIMATION_PROMPT = """\
You are analyzing a piano sheet music image to locate barlines and staff systems.

For each staff system (treble + bass clef pair) visible on the page:
1. Estimate the vertical position of the system as fractions of page height (0.0 = top, 1.0 = bottom).
2. Estimate the horizontal position of each barline as a fraction of page width (0.0 = left, 1.0 = right).
   Include the leftmost barline (start of system) and rightmost barline (end of system).

Return ONLY valid JSON:
{
  "systems": [
    {
      "y_top_frac": <float>,
      "y_bottom_frac": <float>,
      "barline_x_fracs": [<float>, ...]
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _image_to_base64(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Page analysis
# ---------------------------------------------------------------------------

def analyze_page(
    image_bytes: bytes,
    page_number: int,
    num_systems_hint: int | None = None,
    measures_hint: list[int] | None = None,
) -> dict:
    """Send a page image to Claude Vision and get chord analysis.

    Args:
        image_bytes: PNG image of the page.
        page_number: 0-based page number.
        num_systems_hint: Number of detected staff systems (from barline detection).
        measures_hint: Number of measures per system (from barline detection).

    Returns:
        Parsed JSON dict with chord analysis.
    """
    client = _get_client()
    b64 = _image_to_base64(image_bytes)

    # Build hint message
    hints = []
    if num_systems_hint is not None:
        hints.append(f"The page appears to have {num_systems_hint} staff system(s).")
    if measures_hint:
        for i, mc in enumerate(measures_hint):
            hints.append(f"System {i} appears to have {mc} measure(s).")
    hint_text = " ".join(hints) if hints else ""

    user_content = []
    user_content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64,
        },
    })
    user_text = (
        f"Analyze page {page_number + 1} of this piano score. "
        f"For each measure, read every note in both staves carefully "
        f"before determining the chord."
    )
    if hint_text:
        user_text += f"\n\nLayout hints: {hint_text}"
    user_content.append({"type": "text", "text": user_text})

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        response = stream.get_final_message()

    # Extract text block (skip thinking blocks)
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text
            break
    if not response_text:
        # Fallback: try first block regardless of type
        if response.content:
            response_text = getattr(response.content[0], "text", "")
        if not response_text:
            raise ValueError(
                f"Empty response from Claude. Content types: "
                f"{[b.type for b in response.content]}"
            )
    result = _extract_json_from_response(response_text)
    # Ensure page_number is set
    result.setdefault("page_number", page_number)
    return result


# ---------------------------------------------------------------------------
# Per-system analysis (cropped image, more accurate)
# ---------------------------------------------------------------------------

def analyze_system(
    image_bytes: bytes,
    page_number: int,
    system_index: int,
    num_measures: int,
) -> dict:
    """Analyze a single cropped system image for chord identification.

    Args:
        image_bytes: PNG of the cropped system region.
        page_number: 0-based page number (for context).
        system_index: 0-based system index within the page.
        num_measures: Expected number of measures (from barline detection).

    Returns:
        Parsed JSON dict with system_index and measures.
    """
    client = _get_client()
    b64 = _image_to_base64(image_bytes)

    user_text = (
        f"This is system {system_index} from page {page_number + 1} of a piano score. "
        f"It has {num_measures} measure(s) separated by barlines. "
        f"Carefully read every note in both staves and identify the chord for each measure."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_ANALYSIS_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": user_text},
            ],
        }],
    )

    response_text = response.content[0].text
    result = _extract_json_from_response(response_text)
    result.setdefault("system_index", system_index)
    return result


# ---------------------------------------------------------------------------
# Barline estimation (fallback for scanned PDFs)
# ---------------------------------------------------------------------------

def estimate_barlines(image_bytes: bytes) -> dict:
    """Ask Claude to estimate barline positions from a page image.

    Returns dict with 'systems' list containing y_top_frac, y_bottom_frac,
    and barline_x_fracs for each system.
    """
    client = _get_client()
    b64 = _image_to_base64(image_bytes)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=BARLINE_ESTIMATION_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Locate all barlines and staff systems in this piano score page.",
                },
            ],
        }],
    )

    response_text = response.content[0].text
    return _extract_json_from_response(response_text)

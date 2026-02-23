import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project dir, then from videomind dir as fallback
load_dotenv()
load_dotenv(Path.home() / "videomind" / ".env")

# API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# Rendering
RENDER_DPI = 200

# Chord annotation styling
CHORD_COLOR_RH = (0.110, 0.180, 0.353)  # #1C2E5A navy blue — right hand (treble)
CHORD_COLOR_LH = (0.133, 0.420, 0.200)  # #226B33 forest green — left hand (bass)
CHORD_FONT = "helv"  # Helvetica built-in
DEFAULT_FONT_SIZE = 10.0
MIN_FONT_SIZE = 6.0
MAX_FONT_SIZE = 14.0
PADDING_ABOVE_STAFF = 12.0  # points above staff top
PADDING_BELOW_STAFF = 6.0   # points below staff bottom

# Barline detection thresholds
MIN_BARLINE_HEIGHT = 45.0   # must span a full staff system (~52pt), not just a stem (~25pt)
MAX_BARLINE_HEIGHT_FRAC = 0.5  # reject lines taller than 50% of page (borders/margins)
MAX_BARLINE_WIDTH_TOLERANCE = 1.5
BARLINE_DEDUP_DISTANCE = 5.0  # points — merge barlines closer than this

# Notation conversion
ANGLO_TO_LATIN = {
    "C": "Do", "D": "Re", "E": "Mi", "F": "Fa",
    "G": "Sol", "A": "La", "B": "Si",
}
LATIN_TO_ANGLO = {v: k for k, v in ANGLO_TO_LATIN.items()}

# Note extraction thresholds (vector PDF)
STAFF_LINE_MIN_LENGTH = 50.0   # min horizontal length to count as staff line
STAFF_LINE_MAX_DY = 0.5        # max vertical deviation for a horizontal line
STAFF_SPACING_MIN = 4.0        # min inter-line spacing (pt)
STAFF_SPACING_MAX = 6.0        # max inter-line spacing (pt)
STAFF_SPACING_TOLERANCE = 0.5  # max variance across 4 spacings in a group

NOTE_X_GROUP_TOLERANCE = 2.0   # max dx (pt) to consider notes simultaneous
ACCIDENTAL_MAX_DX = 15.0       # max horizontal distance accidental→notehead
ACCIDENTAL_MAX_DY = 3.0        # max vertical distance accidental→notehead

# Chord identification
MIN_CHORD_CONFIDENCE = 0.5     # min score to accept a chord match
HARMONIC_RHYTHM_SPLIT = True   # allow splitting measures into 2 chords

# Analysis method selection
PREFER_ALGORITHMIC = True      # prefer vector extraction over Claude Vision
MIN_NOTEHEADS_FOR_MUSIC = 10   # min noteheads to consider page as vector music

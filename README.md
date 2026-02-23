# Piano Chord Annotator

A CLI tool that automatically annotates piano sheet music PDFs with chord symbols. It uses two complementary approaches:

1. **Algorithmic extraction** from vector PDFs (SMuFL/Leland font glyphs, e.g. MuseScore exports)
2. **Claude Vision API fallback** for scanned or raster PDFs

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (needed for the Claude Vision fallback and barline estimation on non-vector PDFs)

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory with your API key:

```
ANTHROPIC_API_KEY=your_key_here
```

Alternatively, the tool will also look for a `.env` file in `~/videomind/`.

## Usage

```bash
python main.py <input_pdf> [options]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--notation {anglo,latin}` | `anglo` | Chord naming style. `anglo` = C, D, E... / `latin` = Do, Re, Mi... |
| `--font-size {auto,N}` | `auto` | Font size for chord labels (6–14 pt). `auto` scales to fit. |
| `--pages RANGE` | all | Pages to process, e.g. `1-5` or `2,4,6` |

### Examples

```bash
# Annotate all pages with Anglo notation
python main.py partitura.pdf

# Use Latin notation (Do, Re, Mi, Fa, Sol, La, Si)
python main.py partitura.pdf --notation latin

# Process only pages 1 through 3
python main.py partitura.pdf --pages 1-3

# Fixed font size of 10pt
python main.py partitura.pdf --font-size 10
```

The output is saved as `<filename>_acordes.pdf` in the same directory as the input file.

## How it works

### Pipeline

1. **Barline detection** — Scans PDF vector graphics for vertical barlines to determine measure boundaries. Falls back to Claude Vision estimation if none are found.
2. **Note extraction** (vector PDFs) — Reads SMuFL glyph positions from the PDF text layer, detects staff lines, and computes exact pitches by relating each notehead's Y coordinate to the staff.
3. **Chord identification** — Template-matches pitch-class sets against a library of chord types (triads, sevenths, extended chords, slash chords). Supports harmonic rhythm splitting (detecting mid-measure chord changes).
4. **Claude Vision fallback** — For scanned PDFs or when algorithmic extraction fails, renders the page as an image and uses Claude to identify the chords.
5. **Annotation** — Writes chord symbols onto the PDF: treble chords in navy blue above the system, bass chords in forest green below.

### Supported chord types

- Triads: major, minor, diminished, augmented, sus2, sus4
- Seventh chords: maj7, dom7, min7, min(maj7), half-diminished, dim7
- Extended/altered: 7#11, 7b9, 7b13
- Sixth chords: major 6, minor 6
- Slash chords with inversion detection

## Project structure

```
piano-chords/
  main.py              CLI entry point
  note_extractor.py    Staff detection, notehead/accidental extraction, pitch calculation
  chord_identifier.py  Chord templates, matching, harmonic rhythm detection
  pdf_writer.py        Barline detection, layout, PDF annotation
  analyzer.py          Claude Vision API integration
  config.py            Configuration (colors, fonts, thresholds, API settings)
  requirements.txt     Python dependencies
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model used for Vision analysis |
| `RENDER_DPI` | `200` | Resolution for page rendering |
| `PREFER_ALGORITHMIC` | `True` | Prefer vector extraction over Claude Vision |
| `HARMONIC_RHYTHM_SPLIT` | `True` | Allow splitting measures into 2 chords |
| `MIN_CHORD_CONFIDENCE` | `0.5` | Minimum score to accept a chord match |

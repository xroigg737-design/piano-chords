"""PDF rendering, barline detection, and chord annotation using PyMuPDF."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from config import (
    RENDER_DPI,
    CHORD_COLOR_RH,
    CHORD_COLOR_LH,
    CHORD_FONT,
    DEFAULT_FONT_SIZE,
    MIN_FONT_SIZE,
    MAX_FONT_SIZE,
    PADDING_ABOVE_STAFF,
    PADDING_BELOW_STAFF,
    MIN_BARLINE_HEIGHT,
    MAX_BARLINE_HEIGHT_FRAC,
    MAX_BARLINE_WIDTH_TOLERANCE,
    BARLINE_DEDUP_DISTANCE,
    ANGLO_TO_LATIN,
    LATIN_TO_ANGLO,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Barline:
    x: float
    y_top: float
    y_bottom: float


@dataclass
class StaffSystem:
    system_index: int
    y_top: float
    y_bottom: float
    barlines: list[Barline] = field(default_factory=list)

    @property
    def measure_count(self) -> int:
        return max(0, len(self.barlines) - 1)


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    systems: list[StaffSystem] = field(default_factory=list)
    has_vector_barlines: bool = True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_page_to_png(page: fitz.Page, dpi: int = RENDER_DPI) -> bytes:
    """Render a PDF page to PNG bytes at the given DPI."""
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def render_system_to_png(
    page: fitz.Page,
    system: StaffSystem,
    dpi: int = RENDER_DPI,
    margin_above: float = 40.0,
    margin_below: float = 40.0,
) -> bytes:
    """Render a cropped region around a single staff system to PNG."""
    clip = fitz.Rect(
        0,
        max(0, system.y_top - margin_above),
        page.rect.width,
        min(page.rect.height, system.y_bottom + margin_below),
    )
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# Barline detection from vector graphics
# ---------------------------------------------------------------------------

def _is_vertical_line(item: dict, page_height: float) -> list[tuple[float, float, float, float]]:
    """Extract vertical line segments from a drawing item.

    Filters by MIN_BARLINE_HEIGHT (rejects note stems) and
    MAX_BARLINE_HEIGHT_FRAC (rejects page borders / margin lines).

    Returns list of (x, y_top, y_bottom, width) tuples.
    """
    max_h = page_height * MAX_BARLINE_HEIGHT_FRAC
    segments = []
    for path_item in item.get("items", []):
        kind = path_item[0]
        if kind == "l":  # line segment
            p1, p2 = path_item[1], path_item[2]
            dx = abs(p1.x - p2.x)
            dy = abs(p1.y - p2.y)
            if dx < MAX_BARLINE_WIDTH_TOLERANCE and MIN_BARLINE_HEIGHT < dy < max_h:
                x = (p1.x + p2.x) / 2
                y_top = min(p1.y, p2.y)
                y_bottom = max(p1.y, p2.y)
                segments.append((x, y_top, y_bottom, dx))
        elif kind == "re":  # rectangle
            rect = path_item[1]
            w = rect.width
            h = rect.height
            if w < MAX_BARLINE_WIDTH_TOLERANCE and MIN_BARLINE_HEIGHT < h < max_h:
                x = (rect.x0 + rect.x1) / 2
                segments.append((x, rect.y0, rect.y1, w))
    return segments


def detect_barlines(page: fitz.Page) -> PageLayout:
    """Detect barlines from vector graphics on a PDF page.

    Groups barlines into staff systems by vertical proximity,
    then sorts and deduplicates within each system.
    """
    page_rect = page.rect
    layout = PageLayout(
        page_number=page.number,
        width=page_rect.width,
        height=page_rect.height,
    )

    # Collect all vertical segments that look like barlines
    raw_barlines: list[Barline] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        layout.has_vector_barlines = False
        return layout

    for d in drawings:
        for x, y_top, y_bottom, _ in _is_vertical_line(d, page_rect.height):
            raw_barlines.append(Barline(x=x, y_top=y_top, y_bottom=y_bottom))

    if not raw_barlines:
        layout.has_vector_barlines = False
        return layout

    # Group barlines into systems by overlapping Y ranges
    # Sort by y_top first
    raw_barlines.sort(key=lambda b: b.y_top)
    systems: list[list[Barline]] = []
    current_group: list[Barline] = [raw_barlines[0]]
    current_y_bottom = raw_barlines[0].y_bottom

    for bl in raw_barlines[1:]:
        # If this barline's top is within the current system's range
        if bl.y_top < current_y_bottom + 10:
            current_group.append(bl)
            current_y_bottom = max(current_y_bottom, bl.y_bottom)
        else:
            systems.append(current_group)
            current_group = [bl]
            current_y_bottom = bl.y_bottom
    systems.append(current_group)

    # Build StaffSystem objects
    for idx, group in enumerate(systems):
        if len(group) < 2:
            continue

        # Filter out outlier barlines by y_top consistency.
        # Real barlines in a system share a similar y_top; stray elements
        # (braces, brackets, accidentals) may have a shifted y_top.
        y_tops = sorted(b.y_top for b in group)
        median_y_top = y_tops[len(y_tops) // 2]
        y_top_tolerance = 15.0  # points
        group = [b for b in group if abs(b.y_top - median_y_top) < y_top_tolerance]

        if len(group) < 2:
            continue

        # Sort by X within this system
        group.sort(key=lambda b: b.x)

        # Deduplicate barlines too close in X
        deduped: list[Barline] = [group[0]]
        for bl in group[1:]:
            if abs(bl.x - deduped[-1].x) < BARLINE_DEDUP_DISTANCE:
                # Keep the taller one
                if (bl.y_bottom - bl.y_top) > (deduped[-1].y_bottom - deduped[-1].y_top):
                    deduped[-1] = bl
            else:
                deduped.append(bl)

        if len(deduped) < 2:
            continue

        y_top = min(b.y_top for b in deduped)
        y_bottom = max(b.y_bottom for b in deduped)

        layout.systems.append(StaffSystem(
            system_index=idx,
            y_top=y_top,
            y_bottom=y_bottom,
            barlines=deduped,
        ))

    if not layout.systems:
        layout.has_vector_barlines = False

    return layout


def build_layout_from_estimates(page: fitz.Page, estimates: dict) -> PageLayout:
    """Build a PageLayout from Claude's estimated barline positions.

    `estimates` has the shape:
    {
      "systems": [
        {
          "y_top_frac": 0.1, "y_bottom_frac": 0.25,
          "barline_x_fracs": [0.05, 0.25, 0.5, 0.75, 0.95]
        }, ...
      ]
    }
    """
    rect = page.rect
    layout = PageLayout(
        page_number=page.number,
        width=rect.width,
        height=rect.height,
        has_vector_barlines=False,
    )

    for idx, sys_est in enumerate(estimates.get("systems", [])):
        y_top = sys_est["y_top_frac"] * rect.height
        y_bottom = sys_est["y_bottom_frac"] * rect.height
        barlines = []
        for xf in sys_est.get("barline_x_fracs", []):
            bx = xf * rect.width
            barlines.append(Barline(x=bx, y_top=y_top, y_bottom=y_bottom))
        if len(barlines) >= 2:
            layout.systems.append(StaffSystem(
                system_index=idx,
                y_top=y_top,
                y_bottom=y_bottom,
                barlines=barlines,
            ))

    return layout


# ---------------------------------------------------------------------------
# Notation conversion
# ---------------------------------------------------------------------------

_ANGLO_ROOTS = re.compile(r'^([A-G])')
_LATIN_ROOTS = re.compile(r'^(Do|Re|Mi|Fa|Sol|La|Si)', re.IGNORECASE)


def convert_notation(chord: str, target: str) -> str:
    """Convert a chord symbol between anglo and latin notation.

    target: 'anglo' or 'latin'
    """
    if target == "anglo":
        m = _LATIN_ROOTS.match(chord)
        if m:
            root_latin = m.group(1).capitalize()
            # Normalise Sol
            if root_latin == "Sol":
                root_latin = "Sol"
            rest = chord[m.end():]
            anglo_root = LATIN_TO_ANGLO.get(root_latin, root_latin)
            # Handle bass note after /
            if "/" in rest:
                before_slash, bass = rest.rsplit("/", 1)
                bass = convert_notation(bass, target)
                return anglo_root + before_slash + "/" + bass
            return anglo_root + rest
    elif target == "latin":
        m = _ANGLO_ROOTS.match(chord)
        if m:
            root = m.group(1)
            rest = chord[m.end():]
            latin_root = ANGLO_TO_LATIN.get(root, root)
            # Handle bass note after /
            if "/" in rest:
                before_slash, bass = rest.rsplit("/", 1)
                bass = convert_notation(bass, target)
                return latin_root + before_slash + "/" + bass
            return latin_root + rest
    return chord


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _text_width(text: str, font_size: float) -> float:
    """Approximate text width for Helvetica."""
    # Helvetica average char width ≈ 0.52 * font_size
    return len(text) * 0.52 * font_size


def _compute_font_size(text: str, measure_width: float, requested: float | str) -> float:
    """Compute the font size for a chord label.

    If requested is 'auto', scale so text fits in 80% of the measure width.
    Otherwise use the numeric value, clamped to MIN/MAX.
    """
    if isinstance(requested, (int, float)):
        return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, float(requested)))

    # Auto: fit in 80% of measure width
    available = measure_width * 0.80
    if not text:
        return DEFAULT_FONT_SIZE
    estimated = available / (len(text) * 0.52)
    return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, estimated))


def _has_text_at(page: fitz.Page, rect: fitz.Rect) -> bool:
    """Check if there's existing text in the given rectangle."""
    text_dict = page.get_text("dict", clip=rect)
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    return True
    return False


def _place_chords(
    page: fitz.Page,
    chord_texts: list[str],
    left_x: float,
    measure_width: float,
    base_y: float,
    color: tuple[float, float, float],
    font_size: float | str,
    direction: int,
) -> None:
    """Place a list of chord labels within a measure.

    direction: -1 = shift upward on collision (right hand),
               +1 = shift downward on collision (left hand).
    """
    n = len(chord_texts)
    if not n:
        return

    # Compute x positions
    positions: list[float] = []
    if n == 1:
        fs = _compute_font_size(chord_texts[0], measure_width, font_size)
        tw = _text_width(chord_texts[0], fs)
        positions.append(left_x + (measure_width - tw) / 2)
    else:
        for i in range(n):
            frac = (i + 1) / (n + 1)
            fs = _compute_font_size(chord_texts[i], measure_width / n, font_size)
            tw = _text_width(chord_texts[i], fs)
            positions.append(left_x + measure_width * frac - tw / 2)

    for i, chord_text in enumerate(chord_texts):
        fs = _compute_font_size(
            chord_text,
            measure_width if n == 1 else measure_width / n,
            font_size,
        )
        x = positions[i]
        y = base_y

        # Check for collision with existing text and shift away
        tw = _text_width(chord_text, fs)
        check_rect = fitz.Rect(x - 2, y - fs - 2, x + tw + 2, y + 2)
        attempts = 0
        while _has_text_at(page, check_rect) and attempts < 5:
            y += direction * (fs + 2)
            check_rect = fitz.Rect(x - 2, y - fs - 2, x + tw + 2, y + 2)
            attempts += 1

        # Clamp to page bounds
        y = max(fs + 2, min(y, page.rect.height - 2))

        page.insert_text(
            fitz.Point(x, y),
            chord_text,
            fontname=CHORD_FONT,
            fontsize=fs,
            color=color,
        )


def annotate_page(
    page: fitz.Page,
    layout: PageLayout,
    analysis: dict,
    notation: str = "anglo",
    font_size: float | str = "auto",
) -> None:
    """Insert chord annotations onto a PDF page.

    Right-hand chords (treble) are placed above the system in blue.
    Left-hand chords (bass) are placed below the system in green.
    """
    analysis_systems = analysis.get("systems", [])

    for sys_layout in layout.systems:
        # Match analysis system by index
        sys_analysis = None
        for sa in analysis_systems:
            if sa.get("system_index") == sys_layout.system_index:
                sys_analysis = sa
                break

        if sys_analysis is None:
            # Try sequential matching if indices don't align
            si = layout.systems.index(sys_layout)
            if si < len(analysis_systems):
                sys_analysis = analysis_systems[si]
            else:
                continue

        measures = sys_analysis.get("measures", [])
        num_layout_measures = sys_layout.measure_count

        for measure in measures:
            mi = measure.get("measure_index", 0)
            if mi >= num_layout_measures:
                if num_layout_measures > 0:
                    mi = num_layout_measures - 1
                else:
                    continue

            left_x = sys_layout.barlines[mi].x
            right_x = sys_layout.barlines[mi + 1].x
            measure_width = right_x - left_x

            # --- Combined chord symbols above system ---
            chords = measure.get("chords", [])
            # Fallback to old format if needed
            if not chords:
                chords = measure.get("right_hand", [])
            chord_texts = []
            for c in chords:
                ct = c.get("chord", "?")
                if notation == "latin":
                    ct = convert_notation(ct, "latin")
                elif notation == "anglo":
                    ct = convert_notation(ct, "anglo")
                chord_texts.append(ct)

            if chord_texts:
                y = sys_layout.y_top - PADDING_ABOVE_STAFF
                _place_chords(page, chord_texts, left_x, measure_width,
                              y, CHORD_COLOR_RH, font_size, direction=-1)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_annotated_pdf(doc: fitz.Document, input_path: str) -> str:
    """Save the annotated PDF as {name}_acordes.pdf and return the path."""
    p = Path(input_path)
    output_path = p.parent / f"{p.stem}_acordes.pdf"
    doc.save(str(output_path), garbage=4, deflate=True)
    return str(output_path)

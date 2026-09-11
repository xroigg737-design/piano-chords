"""Segmentació de sistemes i compassos en partitures escanejades (raster).

El PDF d'una partitura escanejada no té gràfics vectorials: pentagrames i
barres de compàs només existeixen com a píxels. Aquest mòdul els detecta per
projecció de tinta sobre la imatge en escala de grisos, de manera que es pot
retallar cada sistema per separat abans d'enviar-lo a Claude Vision.

Per què cal: l'API redimensiona qualsevol imatge perquè el costat llarg no
passi de ~1568 px. Una pàgina sencera a 300 DPI (2100x3250) hi arriba reduïda a
uns 120 DPI efectius i els caps de nota queden de 4-5 px: impossible llegir-hi
les altures amb fiabilitat, i d'aquí surten les notes inventades. Retallant
sistema a sistema —i partint els sistemes amples per una barra de compàs— cada
imatge aprofita tot el pressupost de píxels i el detall es multiplica per 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fitz
import numpy as np

# DPI de treball per a la detecció (no per al retall final)
DETECT_DPI = 150

# Llindars expressats en fracció d'amplada/alçada per ser independents de la
# resolució de treball.
_INK_LEVEL = 160          # < 160 = píxel amb tinta
_BLANK_ROW_FRAC = 0.01    # fila "buida" si té menys d'1% d'amplada amb tinta
_STAFF_ROW_FRACS = (0.30, 0.22, 0.15)  # fraccions per detectar línies de pentagrama
_BARLINE_HEIGHT_FRAC = 0.85            # fracció del sistema que ha de cobrir una barra

# L'API redimensiona les imatges perquè el costat llarg no passi d'uns 1568 px
# i l'àrea d'uns 1,15 Mpx. Ens hi quedem just per sota per no perdre detall.
MAX_EDGE_PX = 1500
MAX_PIXELS = 1_100_000
MARGIN_PT = 12.0
JPEG_QUALITY = 88
KEYSIG_MAX_WIDTH_PT = 70.0


@dataclass
class SystemBand:
    """Un sistema (pentagrama doble de piano) dins d'una pàgina."""

    index: int                 # 0-based dins de la pàgina
    y_top: float               # punts PDF
    y_bottom: float
    x_left: float
    x_right: float
    barlines: list[float] = field(default_factory=list)  # x en punts PDF

    @property
    def height(self) -> float:
        return self.y_bottom - self.y_top

    @property
    def measure_spans(self) -> list[tuple[float, float]]:
        """Trams (x0, x1) entre barres de compàs consecutives."""
        xs = self.barlines
        if len(xs) < 2:
            return [(self.x_left, self.x_right)]
        return [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]

    @property
    def measure_count(self) -> int:
        return len(self.measure_spans)


# ---------------------------------------------------------------------------
# Detecció
# ---------------------------------------------------------------------------

def _render_gray(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Trams consecutius de True dins d'una màscara 1D."""
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _staff_line_rows(ink: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Files que són línia de pentagrama, i espaiat típic entre línies."""
    h, w = ink.shape
    row_ink = ink.sum(axis=1)
    groups: list[tuple[int, int]] = []
    for frac in _STAFF_ROW_FRACS:
        cand = _runs(row_ink > frac * w)
        # una línia de pentagrama és prima; descartem blocs gruixuts (text, caixes)
        groups = [g for g in cand if (g[1] - g[0]) < 0.02 * h]
        if len(groups) >= 8:
            break
    if len(groups) < 2:
        return groups, 0.0
    centers = [(a + b) / 2 for a, b in groups]
    diffs = sorted(centers[i + 1] - centers[i] for i in range(len(centers) - 1))
    spacing = diffs[len(diffs) // 2]
    return groups, spacing


def _staves_from_lines(
    line_groups: list[tuple[int, int]], spacing: float
) -> list[tuple[int, int]]:
    """Agrupa les línies detectades en pentagrames (normalment 5 línies)."""
    if not line_groups:
        return []
    staves: list[tuple[int, int]] = []
    cur_top, cur_bot = line_groups[0]
    for a, b in line_groups[1:]:
        if a - cur_bot <= 1.8 * spacing:
            cur_bot = b
        else:
            staves.append((cur_top, cur_bot))
            cur_top, cur_bot = a, b
    staves.append((cur_top, cur_bot))
    # un pentagrama fa uns 4 espais d'alt; la resta són restes de text
    return [st for st in staves if (st[1] - st[0]) >= 2.5 * spacing]


def _content_blocks(ink: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Blocs de files amb contingut, fusionant els separats per un forat petit."""
    h, w = ink.shape
    row_ink = ink.sum(axis=1)
    blocks = _runs(row_ink >= _BLANK_ROW_FRAC * w)
    merged: list[tuple[int, int]] = []
    for b in blocks:
        if merged and b[0] - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged


def _split_by_staves(
    y0: int, y1: int, centers: list[float], staves_per_system: int
) -> list[tuple[int, int]]:
    """Parteix un bloc que conté més d'un sistema pel forat entre pentagrames."""
    inside = [c for c in centers if y0 <= c <= y1]
    n_sys = max(1, round(len(inside) / staves_per_system))
    if n_sys < 2:
        return [(y0, y1)]
    out: list[tuple[int, int]] = []
    top = y0
    for k in range(1, n_sys):
        idx = k * staves_per_system
        if idx >= len(inside):
            break
        cut = int((inside[idx - 1] + inside[idx]) / 2)
        out.append((top, cut))
        top = cut + 1
    out.append((top, y1))
    return out


def _blocks_to_systems(
    blocks: list[tuple[int, int]],
    staves: list[tuple[int, int]],
    spacing: float,
    staves_per_system: int = 2,
) -> list[tuple[int, int]]:
    """Converteix els blocs de contingut en sistemes de pentagrama doble.

    Un bloc és un tros de pàgina entre files buides. Normalment un bloc ja és un
    sistema sencer, però hi ha partitures on el forat entre la mà dreta i la mà
    esquerra és prou gran per partir-lo en dos, i n'hi ha on dos sistemes queden
    enganxats en un sol bloc. Comptar quants pentagrames cauen dins de cada bloc
    desfà tots dos casos sense dependre de cap llindar de distància.
    """
    centers = [(a + b) / 2 for a, b in staves]

    def count_in(y0: int, y1: int) -> int:
        return sum(1 for c in centers if y0 <= c <= y1)

    # blocs sense cap pentagrama = títol, indicacions, número de pàgina
    music = [b for b in blocks if count_in(*b) > 0]
    if not music:
        return []

    heights = sorted(b[1] - b[0] for b in music)
    typical_h = heights[len(heights) // 2]

    systems: list[tuple[int, int]] = []
    i = 0
    while i < len(music):
        y0, y1 = music[i]
        n = count_in(y0, y1)
        # ajuntem blocs mentre no hi hagi els pentagrames d'un sistema sencer.
        # Si el bloc ja fa l'alçada d'un sistema, no l'ajuntem amb res: el que
        # passa és que un pentagrama tenia les línies massa fluixes per detectar.
        while (
            n < staves_per_system
            and i + 1 < len(music)
            and (y1 - y0) < 0.75 * typical_h
        ):
            nxt = music[i + 1]
            if nxt[0] - y1 > 5 * spacing:
                break  # massa lluny: és el sistema següent, no la mà esquerra
            y1 = nxt[1]
            n = count_in(y0, y1)
            i += 1
        if n >= 2 * staves_per_system:
            systems.extend(_split_by_staves(y0, y1, centers, staves_per_system))
        else:
            systems.append((y0, y1))
        i += 1
    return systems


def _close_vertical(mask: np.ndarray, radius: int) -> np.ndarray:
    """Tancament morfològic vertical: reomple forats de fins a 2*radius files."""
    grown = mask.copy()
    for k in range(1, radius + 1):
        grown[k:] |= mask[:-k]
        grown[:-k] |= mask[k:]
    shrunk = grown.copy()
    for k in range(1, radius + 1):
        shrunk[k:] &= grown[:-k]
        shrunk[:-k] &= grown[k:]
    return shrunk


def _detect_barlines(ink: np.ndarray, y0: int, y1: int, min_gap: int) -> list[int]:
    """Columnes amb tinta vertical CONTÍNUA de dalt a baix del sistema.

    Es mira la ratlla vertical ininterrompuda més llarga de cada columna, no la
    quantitat total de tinta: així les pliques, els lligats i les barres de
    corxera (que tenen molta tinta però no travessen el sistema) no compten com
    a barra de compàs. La banda es dilata 2 px en horitzontal i es tanquen els
    forats verticals perquè els escanejos torcen i trenquen les barres.
    """
    band = ink[y0:y1 + 1]
    h, w = band.shape
    if h == 0:
        return []
    dilated = band.copy()
    for shift in (1, 2):
        dilated[:, shift:] |= band[:, :-shift]
        dilated[:, :-shift] |= band[:, shift:]
    dilated = _close_vertical(dilated, 3)

    run = np.zeros(w, dtype=np.int32)
    best = np.zeros(w, dtype=np.int32)
    for row in dilated:
        run = (run + 1) * row
        np.maximum(best, run, out=best)

    runs = _runs(best > _BARLINE_HEIGHT_FRAC * h)
    centers = [(a + b) // 2 for a, b in runs]
    # les barres dobles (fi de secció, repetició) donen dues columnes juntes:
    # ens quedem amb una de sola per no inventar compassos buits
    out: list[int] = []
    for c in centers:
        if out and c - out[-1] < min_gap:
            out[-1] = (out[-1] + c) // 2
            continue
        out.append(c)
    return out


def _clean_barlines(xs: list[int]) -> list[int]:
    """Treu les barres falses (pliques llargues, claudàtors) que parteixen un
    compàs en dos trossos massa estrets per ser-ho de debò."""
    if len(xs) < 4:
        return xs
    spans = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    ordered = sorted(spans)
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return xs
    out = list(xs)
    i = 1
    while i < len(out) - 1:
        span_left = out[i] - out[i - 1]
        span_right = out[i + 1] - out[i]
        if min(span_left, span_right) < 0.35 * median:
            out.pop(i if span_right <= span_left else i - 1)
            continue
        i += 1
    return out


def segment_systems(page: fitz.Page, dpi: int = DETECT_DPI) -> list[SystemBand]:
    """Detecta els sistemes d'una pàgina i les barres de compàs de cadascun.

    Retorna coordenades en punts PDF (no píxels), llestes per a `fitz.Rect`.
    """
    img = _render_gray(page, dpi)
    ink = img < _INK_LEVEL
    h, w = ink.shape
    scale = 72.0 / dpi  # píxels -> punts

    line_groups, spacing = _staff_line_rows(ink)
    if spacing <= 0:
        return []
    staves = _staves_from_lines(line_groups, spacing)
    if not staves:
        return []

    blocks = _content_blocks(ink, max(4, int(1.4 * spacing)))
    bands = _blocks_to_systems(blocks, staves, spacing)

    min_gap = max(3, int(0.9 * spacing))
    systems: list[SystemBand] = []
    for i, (y0, y1) in enumerate(bands):
        inside = [st for st in staves if y0 <= (st[0] + st[1]) / 2 <= y1]
        if inside:
            # la barra va exactament de la línia de dalt del primer pentagrama
            # a la de baix de l'últim; si hi afegim les digitacions, cap columna
            # no arriba mai a l'alçada mínima
            s_top, s_bot = inside[0][0], inside[-1][1]
        else:
            s_top, s_bot = y0, y1

        cols = ink[y0:y1 + 1].sum(axis=0)
        nz = np.nonzero(cols)[0]
        x_left = float(nz[0]) if len(nz) else 0.0
        x_right = float(nz[-1]) if len(nz) else float(w - 1)

        xs = _detect_barlines(ink, s_top, s_bot, min_gap)
        xs = [x for x in xs if x_left - 1 <= x <= x_right + 1]
        xs = _clean_barlines(xs)
        if not xs or xs[0] > x_left + 2 * spacing:
            xs.insert(0, int(x_left))
        if xs[-1] < x_right - 2 * spacing:
            xs.append(int(x_right))

        systems.append(
            SystemBand(
                index=i,
                y_top=y0 * scale,
                y_bottom=y1 * scale,
                x_left=x_left * scale,
                x_right=x_right * scale,
                barlines=[x * scale for x in xs],
            )
        )
    return systems


def system_chunks(
    system: SystemBand,
    max_width_pt: float = 290.0,
) -> list[tuple[float, float, int, int]]:
    """Parteix un sistema en trossos prou estrets per llegir-hi les notes.

    Els talls es reparteixen per amplada i després s'encaixen a la barra de
    compàs més propera, de manera que un sistema amb barres mal detectades
    (acords densos, pliques que travessen tot el sistema) es parteix igualment
    per un lloc raonable i no pel mig d'un cap de nota.

    Retorna (x0, x1, primer_compàs_local, nombre_de_compassos), amb el compàs
    local comptat des de 0 dins del sistema.
    """
    spans = system.measure_spans
    if not spans:
        return []
    total_w = system.x_right - system.x_left
    if total_w <= max_width_pt or len(spans) < 2:
        return [(system.x_left, system.x_right, 0, len(spans))]

    n_chunks = min(max(2, int(np.ceil(total_w / max_width_pt))), len(spans))
    # índexs de barra (1..len(spans)-1) on tallaríem si repartíssim per amplada
    cuts: list[int] = []
    for k in range(1, n_chunks):
        target = system.x_left + total_w * k / n_chunks
        best = min(
            range(1, len(spans)),
            key=lambda idx: abs(spans[idx][0] - target),
        )
        if best not in cuts:
            cuts.append(best)
    cuts.sort()

    chunks: list[tuple[float, float, int, int]] = []
    start = 0
    for cut in cuts + [len(spans)]:
        if cut <= start:
            continue
        chunks.append((spans[start][0], spans[cut - 1][1], start, cut - start))
        start = cut
    return chunks


# ---------------------------------------------------------------------------
# Render dels retalls per enviar a Claude Vision
# ---------------------------------------------------------------------------

def _dpi_for(width_pt: float, height_pt: float) -> int:
    """DPI més alt que encara cap dins del pressupost de píxels de l'API."""
    if width_pt <= 0 or height_pt <= 0:
        return 150
    by_edge = MAX_EDGE_PX * 72.0 / max(width_pt, height_pt)
    by_area = (MAX_PIXELS / (width_pt * height_pt)) ** 0.5 * 72.0
    return max(72, int(min(by_edge, by_area)))


def _pixmap_image(page: fitz.Page, clip: fitz.Rect, dpi: int):
    from io import BytesIO

    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
    return Image.open(BytesIO(pix.tobytes("png"))).convert("L")


def _to_jpeg(img) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def render_chunk(
    page: fitz.Page,
    system: SystemBand,
    chunk: tuple[float, float, int, int],
) -> bytes:
    """Renderitza un tros de sistema, amb la clau i l'armadura al davant.

    Si el tros no comença a l'esquerra del sistema, s'hi enganxa la franja
    inicial (clau, armadura i indicació de compàs) perquè Claude sàpiga sempre
    en quina tonalitat està llegint les notes. Surt en JPEG i en escala de
    grisos: en PNG cada retall d'un escaneig fa 1,3 MB i una partitura sencera
    no cabria en una sola petició.
    """
    from PIL import Image

    x0, x1, first_local, _ = chunk
    y0 = max(0.0, system.y_top - MARGIN_PT)
    y1 = min(page.rect.height, system.y_bottom + MARGIN_PT)

    needs_header = first_local > 0 and len(system.barlines) > 1
    header_w = 0.0
    if needs_header:
        header_w = min(
            0.30 * (x1 - x0),
            max(18.0, system.barlines[1] - system.x_left),
        )

    total_w = (x1 - x0) + header_w
    dpi = _dpi_for(total_w, y1 - y0)
    main = _pixmap_image(page, fitz.Rect(x0, y0, x1, y1), dpi)
    if not needs_header or header_w <= 0:
        return _to_jpeg(main)

    head = _pixmap_image(
        page, fitz.Rect(system.x_left, y0, system.x_left + header_w, y1), dpi
    )
    gap = 10
    out = Image.new(
        "L", (head.width + gap + main.width, max(head.height, main.height)), 255
    )
    out.paste(head, (0, 0))
    out.paste(main, (head.width + gap, 0))
    return _to_jpeg(out)


def render_key_signature(page: fitz.Page, system: SystemBand) -> bytes:
    """Retall ampliat de la clau, l'armadura i la indicació de compàs.

    Comptar sostinguts i bemolls és el pas que invalida tota l'anàlisi si falla,
    i en un retall de sistema sencer l'armadura hi ocupa quatre píxels. Aquí se
    n'agafa només la franja de l'esquerra, amb els dos pentagrames (l'armadura
    hi surt dues vegades, així es pot contrastar), i s'amplia tant com dóna de si.
    """
    x1 = system.x_left + KEYSIG_MAX_WIDTH_PT
    if len(system.barlines) > 1:
        x1 = min(x1, system.barlines[1])
    clip = fitz.Rect(
        max(0.0, system.x_left - 2),
        max(0.0, system.y_top - 2),
        min(page.rect.width, x1),
        min(page.rect.height, system.y_bottom + 2),
    )
    if clip.width <= 0 or clip.height <= 0:
        return b""
    dpi = int(min(1400 * 72 / clip.height, 900 * 72 / clip.width))
    return _to_jpeg(_pixmap_image(page, clip, max(72, dpi)))


def key_signature_crops(page: fitz.Page, max_systems: int = 3) -> list[bytes]:
    """Un retall d'armadura per cadascun dels primers sistemes de la pàgina."""
    out: list[bytes] = []
    for system in segment_systems(page)[:max_systems]:
        img = render_key_signature(page, system)
        if img:
            out.append(img)
    return out


def render_page_systems(page: fitz.Page, page_number: int) -> list[tuple[bytes, str]]:
    """Retalls d'una pàgina llestos per enviar, amb el peu de foto de cadascun."""
    systems = segment_systems(page)
    out: list[tuple[bytes, str]] = []
    for s in systems:
        for chunk in system_chunks(s):
            _, _, first_local, n = chunk
            png = render_chunk(page, s, chunk)
            header_note = (
                " (a l'esquerra s'hi ha repetit la clau i l'armadura del sistema; "
                "no compta com a compàs)"
                if first_local > 0
                else ""
            )
            if 2 <= s.measure_count <= 12:
                where = (
                    f"compassos {first_local + 1}-{first_local + n} d'aquest "
                    f"sistema (el sistema en té {s.measure_count})"
                )
            else:
                # la detecció de barres no és de fiar en aquest sistema
                where = "part esquerra" if first_local == 0 else "continuació"
            caption = (
                f"Pàgina {page_number}, sistema {s.index + 1} de {len(systems)}, "
                f"{where}{header_note}."
            )
            out.append((png, caption))
    return out

import random

import cv2
import numpy as np

from yatzar import ops
from yatzar.looks import register


def make_halftone(gray: np.ndarray, cell_size: int = 6, angle: float = 45.0, dot_gain: float = 1.15) -> np.ndarray:
    """
    Klassischer Winkel-Halbtonraster (wie im Zeitungsdruck): das Bild wird
    gedreht, in Zellen zerlegt und pro Zelle ein schwarzer Punkt gezeichnet,
    dessen Radius von der mittleren Dunkelheit der Zelle abhängt. Danach wird
    das Rasterbild zurückgedreht und auf die Originalgröße zugeschnitten.
    """
    h, w = gray.shape[:2]
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2))) + 2 * cell_size
    pad_top = (diag - h) // 2
    pad_left = (diag - w) // 2

    padded = cv2.copyMakeBorder(
        gray, pad_top, diag - h - pad_top, pad_left, diag - w - pad_left,
        cv2.BORDER_REPLICATE
    )

    rot_fwd = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rotated_src = cv2.warpAffine(
        padded, rot_fwd, (diag, diag),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    canvas = np.full((diag, diag), 255, dtype=np.uint8)
    n_cells = diag // cell_size
    for cy in range(n_cells):
        y0, y1 = cy * cell_size, cy * cell_size + cell_size
        for cx in range(n_cells):
            x0, x1 = cx * cell_size, cx * cell_size + cell_size
            darkness = 1.0 - (rotated_src[y0:y1, x0:x1].mean() / 255.0)
            radius = (cell_size / 2.0) * min(1.0, darkness * dot_gain)
            if radius < 0.4:
                continue
            center = (x0 + cell_size // 2, y0 + cell_size // 2)
            cv2.circle(canvas, center, int(round(radius)), 0, thickness=-1, lineType=cv2.LINE_AA)

    rot_back = cv2.getRotationMatrix2D((diag / 2, diag / 2), -angle, 1.0)
    unrotated = cv2.warpAffine(
        canvas, rot_back, (diag, diag),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    return unrotated[pad_top:pad_top + h, pad_left:pad_left + w]


@register("gazette_halftone")
def gazette_halftone(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    """
    Variante mit echtem gedrehtem Halbtonraster (statt sanftem SW-Korn wie bei
    `gazette`) — sichtbares Druckpunktmuster wie im klassischen Zeitungsdruck.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast_cfg = cfg.get("contrast", {})
    gray = ops.clip_uint8(
        gray.astype(np.float32) * float(contrast_cfg.get("alpha", 1.12))
        + float(contrast_cfg.get("beta", 0))
    )

    pre_blur_cfg = cfg.get("pre_blur", {})
    if pre_blur_cfg.get("enabled", False):
        k = ops.ensure_odd(int(pre_blur_cfg.get("kernel", 3)))
        gray = cv2.GaussianBlur(gray, (k, k), float(pre_blur_cfg.get("sigma", 0.8)))

    halftone_cfg = cfg.get("halftone", {})
    dotted = make_halftone(
        gray,
        cell_size=int(halftone_cfg.get("cell_size", 5)),
        angle=float(halftone_cfg.get("angle", 45.0)),
        dot_gain=float(halftone_cfg.get("dot_gain", 0.9)),
    )

    out = cv2.cvtColor(dotted, cv2.COLOR_GRAY2BGR)

    soft_blur_cfg = cfg.get("soft_blur", {})
    if soft_blur_cfg.get("enabled", False):
        k = ops.ensure_odd(int(soft_blur_cfg.get("kernel", 3)))
        out = cv2.GaussianBlur(out, (k, k), float(soft_blur_cfg.get("sigma", 0.2)))

    paper_cfg = cfg.get("paper_tone", {})
    if paper_cfg.get("enabled", False):
        out = ops.apply_paper_tone(out, float(paper_cfg.get("sepia_amount", 0.12)))

    grain_cfg = cfg.get("grain", {})
    if grain_cfg.get("enabled", False):
        out = ops.add_monochrome_grain(
            out,
            amount=float(grain_cfg.get("amount", 6)),
            opacity=float(grain_cfg.get("opacity", 0.06)),
        )

    vig_cfg = cfg.get("vignette", {})
    if vig_cfg.get("enabled", False):
        out = ops.add_vignette(out, float(vig_cfg.get("strength", 0.08)))

    out = ops.maybe_motion_blur(out, cfg, rng)

    return ops.clip_uint8(out)

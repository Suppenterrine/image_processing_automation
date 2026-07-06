import random

import numpy as np

from yatzar import ops
from yatzar.looks import register


@register("gazette")
def gazette(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    """
    Annäherung an alte Zeitungsfotos (19./frühes 20. Jhd.): wie tri_x_pushed,
    aber mit weicherer Schärfung und deutlich gröberem, klumpigem Korn statt
    feinem Pixelrauschen oder sichtbarem Druckraster — echte Abzüge dieser Ära
    wirkten körnig-weich, nicht wie ein aufgelöstes Halbtonmuster.
    """
    out = ops.to_bw(img)

    contrast_cfg = cfg.get("contrast", {})
    out = ops.add_contrast(
        out,
        alpha=float(contrast_cfg.get("alpha", 1.25)),
        beta=float(contrast_cfg.get("beta", -10)),
    )

    sharpen_amount = float(cfg.get("sharpen", {}).get("amount", 0.3))
    if sharpen_amount:
        out = ops.unsharp_mask(out, sharpen_amount)

    grain_cfg = cfg.get("grain", {})
    out = ops.add_coarse_grain(
        out,
        amount=float(grain_cfg.get("amount", 22)),
        opacity=float(grain_cfg.get("opacity", 0.24)),
        scale=int(grain_cfg.get("scale", 4)),
        rng=rng,
    )

    paper_cfg = cfg.get("paper_tone", {})
    if paper_cfg.get("enabled", False):
        out = ops.apply_paper_tone(out, float(paper_cfg.get("sepia_amount", 0.12)))

    out = ops.add_vignette(out, float(cfg.get("vignette", {}).get("strength", 0.22)))

    out = ops.maybe_motion_blur(out, cfg, rng)

    return ops.clip_uint8(out)

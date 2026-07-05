import random

import numpy as np

from yatzar import ops
from yatzar.looks import register


@register("tri_x_pushed")
def tri_x_pushed(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    """
    Scharfer, kontrastreicher SW-Look als Annäherung an Kodak Tri-X pushed.
    Kein physikalisch exakter Film-Emulator, aber gute stilistische Annäherung.
    """
    out = ops.to_bw(img)

    contrast_cfg = cfg.get("contrast", {})
    out = ops.add_contrast(
        out,
        alpha=float(contrast_cfg.get("alpha", 1.22)),
        beta=float(contrast_cfg.get("beta", -8)),
    )

    sharpen_amount = float(cfg.get("sharpen", {}).get("amount", 1.2))
    out = ops.unsharp_mask(out, sharpen_amount)

    grain_cfg = cfg.get("grain", {})
    out = ops.add_monochrome_grain(
        out,
        amount=float(grain_cfg.get("amount", 22)),
        opacity=float(grain_cfg.get("opacity", 0.18)),
    )

    vignette_strength = float(cfg.get("vignette", {}).get("strength", 0.18))
    out = ops.add_vignette(out, vignette_strength)

    out = ops.maybe_motion_blur(out, cfg, rng)

    return ops.clip_uint8(out)

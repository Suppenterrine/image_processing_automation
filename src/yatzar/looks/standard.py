import random

import numpy as np

from yatzar import ops
from yatzar.looks import register


@register("standard")
def standard(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    out = img.copy()

    dist_cfg = cfg.get("barrel_distortion", {})
    if dist_cfg.get("enabled", False):
        out = ops.barrel_distortion(
            out,
            k1=float(dist_cfg.get("k1", 0.028)),
            k2=float(dist_cfg.get("k2", 0.00008)),
        )

    blur_cfg = cfg.get("gaussian_blur", {})
    if blur_cfg.get("enabled", False):
        out = ops.gaussian_blur(
            out,
            kernel=int(blur_cfg.get("kernel", 15)),
            sigma=float(blur_cfg.get("sigma", 1.5)),
        )

    out = ops.maybe_motion_blur(out, cfg, rng)

    grain_cfg = cfg.get("grain", {})
    if grain_cfg.get("enabled", False):
        out = ops.add_monochrome_grain(
            out,
            amount=float(grain_cfg.get("amount", 16)),
            opacity=float(grain_cfg.get("opacity", 0.10)),
        )

    color_cfg = cfg.get("color", {})
    if color_cfg.get("enabled", False):
        out = ops.adjust_hsv(
            out,
            sat_mult=float(color_cfg.get("sat_mult", 1.0)),
            val_mult=float(color_cfg.get("val_mult", 1.0)),
        )

    tone_cfg = cfg.get("tone", {})
    if tone_cfg.get("enabled", False):
        out = ops.add_contrast(
            out,
            alpha=float(tone_cfg.get("alpha", 1.0)),
            beta=float(tone_cfg.get("beta", 0.0)),
        )

    vig_cfg = cfg.get("vignette", {})
    if vig_cfg.get("enabled", False):
        out = ops.add_vignette(out, float(vig_cfg.get("strength", 0.2)))

    return out

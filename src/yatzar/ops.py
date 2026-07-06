"""Zustandslose, wiederverwendbare Bildoperationen. Keine I/O, keine Config-Kenntnis."""

import random

import cv2
import numpy as np


def ensure_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def barrel_distortion(img: np.ndarray, k1: float = 0.028, k2: float = 0.00008) -> np.ndarray:
    """Barrel distortion / Fisheye."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = map_x - cx
    dy = map_y - cy

    r = np.sqrt(dx * dx + dy * dy)
    r_max = np.sqrt(cx * cx + cy * cy)
    r_norm = r / max(r_max, 1e-6)

    factor = 1.0 + k1 * r_norm + k2 * (r_norm ** 2)

    map_x_new = cx + dx * factor
    map_y_new = cy + dy * factor

    return cv2.remap(
        img,
        map_x_new.astype(np.float32),
        map_y_new.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def gaussian_blur(img: np.ndarray, kernel: int = 15, sigma: float = 1.5) -> np.ndarray:
    kernel = ensure_odd(kernel)
    return cv2.GaussianBlur(img, (kernel, kernel), sigma)


def motion_blur(img: np.ndarray, length: int = 15, angle: float = 0.0) -> np.ndarray:
    """Linear motion blur with arbitrary angle."""
    length = max(1, int(length))
    size = ensure_odd(length)

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0

    center = (size / 2 - 0.5, size / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def maybe_motion_blur(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    blur_cfg = cfg.get("motion_blur", {})
    if not blur_cfg.get("enabled", False):
        return img

    chance = float(blur_cfg.get("chance", 0.0))
    if rng.random() > chance:
        return img

    length_min = int(blur_cfg.get("length_min", 5))
    length_max = int(blur_cfg.get("length_max", 25))
    angle_min = float(blur_cfg.get("angle_min", 0.0))
    angle_max = float(blur_cfg.get("angle_max", 180.0))

    length = rng.randint(length_min, length_max)
    angle = rng.uniform(angle_min, angle_max)

    return motion_blur(img, length=length, angle=angle)


def unsharp_mask(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def add_monochrome_grain(img: np.ndarray, amount: float = 16.0, opacity: float = 0.10, rng: random.Random | None = None) -> np.ndarray:
    """Monochrome film grain without uint8 wraparound."""
    img_f = img.astype(np.float32)
    h, w = img.shape[:2]
    rng = rng or random.Random()
    noise = np.array(
        [[rng.normalvariate(0.0, amount) for _ in range(w)] for _ in range(h)],
        dtype=np.float32,
    )
    noise = noise[:, :, np.newaxis]
    noise = np.repeat(noise, 3, axis=2)

    grain_layer = img_f + noise
    out = cv2.addWeighted(img_f, 1.0 - opacity, grain_layer, opacity, 0.0)
    return clip_uint8(out)


def add_coarse_grain(img: np.ndarray, amount: float = 20.0, opacity: float = 0.22, scale: int = 4, rng: random.Random | None = None) -> np.ndarray:
    """
    Gröberes, klumpiges Filmkorn: das Rauschen wird in reduzierter Auflösung
    erzeugt und wieder hochskaliert, statt pixelweise — dadurch entstehen
    größere, weichere Körner statt feines Pixelrauschen.
    """
    img_f = img.astype(np.float32)
    h, w = img.shape[:2]
    small_h, small_w = max(1, h // scale), max(1, w // scale)

    rng = rng or random.Random()
    noise_small = np.array(
        [[rng.normalvariate(0.0, amount) for _ in range(small_w)] for _ in range(small_h)],
        dtype=np.float32,
    )
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = noise[:, :, np.newaxis]
    noise = np.repeat(noise, 3, axis=2)

    grain_layer = img_f + noise
    out = cv2.addWeighted(img_f, 1.0 - opacity, grain_layer, opacity, 0.0)
    return clip_uint8(out)

def adjust_hsv(img: np.ndarray, sat_mult: float = 1.0, val_mult: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= sat_mult
    hsv[:, :, 2] *= val_mult
    return cv2.cvtColor(clip_uint8(hsv), cv2.COLOR_HSV2BGR)


def add_contrast(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """alpha = contrast, beta = brightness"""
    return clip_uint8(img.astype(np.float32) * alpha + beta)


def add_vignette(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w / 2.0)
    kernel_y = cv2.getGaussianKernel(h, h / 2.0)
    mask = kernel_y @ kernel_x.T
    mask = mask / mask.max()

    vignette = 1.0 - strength * (1.0 - mask)
    out = img.astype(np.float32).copy()
    for c in range(3):
        out[:, :, c] *= vignette

    return clip_uint8(out)


def to_bw(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_paper_tone(img: np.ndarray, amount: float = 0.12) -> np.ndarray:
    """Leichte warme Tönung, wie sie gescanntes/gealtertes Zeitungspapier zeigt."""
    out = img.astype(np.float32)
    out[:, :, 0] *= (1.0 - amount * 0.5)  # Blau zurücknehmen
    out[:, :, 2] *= (1.0 + amount * 0.3)  # Rot leicht anheben
    return clip_uint8(out)

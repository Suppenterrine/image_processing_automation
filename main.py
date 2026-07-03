import argparse
import glob
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def ensure_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def add_monochrome_grain(img: np.ndarray, amount: float = 16.0, opacity: float = 0.10) -> np.ndarray:
    """Monochrome film grain without uint8 wraparound."""
    img_f = img.astype(np.float32)
    h, w = img.shape[:2]

    noise = np.random.normal(0.0, amount, (h, w, 1)).astype(np.float32)
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


def tri_x_pushed(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Scharfer, kontrastreicher SW-Look als Annäherung an Kodak Tri-X pushed.
    Kein physikalisch exakter Film-Emulator, aber gute stilistische Annäherung.
    """
    out = to_bw(img)

    # Mikro-Kontrast / Push
    out = add_contrast(
        out,
        alpha=float(cfg.get("contrast_alpha", 1.22)),
        beta=float(cfg.get("contrast_beta", -8))
    )

    # Leichte lokale Schärfung
    sharp_amount = float(cfg.get("sharpen_amount", 1.2))
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.0 + sharp_amount, blur, -sharp_amount, 0)

    # Korn
    out = add_monochrome_grain(
        out,
        amount=float(cfg.get("grain_amount", 22)),
        opacity=float(cfg.get("grain_opacity", 0.18))
    )

    # Optional subtile Vignette
    vignette_strength = float(cfg.get("vignette_strength", 0.18))
    out = add_vignette(out, vignette_strength)

    return clip_uint8(out)


def add_coarse_grain(img: np.ndarray, amount: float = 20.0, opacity: float = 0.22, scale: int = 4) -> np.ndarray:
    """
    Gröberes, klumpiges Filmkorn: das Rauschen wird in reduzierter Auflösung
    erzeugt und wieder hochskaliert, statt pixelweise — dadurch entstehen
    größere, weichere Körner statt feines Pixelrauschen.
    """
    img_f = img.astype(np.float32)
    h, w = img.shape[:2]
    small_h, small_w = max(1, h // scale), max(1, w // scale)

    noise_small = np.random.normal(0.0, amount, (small_h, small_w)).astype(np.float32)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)

    grain_layer = img_f + noise
    out = cv2.addWeighted(img_f, 1.0 - opacity, grain_layer, opacity, 0.0)
    return clip_uint8(out)


def gazette(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Annäherung an alte Zeitungsfotos (19./frühes 20. Jhd.): wie tri_x_pushed,
    aber mit weicherer Schärfung und deutlich gröberem, klumpigem Korn statt
    feinem Pixelrauschen oder sichtbarem Druckraster — echte Abzüge dieser Ära
    wirkten körnig-weich, nicht wie ein aufgelöstes Halbtonmuster.
    """
    out = to_bw(img)

    out = add_contrast(
        out,
        alpha=float(cfg.get("contrast_alpha", 1.25)),
        beta=float(cfg.get("contrast_beta", -10))
    )

    sharpen_amount = float(cfg.get("sharpen_amount", 0.3))
    if sharpen_amount:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

    out = add_coarse_grain(
        out,
        amount=float(cfg.get("grain_amount", 22)),
        opacity=float(cfg.get("grain_opacity", 0.24)),
        scale=int(cfg.get("grain_scale", 4))
    )

    paper_cfg = cfg.get("paper_tone", {})
    if paper_cfg.get("enabled", False):
        out = apply_paper_tone(out, float(paper_cfg.get("sepia_amount", 0.12)))

    out = add_vignette(out, float(cfg.get("vignette_strength", 0.22)))

    return clip_uint8(out)


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


def apply_paper_tone(img: np.ndarray, amount: float = 0.12) -> np.ndarray:
    """Leichte warme Tönung, wie sie gescanntes/gealtertes Zeitungspapier zeigt."""
    out = img.astype(np.float32)
    out[:, :, 0] *= (1.0 - amount * 0.5)  # Blau zurücknehmen
    out[:, :, 2] *= (1.0 + amount * 0.3)  # Rot leicht anheben
    return clip_uint8(out)


def gazette_halftone(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Variante mit echtem gedrehtem Halbtonraster (statt sanftem SW-Korn wie bei
    `gazette`) — sichtbares Druckpunktmuster wie im klassischen Zeitungsdruck.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = clip_uint8(
        gray.astype(np.float32) * float(cfg.get("contrast_alpha", 1.12))
        + float(cfg.get("contrast_beta", 0))
    )

    pre_blur_cfg = cfg.get("pre_blur", {})
    if pre_blur_cfg.get("enabled", False):
        k = ensure_odd(int(pre_blur_cfg.get("kernel", 3)))
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
        k = ensure_odd(int(soft_blur_cfg.get("kernel", 3)))
        out = cv2.GaussianBlur(out, (k, k), float(soft_blur_cfg.get("sigma", 0.2)))

    paper_cfg = cfg.get("paper_tone", {})
    if paper_cfg.get("enabled", False):
        out = apply_paper_tone(out, float(paper_cfg.get("sepia_amount", 0.12)))

    grain_cfg = cfg.get("grain", {})
    if grain_cfg.get("enabled", False):
        out = add_monochrome_grain(
            out,
            amount=float(grain_cfg.get("amount", 6)),
            opacity=float(grain_cfg.get("opacity", 0.06))
        )

    vig_cfg = cfg.get("vignette", {})
    if vig_cfg.get("enabled", False):
        out = add_vignette(out, float(vig_cfg.get("strength", 0.08)))

    return clip_uint8(out)


def apply_standard_pipeline(img: np.ndarray, look_cfg: dict, rng: random.Random) -> np.ndarray:
    out = img.copy()

    # Distortion
    dist_cfg = look_cfg.get("barrel_distortion", {})
    if dist_cfg.get("enabled", False):
        out = barrel_distortion(
            out,
            k1=float(dist_cfg.get("k1", 0.028)),
            k2=float(dist_cfg.get("k2", 0.00008))
        )

    # Base blur
    blur_cfg = look_cfg.get("gaussian_blur", {})
    if blur_cfg.get("enabled", False):
        out = gaussian_blur(
            out,
            kernel=int(blur_cfg.get("kernel", 15)),
            sigma=float(blur_cfg.get("sigma", 1.5))
        )

    # Optional random motion blur
    out = maybe_motion_blur(out, look_cfg, rng)

    # Grain
    grain_cfg = look_cfg.get("grain", {})
    if grain_cfg.get("enabled", False):
        out = add_monochrome_grain(
            out,
            amount=float(grain_cfg.get("amount", 16)),
            opacity=float(grain_cfg.get("opacity", 0.10))
        )

    # HSV / color tuning
    color_cfg = look_cfg.get("color", {})
    if color_cfg.get("enabled", False):
        out = adjust_hsv(
            out,
            sat_mult=float(color_cfg.get("sat_mult", 1.0)),
            val_mult=float(color_cfg.get("val_mult", 1.0))
        )

    # Contrast / brightness
    tone_cfg = look_cfg.get("tone", {})
    if tone_cfg.get("enabled", False):
        out = add_contrast(
            out,
            alpha=float(tone_cfg.get("alpha", 1.0)),
            beta=float(tone_cfg.get("beta", 0.0))
        )

    # Vignette
    vig_cfg = look_cfg.get("vignette", {})
    if vig_cfg.get("enabled", False):
        out = add_vignette(out, float(vig_cfg.get("strength", 0.2)))

    return out


def collect_input_files(input_dir: str) -> list[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for pattern in patterns:
        files.extend(Path(input_dir).glob(pattern))
    return sorted(files)


def save_image(output_dir: Path, look_name: str, input_path: Path, img: np.ndarray, ext: str = ".jpg") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{look_name}_{input_path.stem}{ext}"
    out_path = output_dir / out_name
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"Konnte Bild nicht speichern: {out_path}")
    return out_path


def create_gif_from_outputs(output_dir: Path, pattern: str = "*.jpg", gif_name: str = "animation.gif", delay: int = 15):
    files = sorted(output_dir.glob(pattern))
    if not files:
        print("Keine Dateien für GIF gefunden.")
        return

    cmd = ["magick", "-delay", str(delay), "-loop", "0"]
    cmd.extend([f.name for f in files])
    cmd.append(gif_name)

    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        shell=False
    )

    if result.returncode != 0:
        print("ImageMagick Fehler:")
        print(result.stderr.strip())
    else:
        print(f"GIF erstellt: {output_dir / gif_name}")


def main():
    parser = argparse.ArgumentParser(description="Apply configurable aesthetic looks to images.")
    parser.add_argument("--config", default="looks.yaml", help="Pfad zur YAML-Konfiguration")
    parser.add_argument("--look", required=True, help="Name des Looks aus der YAML")
    parser.add_argument("--input", default="img", help="Input-Ordner")
    parser.add_argument("--output", default="output", help="Output-Ordner")
    parser.add_argument("--gif", action="store_true", help="Erzeuge zusätzlich ein GIF aus den Ausgabe-Bildern")
    parser.add_argument("--seed", type=int, default=None, help="Optionaler Seed für reproduzierbare Random-Effekte")
    args = parser.parse_args()

    cfg = load_config(args.config)
    looks = cfg.get("looks", {})
    if args.look not in looks:
        raise ValueError(f"Look '{args.look}' nicht gefunden. Verfügbar: {', '.join(looks.keys())}")

    look_cfg = looks[args.look]
    rng = random.Random(args.seed)

    input_files = collect_input_files(args.input)
    if not input_files:
        print(f"Keine Bilder in '{args.input}' gefunden.")
        return

    output_dir = Path(args.output)
    print(f"Verarbeite {len(input_files)} Bilder mit Look: {args.look}")

    for file_path in input_files:
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Fehler beim Laden: {file_path}")
            continue

        look_type = look_cfg.get("type", "standard")

        if look_type == "tri_x_pushed":
            out = tri_x_pushed(img, look_cfg)
            # optional noch random motion blur oben drauf
            out = maybe_motion_blur(out, look_cfg, rng)
        elif look_type == "gazette":
            out = gazette(img, look_cfg)
            out = maybe_motion_blur(out, look_cfg, rng)
        elif look_type == "gazette_halftone":
            out = gazette_halftone(img, look_cfg)
            out = maybe_motion_blur(out, look_cfg, rng)
        else:
            out = apply_standard_pipeline(img, look_cfg, rng)

        saved = save_image(output_dir, args.look, file_path, out, ext=".jpg")
        print(f"OK {saved.name}")

    if args.gif:
        create_gif_from_outputs(output_dir, pattern=f"{args.look}_*.jpg")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen (Strg+C).")
        sys.exit(130)
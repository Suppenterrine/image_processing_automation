import random

import cv2
import numpy as np

from yatzar import ops
from yatzar.looks import register


def compute_structure_tensor_flow(
    gray: np.ndarray,
    sigma_grad: float,
    sigma_tensor: float,
    default_angle_deg: float = 45.0,
    coherence_floor: float = 0.15,
    coherence_gamma: float = 1.5,
    energy_floor: float = 0.08,
    flow_smooth_sigma: float = 0.0,
):
    """
    Annäherung an Edge Tangent Flow (Kang et al., "Coherent Line Drawing"): statt der
    teuren iterativen ETF-Verfeinerung wird der Struktur-Tensor selbst geglättet, was
    ein ähnlich konturfolgendes Richtungsfeld liefert. Wo keine echte Kante vorliegt,
    kippt die Richtung zu `default_angle_deg`, damit dort ein sauberes Standard-Hatching
    statt Gradienten-Rauschen entsteht.

    Zwei Signale entscheiden, ob eine Richtung "echt" ist:
    - Kohärenz (Eigenwert-Verhältnis) beschreibt, WIE gerichtet die Struktur lokal ist.
    - Energie (Eigenwert-Summe = Gradientstärke) beschreibt, OB überhaupt genug Kontrast
      da ist, um das zu beurteilen.
    Kohärenz allein reicht nicht: sie ist ein reines Verhältnis und daher blind für die
    absolute Stärke — auch eine flache Wand mit nur Sensor-/Putzrauschen kann per Zufall
    eine hohe Kohärenz zeigen (führt zu Fingerabdruck-artigen Wirbeln im Hatching). Erst
    die Kombination aus hoher Kohärenz UND hoher Energie (relativ zu den stärksten echten
    Kanten im Bild) zählt als vertrauenswürdige Richtung.
    Rückgabe: (Tangentenwinkel in rad, kombinierte Konfidenz 0..1) pro Pixel.
    """
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    if sigma_grad > 0:
        gx = cv2.GaussianBlur(gx, (0, 0), sigma_grad)
        gy = cv2.GaussianBlur(gy, (0, 0), sigma_grad)

    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_tensor)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_tensor)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_tensor)

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    diff = np.sqrt(np.maximum(0.0, (trace / 2.0) ** 2 - det))
    lambda1 = trace / 2.0 + diff
    lambda2 = trace / 2.0 - diff
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)

    coherence_adj = np.clip((coherence - coherence_floor) / max(1.0 - coherence_floor, 1e-6), 0.0, 1.0)
    coherence_adj = coherence_adj ** coherence_gamma

    energy_ref = float(np.percentile(trace, 95)) + 1e-6
    energy_norm = np.clip(trace / energy_ref, 0.0, 1.0)
    energy_conf = np.clip((energy_norm - energy_floor) / max(1.0 - energy_floor, 1e-6), 0.0, 1.0)

    confidence = coherence_adj * energy_conf

    theta_grad = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy) + 1e-12)
    theta_tangent = theta_grad + np.pi / 2.0

    # Winkel sind hier π-periodisch (eine Hatch-Linie hat keine Pfeilrichtung), daher
    # Mittelung in der verdoppelten Winkeldomäne statt naiver linearer Interpolation.
    default_angle = np.deg2rad(default_angle_deg)
    cos2 = np.cos(2.0 * theta_tangent) * confidence + np.cos(2.0 * default_angle) * (1.0 - confidence)
    sin2 = np.sin(2.0 * theta_tangent) * confidence + np.sin(2.0 * default_angle) * (1.0 - confidence)
    theta_blended = 0.5 * np.arctan2(sin2, cos2)

    if flow_smooth_sigma > 0:
        # Zusätzliche Annäherung an die iterative ETF-Verfeinerung: das bereits
        # vertrauenswürdigkeits-gewichtete Richtungsfeld wird selbst nochmal grossflächig
        # gemittelt (wieder in der Doppelwinkeldomäne), statt nur den Struktur-Tensor zu
        # glätten. Das drückt kleinteilige Richtungs-Wirbel (z.B. auf Haut, Holzmaserung)
        # zu einem größeren, ruhigeren Fluss, ohne echte starke Kanten zu verwässern.
        vx = confidence * np.cos(2.0 * theta_blended)
        vy = confidence * np.sin(2.0 * theta_blended)
        vx_b = cv2.GaussianBlur(vx, (0, 0), flow_smooth_sigma)
        vy_b = cv2.GaussianBlur(vy, (0, 0), flow_smooth_sigma)
        w_b = cv2.GaussianBlur(confidence, (0, 0), flow_smooth_sigma) + 1e-6
        theta_blended = 0.5 * np.arctan2(vy_b, vx_b)
        confidence = np.clip(np.sqrt(vx_b ** 2 + vy_b ** 2) / w_b, 0.0, 1.0)

    return theta_blended, confidence


def tone_to_duty(tone: np.ndarray, t_hi: float, t_lo: float, d_max: float) -> np.ndarray:
    """Deckungsgrad-Rampe: 0 bei tone >= t_hi, wächst linear bis d_max bei tone <= t_lo."""
    span = max(t_hi - t_lo, 1e-6)
    duty = d_max * (t_hi - tone) / span
    return np.clip(duty, 0.0, d_max)


def hatch_layer(theta: np.ndarray, xs: np.ndarray, ys: np.ndarray, angle_offset: float, pitch: float, duty: np.ndarray) -> np.ndarray:
    """Ein Hatch-Layer: Tinte, wo die Phase quer zur Linienrichtung unter `duty` (Deckungsgrad) liegt."""
    phi = theta + angle_offset
    coord = -xs * np.sin(phi) + ys * np.cos(phi)
    phase = np.mod(coord, pitch) / pitch
    return phase < duty


@register("engraving")
def engraving(img: np.ndarray, cfg: dict, rng: random.Random) -> np.ndarray:
    """
    Annäherung an Stahlstich-/Holzstich-Gravur (Hetzel-Plates): Linienrichtung folgt
    pro Pixel dem lokalen Struktur-Tensor (konturfolgend statt fixer Winkel), Tonwert
    steuert die Liniendichte über gestaffelte Hatch-Layer — Haupt-Hatch, dann
    Cross-Hatch im Schatten, dann eine dritte Diagonale in den Tiefen. Ergebnis ist
    reines Schwarz/Weiß, kein Grauwert.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Alle Pixelmaße (Blur-Sigmen, Hatch-Pitch) sind auf `reference_width` kalibriert,
    # sonst würden Handy-Fotos mit 4K-Breite die Hatch-Linien zu Rauschen auflösen.
    reference_width = float(cfg.get("reference_width", 1600))
    scale = max(w / reference_width, 1e-3)

    pre_blur_cfg = cfg.get("pre_blur", {})
    if pre_blur_cfg.get("enabled", False):
        k = ops.ensure_odd(int(round(float(pre_blur_cfg.get("kernel", 5)) * scale)))
        gray = cv2.GaussianBlur(gray, (k, k), float(pre_blur_cfg.get("sigma", 1.6)) * scale)

    tensor_cfg = cfg.get("tensor", {})
    theta, _coherence = compute_structure_tensor_flow(
        gray,
        sigma_grad=float(tensor_cfg.get("sigma_grad", 3.0)) * scale,
        sigma_tensor=float(tensor_cfg.get("sigma_tensor", 14.0)) * scale,
        default_angle_deg=float(tensor_cfg.get("default_angle", 45.0)),
        coherence_floor=float(tensor_cfg.get("coherence_floor", 0.15)),
        coherence_gamma=float(tensor_cfg.get("coherence_gamma", 2.0)),
        energy_floor=float(tensor_cfg.get("energy_floor", 0.18)),
        flow_smooth_sigma=float(tensor_cfg.get("flow_smooth_sigma", 0.0)) * scale,
    )

    tone_blur_cfg = cfg.get("tone_blur", {})
    k = ops.ensure_odd(int(round(float(tone_blur_cfg.get("kernel", 5)) * scale)))
    tone_gray = cv2.GaussianBlur(gray, (k, k), float(tone_blur_cfg.get("sigma", 1.2)) * scale)
    tone = tone_gray.astype(np.float32) / 255.0

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    hatch_cfg = cfg.get("hatch", {})
    pitch = max(float(hatch_cfg.get("pitch", 6)) * scale, 1e-3)
    white_thresh = float(hatch_cfg.get("white_threshold", 0.86))
    cross_thresh = float(hatch_cfg.get("crosshatch_threshold", 0.45))
    black_thresh = float(hatch_cfg.get("black_threshold", 0.12))

    duty_main = tone_to_duty(tone, white_thresh, cross_thresh, float(hatch_cfg.get("max_duty", 0.55)))
    ink = hatch_layer(theta, xs, ys, 0.0, pitch, duty_main)

    duty_cross = tone_to_duty(tone, cross_thresh, black_thresh, float(hatch_cfg.get("max_duty_cross", 0.5)))
    ink |= hatch_layer(theta, xs, ys, np.pi / 2.0, pitch, duty_cross)

    duty_deep = tone_to_duty(tone, black_thresh, 0.0, float(hatch_cfg.get("max_duty_deep", 0.6)))
    ink |= hatch_layer(theta, xs, ys, np.pi / 4.0, pitch, duty_deep)

    canvas = np.where(ink, 0, 255).astype(np.uint8)

    outline_cfg = cfg.get("outline", {})
    if outline_cfg.get("enabled", False):
        # Kanten auf dem staerker geglaetteten tone_gray suchen, nicht auf gray:
        # sonst haelt Canny Sensor-/JPEG-Rauschen in eigentlich flachen Flaechen
        # (Wand, Himmel) faelschlich fuer Kanten und uebersaet das Bild mit Punkten.
        edges = cv2.Canny(
            tone_gray,
            float(outline_cfg.get("low", 60)),
            float(outline_cfg.get("high", 150))
        )
        thickness = max(1, int(round(float(outline_cfg.get("thickness", 1)) * scale)))
        if thickness > 1:
            edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))
        canvas[edges > 0] = 0

    out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    paper_cfg = cfg.get("paper_tone", {})
    if paper_cfg.get("enabled", False):
        out = ops.apply_paper_tone(out, float(paper_cfg.get("sepia_amount", 0.08)))

    grain_cfg = cfg.get("grain", {})
    if grain_cfg.get("enabled", False):
        out = ops.add_monochrome_grain(
            out,
            amount=float(grain_cfg.get("amount", 5)),
            opacity=float(grain_cfg.get("opacity", 0.05)),
            rng=rng,
        )

    vig_cfg = cfg.get("vignette", {})
    if vig_cfg.get("enabled", False):
        out = ops.add_vignette(out, float(vig_cfg.get("strength", 0.10)))

    out = ops.maybe_motion_blur(out, cfg, rng)

    return ops.clip_uint8(out)

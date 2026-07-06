import numpy as np
import random

from yatzar import ops


def _make_img(h=32, w=32):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_clip_uint8_clamps_values():
    img = np.array([[-10, 300]], dtype=np.float32)
    out = ops.clip_uint8(img)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 255]]


def test_ensure_odd_adjusts_even():
    assert ops.ensure_odd(4) == 5
    assert ops.ensure_odd(5) == 5
    assert ops.ensure_odd(0) == 1


def test_gaussian_blur_accepts_parameters_and_returns_valid_image():
    img = _make_img()
    out = ops.gaussian_blur(img, kernel=5, sigma=1.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_motion_blur_orientation():
    img = _make_img()
    out = ops.motion_blur(img, length=7, angle=0.0)
    assert out.shape == img.shape


def test_add_monochrome_grain_uses_passed_rng():
    img = _make_img()
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    out_a = ops.add_monochrome_grain(img, amount=16.0, opacity=0.10, rng=rng_a)
    out_b = ops.add_monochrome_grain(img, amount=16.0, opacity=0.10, rng=rng_b)
    assert out_a.tolist() == out_b.tolist()


def test_add_monochrome_grain_zero_opacity_identity():
    img = _make_img()
    rng = random.Random(0)
    out = ops.add_monochrome_grain(img, amount=16.0, opacity=0.0, rng=rng)
    assert out.tolist() == img.tolist()


def test_add_coarse_grain_uses_passed_rng():
    img = _make_img()
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    out_a = ops.add_coarse_grain(img, amount=12.0, opacity=0.15, scale=4, rng=rng_a)
    out_b = ops.add_coarse_grain(img, amount=12.0, opacity=0.15, scale=4, rng=rng_b)
    assert out_a.tolist() == out_b.tolist()


def test_add_coarse_grain_clamps_output():
    img = _make_img()
    out = ops.add_coarse_grain(img, amount=12.0, opacity=0.15, scale=4, rng=random.Random(0))
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_adjust_hsv_affects_saturation():
    img = _make_img()
    out = ops.adjust_hsv(img, sat_mult=2.0, val_mult=1.0)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_add_contrast_clamps_output():
    img = _make_img()
    out = ops.add_contrast(img, alpha=1.5, beta=-20)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_add_vignette_returns_bounded_uint8():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = ops.add_vignette(img, strength=0.5)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255
    assert np.mean(img) != np.mean(out)


def test_to_bw_grayscale():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = ops.to_bw(img)
    assert out.shape == (10, 10, 3)
    assert np.all(out[:, :, 0] == out[:, :, 1])
    assert np.all(out[:, :, 1] == out[:, :, 2])
    assert out.dtype == np.uint8


def test_apply_paper_tone_shifts_channels():
    img = _make_img()
    out = ops.apply_paper_tone(img, amount=0.2)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_unsharp_mask_returns_valid_image():
    img = _make_img()
    out = ops.unsharp_mask(img, amount=1.0, sigma=1.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

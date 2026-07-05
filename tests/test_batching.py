import pytest

from yatzar.batching import compute_batches, decide_workers


def test_empty_input():
    assert compute_batches(0, 25) == []


def test_exact_multiple_of_batch_size():
    bounds = compute_batches(100, 25)
    assert bounds == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_n_equals_batch_size_yields_single_batch():
    bounds = compute_batches(25, 25)
    assert bounds == [(0, 25)]


def test_n_equals_batch_size_plus_one_rebalances():
    bounds = compute_batches(26, 25)
    # 2 Batches statt 1x25 + 1x1
    sizes = [end - start for start, end in bounds]
    assert sizes == [13, 13]


def test_uneven_remainder_rebalances_instead_of_leftover_batch():
    bounds = compute_batches(110, 25)
    sizes = [end - start for start, end in bounds]
    assert sizes == [22, 22, 22, 22, 22]


def test_batches_cover_all_items_without_gaps_or_overlap():
    bounds = compute_batches(137, 10)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 137
    for (_, end), (next_start, _) in zip(bounds, bounds[1:]):
        assert end == next_start


def test_invalid_batch_size_raises():
    with pytest.raises(ValueError):
        compute_batches(10, 0)


def test_decide_workers_caps_at_num_batches():
    assert decide_workers(2, requested=8) == 2


def test_decide_workers_respects_explicit_request():
    assert decide_workers(10, requested=3) == 3


def test_decide_workers_zero_batches():
    assert decide_workers(0, requested=4) == 0

import numpy as np

import mrsnappy.preprocess as preprocess


def test_stage1_preprocessing_order_is_background_normalize_smooth(monkeypatch) -> None:
    calls: list[str] = []

    def background(volume, cfg):
        calls.append("background")
        return volume + 1

    def normalize(volume, cfg):
        calls.append("normalize")
        return volume * 2

    def smooth(volume, cfg):
        calls.append("smooth")
        return volume - 3

    monkeypatch.setattr(preprocess, "_apply_background_correction", background)
    monkeypatch.setattr(preprocess, "_apply_normalization", normalize)
    monkeypatch.setattr(preprocess, "_apply_preprocessing_filter", smooth)

    out = preprocess.apply_preprocessing(np.zeros((3, 3, 3), dtype=np.float32), {})

    assert calls == ["background", "normalize", "smooth"]
    assert np.all(out == -1)


def test_stage1_normalization_is_always_robust_z_score() -> None:
    volume = np.asarray([0, 1, 2, 3, 100], dtype=np.float32)

    out = preprocess._apply_normalization(volume, {"norm_enabled": False, "norm_method": "none"})

    expected_sigma = 1.0 / 0.6744897501960817
    expected = (volume - 2.0) / expected_sigma
    assert np.allclose(out, expected.astype(np.float32))


def test_stage1_normalization_handles_zero_mad_without_nan() -> None:
    volume = np.asarray([5, 5, 5, 5], dtype=np.float32)

    out = preprocess._apply_normalization(volume, {})

    assert np.array_equal(out, np.zeros_like(volume))


def test_slice_opening_2d_background_correction(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_opening(image, footprint):
        calls.append((image.shape, footprint.shape))
        return np.full_like(image, 2.0)

    monkeypatch.setattr(preprocess, "opening", fake_opening)
    volume = np.full((3, 4, 5), 5.0, dtype=np.float32)

    out = preprocess._apply_background_correction(
        volume,
        {
            "background_enabled": True,
            "background_method": "slice_opening_2d",
            "background_param": 2.0,
            "background_clip": False,
        },
    )

    assert calls == [((4, 5), (5, 5)), ((4, 5), (5, 5)), ((4, 5), (5, 5))]
    assert np.array_equal(out, np.full_like(volume, 3.0))


def test_stage1_background_correction_uses_2d_rolling_ball(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], float, dict]] = []

    def fake_rolling_ball(image, radius, **kwargs):
        calls.append((image.shape, radius, kwargs))
        return np.full_like(image, 1.0)

    monkeypatch.setattr(preprocess, "rolling_ball", fake_rolling_ball)
    volume = np.full((3, 4, 5), 2.0, dtype=np.float32)

    out = preprocess._apply_background_correction(
        volume,
        {
            "background_enabled": True,
            "background_method": "rolling_ball_2d",
            "background_param": 10.0,
        },
    )

    assert calls == [
        ((4, 5), 10.0, {"workers": -1}),
        ((4, 5), 10.0, {"workers": -1}),
        ((4, 5), 10.0, {"workers": -1}),
    ]
    assert np.array_equal(out, np.ones_like(volume))


def test_stage1_background_correction_uses_exact_3d_rolling_ball(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], float, dict]] = []

    def fake_rolling_ball(volume, radius, **kwargs):
        calls.append((volume.shape, radius, kwargs))
        return np.ones_like(volume)

    monkeypatch.setattr(preprocess, "rolling_ball", fake_rolling_ball)
    volume = np.full((3, 4, 5), 2.0, dtype=np.float32)

    out = preprocess._apply_background_correction(
        volume,
        {
            "background_enabled": True,
            "background_method": "rolling_ball_3d",
            "background_param": 10.0,
        },
    )

    assert calls == [((3, 4, 5), 10.0, {"workers": -1})]
    assert np.array_equal(out, np.ones_like(volume))


def test_stage1_background_correction_uses_scipy_3d_box(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_grey_opening(volume, size):
        calls.append((volume.shape, size))
        return np.ones_like(volume)

    monkeypatch.setattr(preprocess.ndi, "grey_opening", fake_grey_opening)
    volume = np.full((3, 4, 5), 2.0, dtype=np.float32)

    out = preprocess._apply_background_correction(
        volume,
        {
            "background_enabled": True,
            "background_method": "rolling_box_3d",
            "background_param": 3.0,
        },
    )

    assert calls == [((3, 4, 5), (7, 7, 7))]
    assert np.array_equal(out, np.ones_like(volume))


def test_stage1_background_rejects_unknown_methods() -> None:
    volume = np.ones((3, 3, 3), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        preprocess._apply_background_correction(
            volume,
            {"background_enabled": True, "background_method": "top_hat", "background_param": 10.0},
        )


def test_stage1_smoothing_rejects_non_gaussian_methods() -> None:
    volume = np.ones((3, 3, 3), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        preprocess._apply_preprocessing_filter(volume, {"preproc_enabled": True, "preproc_method": "median"})

# type: ignore
# pylint: disable=missing-function-docstring, missing-module-docstring
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

from richness import complexity

from . import preseqr_installed


def simulate(
    seed: int = 11,
    library_size: int = 50_000,
    shape: float = 0.5,
    scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a gamma-Poisson library of known composition.

    Returns a tuple (counts, freqs, rates), where counts and freqs are the
    frequency count histogram of the initial experiment and rates are the
    per molecule sequencing rates, which give the exact expected yield.
    """
    generator = np.random.default_rng(seed)
    rates = generator.gamma(shape, scale, library_size)
    observed = generator.poisson(rates)
    freqs, counts = np.unique(observed[observed > 0], return_counts=True)
    return counts.astype(float), freqs, rates


def expected_yield(rates: np.ndarray, t: float) -> float:
    """Get the exact expected number of molecules seen at depth `t`."""
    return float(np.sum(1 - np.exp(-t * rates)))


COUNTS, FREQS, RATES = simulate()
S_OBS = float(COUNTS.sum())
N_OBS = float((COUNTS * FREQS).sum())


def test_quotient_difference_round_trip() -> None:
    # The continued fraction must reproduce the Good-Toulmin coefficients
    # (-1)^i n_(i+1) of the series it was built from, which also pins down
    # the sign convention and the truncation of the histogram.
    histogram = np.zeros(int(FREQS.max()))
    histogram[FREQS - 1] = COUNTS
    empty = np.flatnonzero(histogram <= 0)
    limit = int(empty[0]) if empty.size else histogram.size
    usable = histogram[: min(limit, complexity.MAX_TERMS)]
    series = usable * (-1.0) ** np.arange(usable.size)
    fraction = complexity.ContinuedFraction.from_counts(COUNTS, FREQS)
    assert fraction.terms >= complexity.MIN_TERMS
    assert fraction.series_coefficients() == pytest.approx(
        series[: fraction.terms], rel=1e-9
    )


def test_geometric_series() -> None:
    # 1/(1+x) has the exact continued fraction 1/(1 + 1x).
    fraction = complexity.ContinuedFraction.from_series(
        np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    )
    assert fraction.coefficients[:3] == pytest.approx([1.0, 1.0, 0.0])
    assert fraction(np.array([0.0, 0.5, 3.0])) == pytest.approx(
        1 / (1 + np.array([0.0, 0.5, 3.0]))
    )


def test_yield_at_observed_depth() -> None:
    assert complexity.yield_rational(COUNTS, FREQS, 1.0)[0] == pytest.approx(
        S_OBS
    )


def test_rarefaction_matches_closed_form() -> None:
    t = np.array([0.1, 0.5, 0.9, 1.0])
    expected = [np.sum(COUNTS * (1 - (1 - ti) ** FREQS)) for ti in t]
    assert complexity.rarefaction_yield(COUNTS, FREQS, t) == pytest.approx(
        expected
    )


def test_yield_below_observed_depth_is_exact() -> None:
    # Below the observed depth the Good-Toulmin series converges, so the
    # prediction must equal the exact rarefaction of the sample.
    t = np.array([0.25, 0.75])
    assert complexity.yield_rational(COUNTS, FREQS, t) == pytest.approx(
        complexity.rarefaction_yield(COUNTS, FREQS, t)
    )


def test_rarefaction_rejects_extrapolation() -> None:
    with pytest.raises(ValueError):
        complexity.rarefaction_yield(COUNTS, FREQS, 2.0)


def test_approximation_is_even_order() -> None:
    # An even number of terms bounds the predicted yield, and is the odd
    # order approximation that converges from below.
    fraction, _ = complexity.fit_rational(COUNTS, FREQS, max_t=10.0)
    assert fraction.terms % 2 == 0


def test_extrapolation_accuracy() -> None:
    # The library composition is known, so the true yield is known exactly.
    for t in (2.0, 5.0, 10.0):
        predicted = complexity.yield_rational(COUNTS, FREQS, t)[0]
        assert predicted == pytest.approx(expected_yield(RATES, t), rel=0.05)


def test_predicted_yield_is_bounded() -> None:
    # An even order approximation must saturate rather than grow without
    # bound, since a library holds finitely many molecules.
    fraction, _ = complexity.fit_rational(COUNTS, FREQS, max_t=10.0)
    far = S_OBS + fraction.delta(np.array([1e3, 1e6, 1e9]))
    assert np.all(np.isfinite(far))
    assert far[-1] < 20 * S_OBS


def test_curve_is_sound() -> None:
    curve = complexity.complexity_curve(
        COUNTS, FREQS, max_extrapolation=10.0, steps=10, bootstraps=25
    )
    assert np.all(np.isfinite(curve.to_numpy()))
    assert np.all(np.diff(curve["Expected Distinct"]) > 0)
    assert np.all(curve["95% Lower"] <= curve["Bootstrap Median"])
    assert np.all(curve["Bootstrap Median"] <= curve["95% Upper"])
    assert curve.attrs["bootstraps"] == 25


def test_curve_passes_through_the_observed_experiment() -> None:
    curve = complexity.complexity_curve(
        COUNTS, FREQS, max_extrapolation=10.0, steps=7, bootstraps=0
    )
    assert N_OBS in curve.index
    assert curve.loc[N_OBS, "Expected Distinct"] == pytest.approx(S_OBS)


def test_curve_is_reproducible() -> None:
    kwargs = {"max_extrapolation": 8.0, "steps": 5, "bootstraps": 15}
    first = complexity.complexity_curve(COUNTS, FREQS, seed=3, **kwargs)
    second = complexity.complexity_curve(COUNTS, FREQS, seed=3, **kwargs)
    other = complexity.complexity_curve(COUNTS, FREQS, seed=4, **kwargs)
    assert first.equals(second)
    assert not first.equals(other)


def test_too_few_frequency_classes() -> None:
    # A histogram with a gap below the fourth frequency cannot be expanded.
    with pytest.raises(complexity.ComplexityError):
        complexity.yield_rational(
            np.array([3.0, 2.0, 1.0]), np.array([1, 2, 5]), 2.0
        )


def test_saturated_library_reports_its_limit() -> None:
    # A homogeneous library supports only a modest extrapolation, which must
    # be reported rather than silently exceeded.
    counts, freqs, _ = simulate(
        seed=3, library_size=30_000, shape=1e6, scale=1e-6
    )
    fraction, sound_t = complexity.fit_rational(counts, freqs, max_t=10.0)
    assert fraction is not None
    assert 1.0 < sound_t < 10.0
    with pytest.raises(complexity.ComplexityError, match="supports"):
        complexity.yield_rational(counts, freqs, 10.0)
    with pytest.warns(RuntimeWarning, match="truncated"):
        curve = complexity.complexity_curve(
            counts, freqs, max_extrapolation=10.0, steps=20, bootstraps=10
        )
    n_obs = float((counts * freqs).sum())
    assert curve.index.max() <= sound_t * n_obs
    assert curve.attrs["max_sound_t"] == pytest.approx(sound_t)


def test_metrics_statistics() -> None:
    frequencies = pd.Series(
        np.repeat(FREQS, COUNTS.astype(int)),
        index=[f"m{i}" for i in range(int(S_OBS))],
    )
    statistics, curve = complexity.complexity_metrics(
        frequencies, max_extrapolation=5.0, steps=5, bootstraps=10
    )
    assert statistics.loc["# distinct molecules", "Value"] == S_OBS
    assert statistics.loc["# sequenced reads", "Value"] == N_OBS
    assert statistics.loc["# terms in approximation", "Value"] == (
        curve.attrs["terms"]
    )


@pytest.mark.skipif(not preseqr_installed, reason="preseqR is unavailable")
def test_preseqr_agreement() -> None:
    # preseqR is the reference implementation. It selects the order of the
    # approximation by a different heuristic, so the curves agree closely
    # but not exactly.
    preseqr = rpackages.importr("preseqR")
    histogram = ro.r["matrix"](
        ro.FloatVector(np.concatenate([FREQS, COUNTS])), ncol=2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = preseqr.ds_rSAC(histogram, r=1, mt=100)
    for t in (1.0, 2.0, 5.0, 10.0):
        expected = float(reference(t)[0])
        predicted = complexity.yield_rational(COUNTS, FREQS, t)[0]
        assert predicted == pytest.approx(expected, rel=0.05)

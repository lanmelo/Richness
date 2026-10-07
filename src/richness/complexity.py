r"""Prediction of sequencing library complexity by rational approximation.

Where the estimators in :mod:`richness` answer "how many species exist?",
this module answers "how many *distinct molecules* will be observed if the
library is sequenced :math:`t` times more deeply?". The expected yield of a
library sequenced to a depth of :math:`t` times the initial experiment is
given by the Good-Toulmin power series

.. math::
    Y(t) = S_{obs} + \sum_{j=1}^{\infty} (-1)^{j+1} (t-1)^j n_j

where :math:`n_j` is the number of molecules observed exactly `j` times in
the initial experiment. Truncating that series is only guaranteed to
converge for :math:`t \le 2`, which is far too little for sequencing
applications. Following Daley and Smith (2013), the series is instead
approximated by a rational function, expressed as a truncated continued
fraction whose coefficients are obtained with the quotient-difference
algorithm. Daley and Smith report accurate extrapolations of 30 times the
initial experiment or more.

The complexity (yield) curve of a frequency Series is calculated with

    complexity_metrics(frequencies)

The approximation is built from an even number of power series terms. The
marginal yield is then a ratio of polynomials whose denominator is of higher
degree than its numerator, so that the predicted yield is bounded, as a
finite library requires; in the notation of Daley and Smith this is an odd
order approximation, which converges to the true yield from below, and so
is conservative, when the expected frequency counts are known exactly.
Because they are instead estimated from counts, high order approximations
amplify sampling noise, so the order is increased only while successive
approximations agree. Approximations also have poles, which limit the depth
to which a given library can be extrapolated at all; where the requested
depth cannot be reached, the curve is reported as far as it is sound.

Note that this is a prediction at a *stated sequencing depth*, not an
asymptotic richness estimate: extrapolating to :math:`t = \infty` is
equivalent to estimating the library size, which is unidentifiable and has
no unbiased estimator without further assumptions. Such estimates are
therefore reported separately from the asymptotic estimators in
:mod:`richness`.

Unlike the rest of this package, this module uses double-precision NumPy
rather than JAX. The Good-Toulmin series is alternating with large terms,
and both the quotient-difference recursion and the continued fraction
evaluation lose catastrophically many digits in the single precision that
JAX uses by default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import pandas as pd
import scipy.stats
from jax import Array
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .richness import get_frequency_counts

Floats: TypeAlias = NDArray[np.float64]

#: Frequency classes required before the first unobserved frequency.
MIN_COUNTS = 4
#: Smallest number of power series terms used in an approximation.
MIN_TERMS = 4
#: Relative disagreement between successive orders that stops refinement.
TOLERANCE = 0.05
#: Largest number of power series terms used in an approximation.
MAX_TERMS = 100


class ComplexityError(RuntimeError):
    """Raised when a complexity curve cannot be extrapolated."""


def _dense_histogram(counts: Array, freqs: Array) -> Floats:
    """Convert sparse frequency counts into a dense histogram.

    Args:
        counts: A float Array of frequency counts.
        freqs: An int Array of their corresponding frequencies.

    Returns:
        A float Array where element `j` holds the number of species observed
        exactly `j + 1` times, including empty frequency classes.
    """
    count_array = np.asarray(counts, dtype=np.float64).ravel()
    freq_array = np.asarray(freqs, dtype=np.int64).ravel()
    if count_array.size != freq_array.size:
        raise ValueError("counts and freqs must have the same size")
    if freq_array.size == 0:
        raise ComplexityError("no observed species")
    if freq_array.min() < 1:
        raise ValueError("freqs must be positive")
    histogram = np.zeros(int(freq_array.max()), dtype=np.float64)
    histogram[freq_array - 1] = count_array
    return histogram


def _usable_histogram(histogram: Floats, max_terms: int = MAX_TERMS) -> Floats:
    """Truncate a histogram before its first empty frequency class.

    The quotient-difference algorithm divides by consecutive power series
    coefficients, so an empty frequency class terminates the usable series.

    Args:
        histogram: A dense float Array of frequency counts.
        max_terms: The largest number of frequency classes to retain.

    Returns:
        The leading run of nonempty frequency classes.
    """
    empty = np.flatnonzero(histogram <= 0)
    limit = int(empty[0]) if empty.size else int(histogram.size)
    limit = min(limit, max_terms)
    if limit < MIN_COUNTS:
        raise ComplexityError(
            f"only {limit} nonempty frequency class(es) before the first"
            f" empty class, but at least {MIN_COUNTS} are required;"
            " the library is too shallowly sequenced to extrapolate"
        )
    return histogram[:limit]


def _power_series(histogram: Floats) -> Floats:
    r"""Get Good-Toulmin coefficients :math:`q_i = (-1)^i n_{i+1}`.

    The marginal yield of the complete experiment is
    :math:`\Delta(t) = x \sum_i q_i x^i` with :math:`x = t - 1`, so the
    alternating signs of the Good-Toulmin series are absorbed here.
    """
    signs = np.where(np.arange(histogram.size) % 2 == 0, 1.0, -1.0)
    return cast(Floats, histogram * signs)


def _quotient_difference(series: Floats) -> Floats:
    """Convert power series coefficients to continued fraction coefficients.

    Implements Rutishauser's quotient-difference algorithm. The `q` and `e`
    tables are built by the recursions

        q[1][i] = c[i+1] / c[i]
        e[k][i] = q[k][i+1] - q[k][i] + e[k-1][i+1]
        q[k+1][i] = q[k][i+1] * e[k][i+1] / e[k][i]

    whose valid index ranges shrink by one entry per level, and the leading
    column of the tables gives the coefficients of the continued fraction.
    Coefficients are computed for as many levels as the series supports; a
    vanishing divisor ends the expansion, so the result may be shorter than
    the series.

    Args:
        series: A float Array of power series coefficients.

    Returns:
        A float Array of continued fraction coefficients.
    """
    depth = int(series.size)
    levels = depth // 2 + 1
    q_table = np.zeros((levels + 1, depth + 1), dtype=np.float64)
    e_table = np.zeros((levels + 1, depth + 1), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        q_table[1, : depth - 1] = series[1:] / series[:-1]
        e_table[1, : depth - 2] = np.diff(q_table[1, : depth - 1])
        for level in range(2, levels + 1):
            width = max(depth - 2 * level + 1, 0)
            q_table[level, :width] = (
                q_table[level - 1, 1 : width + 1]
                * e_table[level - 1, 1 : width + 1]
                / e_table[level - 1, :width]
            )
            width = max(depth - 2 * level, 0)
            e_table[level, :width] = (
                q_table[level, 1 : width + 1]
                - q_table[level, :width]
                + e_table[level - 1, 1 : width + 1]
            )
    coefficients = np.empty(depth, dtype=np.float64)
    coefficients[0] = series[0]
    for index in range(1, depth):
        coefficients[index] = (
            -q_table[(index + 1) // 2, 0]
            if index % 2
            else -e_table[index // 2, 0]
        )
    finite = np.isfinite(coefficients)
    if finite.all():
        return coefficients
    return coefficients[: int(np.argmin(finite))]


def _series_inverse(series: Floats, terms: int) -> Floats:
    """Invert a power series with a nonzero constant term."""
    inverse = np.zeros(terms, dtype=np.float64)
    inverse[0] = 1 / series[0]
    for order in range(1, terms):
        inverse[order] = (
            -series[1 : order + 1] @ inverse[order - 1 :: -1] / series[0]
        )
    return inverse


@dataclass(frozen=True)
class ContinuedFraction:
    r"""A truncated continued fraction approximating a power series.

    Represents the rational approximation

    .. math::
        Q(x) = \frac{c_0}{1 + \cfrac{c_1 x}{1 + \cfrac{c_2 x}{1 + \dots}}}

    of the Good-Toulmin power series in :math:`x = t - 1`, so that the
    marginal yield of the complete experiment is :math:`x Q(x)`.
    """

    coefficients: Floats

    @property
    def terms(self) -> int:
        """The number of power series coefficients reproduced exactly."""
        return int(self.coefficients.size)

    @classmethod
    def from_series(cls, series: Floats) -> ContinuedFraction:
        """Build the continued fraction of a power series."""
        return cls(_quotient_difference(np.asarray(series, dtype=np.float64)))

    @classmethod
    def from_counts(
        cls, counts: Array, freqs: Array, max_terms: int = MAX_TERMS
    ) -> ContinuedFraction:
        """Build the continued fraction of a frequency count histogram."""
        histogram = _usable_histogram(
            _dense_histogram(counts, freqs), max_terms
        )
        return cls.from_series(_power_series(histogram))

    def truncate(self, terms: int) -> ContinuedFraction:
        """Get the approximation using only the first `terms` coefficients.

        Lower order approximations are obtained by truncating a single high
        order continued fraction, which avoids recomputing coefficients.
        """
        if terms < 1:
            raise ValueError("terms must be positive")
        return ContinuedFraction(self.coefficients[:terms])

    def __call__(self, x: Array | Floats | float) -> Floats:
        """Evaluate the continued fraction.

        Uses Euler's forward recursion for the numerator and denominator
        convergents, renormalizing after every level so that intermediate
        magnitudes remain manageable.

        Args:
            x: The point(s) at which to evaluate, equal to `t - 1`.

        Returns:
            A float Array of values, which is not finite at a pole.
        """
        values = np.atleast_1d(np.asarray(x, dtype=np.float64))
        numerator = np.zeros_like(values)
        numerator_prev = np.ones_like(values)
        denominator = np.ones_like(values)
        denominator_prev = np.zeros_like(values)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            for index, coefficient in enumerate(self.coefficients):
                partial = (
                    np.full_like(values, coefficient)
                    if index == 0
                    else coefficient * values
                )
                numerator, numerator_prev = (
                    numerator + partial * numerator_prev,
                    numerator,
                )
                denominator, denominator_prev = (
                    denominator + partial * denominator_prev,
                    denominator,
                )
                scale = np.maximum(np.abs(numerator), np.abs(denominator))
                scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
                numerator, numerator_prev = numerator / scale, (
                    numerator_prev / scale
                )
                denominator, denominator_prev = denominator / scale, (
                    denominator_prev / scale
                )
            return numerator / denominator

    def series_coefficients(self, terms: int | None = None) -> Floats:
        """Expand the continued fraction back into a power series.

        The first :attr:`terms` coefficients reproduce those of the series
        the approximation was built from.
        """
        terms = self.terms if terms is None else terms
        expansion = np.zeros(terms, dtype=np.float64)
        expansion[0] = 1.0
        for coefficient in self.coefficients[:0:-1]:
            inverse = _series_inverse(expansion, terms)
            expansion = np.zeros(terms, dtype=np.float64)
            expansion[0] = 1.0
            expansion[1:] = coefficient * inverse[:-1]
        return cast(
            Floats, self.coefficients[0] * _series_inverse(expansion, terms)
        )

    def delta(self, t: Array | Floats | float) -> Floats:
        r"""Get the marginal yield :math:`\Delta(t) = (t-1) Q(t-1)`.

        This is the expected number of molecules observed in an experiment
        `t` times the size of the initial experiment that were unobserved in
        the initial experiment.
        """
        x = np.atleast_1d(np.asarray(t, dtype=np.float64)) - 1
        return x * self(x)


def _rarefaction(histogram: Floats, t: Floats) -> Floats:
    r"""Get the exact expected yield at or below the observed depth.

    A molecule observed `j` times is missed by a random sample of `t` times
    the reads with probability :math:`(1-t)^j`, so

    .. math::
        Y(t) = \sum_j n_j \left( 1 - (1-t)^j \right)

    which is the Good-Toulmin series where it converges.

    Args:
        histogram: A dense float Array of frequency counts.
        t: A float Array of depths, in multiples of the initial experiment.

    Returns:
        A float Array of expected distinct molecules.
    """
    freqs = np.arange(1, histogram.size + 1, dtype=np.float64)
    missed = np.power.outer(1 - t, freqs)
    return cast(Floats, (histogram * (1 - missed)).sum(axis=-1))


def _yields(
    histogram: Floats, fraction: ContinuedFraction | None, t: Floats
) -> Floats:
    """Get expected yields, interpolating at or below the observed depth."""
    estimates = np.empty_like(t)
    interpolated = t <= 1
    estimates[interpolated] = _rarefaction(histogram, t[interpolated])
    if fraction is not None:
        estimates[~interpolated] = histogram.sum() + fraction.delta(
            t[~interpolated]
        )
    else:
        estimates[~interpolated] = np.nan
    return estimates


def _soundness_grid(max_t: float, grid_size: int = 200) -> Floats:
    """Get depths at which to check an approximation for defects.

    Covers the requested range densely, and the whole ray beyond it through
    the substitution :math:`u = (t-1)/t`, which maps every depth from 1 to
    infinity onto the unit interval.

    Args:
        max_t: The largest depth of interest.
        grid_size: The number of depths in each part of the grid.

    Returns:
        A sorted float Array of depths beginning at 1.
    """
    u = np.linspace(0.0, 1 - 1e-9, grid_size, dtype=np.float64)
    dense = np.linspace(1.0, max_t, grid_size, dtype=np.float64)
    return np.union1d(dense, 1 / (1 - u))


def _sound_limit(
    S_obs: float, fraction: ContinuedFraction, grid: Floats
) -> float:
    """Get the largest depth up to which an approximation has no defect.

    A rational approximation of a given order has poles, and a pole shows up
    as a yield that is not finite, or that spikes and then falls back. The
    curve is only trustworthy below the first such depth.

    Args:
        S_obs: The number of distinct molecules observed.
        fraction: The approximation to check.
        grid: A float Array of depths beginning at 1.

    Returns:
        The largest depth in `grid` below the first defect, or 1 if the
        approximation is defective throughout.
    """
    estimates = S_obs + fraction.delta(grid)
    sound = np.isfinite(estimates) & (estimates >= S_obs)
    sound[1:] &= np.diff(estimates) > 0
    defects = np.flatnonzero(~sound)
    if not defects.size:
        return float(grid[-1])
    return 1.0 if defects[0] == 0 else float(grid[defects[0] - 1])


def _fit_histogram(
    histogram: Floats,
    max_t: float,
    max_terms: int = MAX_TERMS,
    min_terms: int = MIN_TERMS,
    tolerance: float = TOLERANCE,
    grid_size: int = 200,
) -> tuple[ContinuedFraction, float]:
    """Select an approximation of a histogram and the depth it supports.

    Candidates of increasing even order are checked for defects. The lowest
    order candidate that reaches the requested depth is refined by taking
    successively higher orders while they agree with it, which guards
    against the sampling noise that high order approximations amplify. If no
    candidate reaches the requested depth, the one reaching furthest is
    returned instead.

    Args:
        histogram: A dense float Array of frequency counts.
        max_t: The depth the approximation should reach, in multiples of the
            initial experiment.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.
        tolerance: Relative disagreement between successive orders that
            stops further refinement.
        grid_size: The number of depths at which to check for defects.

    Returns:
        A tuple (fraction, max_sound_t), where fraction is the selected
        ContinuedFraction and max_sound_t is the largest depth it supports,
        which is at most `max_t`.
    """
    if max_t <= 1:
        raise ValueError("max_t must be greater than 1 to extrapolate")
    usable = _usable_histogram(histogram, max_terms)
    series = _power_series(usable)
    S_obs = float(histogram.sum())
    grid = _soundness_grid(max_t, grid_size)
    full = ContinuedFraction.from_series(series)

    candidates: list[tuple[int, ContinuedFraction, float]] = []
    lowest = max(min_terms, 2) + max(min_terms, 2) % 2
    for terms in range(lowest, int(usable.size) + 1, 2):
        # Truncating one high order fraction is cheaper than recomputing
        # coefficients, but a vanishing divisor may have ended that
        # expansion early, in which case fewer terms may still expand.
        candidate = (
            full.truncate(terms)
            if full.terms >= terms
            else ContinuedFraction.from_series(series[:terms])
        )
        if candidate.terms == terms:
            candidates.append(
                (terms, candidate, _sound_limit(S_obs, candidate, grid))
            )
    if not candidates:
        raise ComplexityError(
            "no rational approximation could be built from"
            f" {usable.size} frequency class(es)"
        )

    reaching = [
        fraction for _terms, fraction, limit in candidates if limit >= max_t
    ]
    if not reaching:
        _terms, fraction, limit = max(
            candidates, key=lambda entry: (entry[2], -entry[0])
        )
        if limit <= 1:
            raise ComplexityError(
                "every rational approximation of this library is defective"
                " immediately above the observed depth; it may be saturated"
                " or too shallowly sequenced to extrapolate"
            )
        return fraction, limit

    comparison = np.linspace(1.0, max_t, 50, dtype=np.float64)
    selected = reaching[0]
    for candidate in reaching[1:]:
        previous = S_obs + selected.delta(comparison)
        refined = S_obs + candidate.delta(comparison)
        if np.max(np.abs(refined / previous - 1)) > tolerance:
            break
        selected = candidate
    return selected, max_t


def fit_rational(
    counts: Array,
    freqs: Array,
    max_t: float = 10.0,
    max_terms: int = MAX_TERMS,
    min_terms: int = MIN_TERMS,
) -> tuple[ContinuedFraction, float]:
    """Select an approximation of a library and the depth it supports.

    Args:
        counts: A float Array of frequency counts.
        freqs: An int Array of their corresponding frequencies.
        max_t: The depth the approximation should reach, in multiples of the
            initial experiment.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.

    Returns:
        A tuple (fraction, max_sound_t), where fraction is the selected
        ContinuedFraction and max_sound_t is the largest depth it supports,
        which is at most `max_t`.
    """
    return _fit_histogram(
        _dense_histogram(counts, freqs), max_t, max_terms, min_terms
    )


def rarefaction_yield(
    counts: Array, freqs: Array, t: Array | Floats | float
) -> Floats:
    """Get the exact expected yield at or below the observed depth.

    Args:
        counts: A float Array of frequency counts.
        freqs: An int Array of their corresponding frequencies.
        t: The depth(s), in multiples of the initial experiment.

    Returns:
        A float Array of expected distinct molecules.
    """
    depths = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if np.any(depths > 1):
        raise ValueError("t must not exceed 1; use yield_rational instead")
    return _rarefaction(_dense_histogram(counts, freqs), depths)


def yield_rational(
    counts: Array,
    freqs: Array,
    t: Array | Floats | float,
    max_terms: int = MAX_TERMS,
    min_terms: int = MIN_TERMS,
) -> Floats:
    """Predict the yield of a library sequenced to a greater depth.

    Args:
        counts: A float Array of frequency counts.
        freqs: An int Array of their corresponding frequencies.
        t: The depth(s), in multiples of the initial experiment.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.

    Returns:
        A float Array of expected distinct molecules.

    Raises:
        ComplexityError: If the library cannot be extrapolated as far as the
            deepest requested depth. Use complexity_curve to get the curve
            over the range that is supported.
    """
    depths = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if np.any(depths <= 0):
        raise ValueError("t must be positive")
    histogram = _dense_histogram(counts, freqs)
    fraction = None
    if np.any(depths > 1):
        max_t = float(depths.max())
        fraction, sound_t = _fit_histogram(
            histogram, max_t, max_terms, min_terms
        )
        if sound_t < max_t:
            raise ComplexityError(
                f"this library supports extrapolation to {sound_t:.4g} times"
                f" the initial experiment, but {max_t:.4g} was requested"
            )
    return _yields(histogram, fraction, depths)


def _resample_histogram(
    histogram: Floats, generator: np.random.Generator
) -> Floats:
    """Resample the species of a histogram with replacement."""
    species = int(round(float(histogram.sum())))
    return generator.multinomial(species, histogram / histogram.sum()).astype(
        np.float64
    )


def _bootstrap_yields(
    histogram: Floats,
    reads: Floats,
    bootstraps: int,
    max_terms: int,
    min_terms: int,
    seed: int | None,
) -> Floats:
    """Get bootstrap replicates of the complexity curve.

    Species are resampled with replacement from the observed frequency
    distribution, and each replicate is refit and evaluated at the same
    absolute numbers of reads. Replicates that cannot be extrapolated are
    discarded and retried.

    Args:
        histogram: A dense float Array of frequency counts.
        reads: A float Array of total reads at which to evaluate.
        bootstraps: The number of replicates to collect.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.
        seed: Seed for the random generator.

    Returns:
        A float Array of shape (replicates, reads) of expected yields.
    """
    generator = np.random.default_rng(seed)
    replicates: list[Floats] = []
    for _ in range(10 * bootstraps):
        if len(replicates) == bootstraps:
            break
        resampled = _resample_histogram(histogram, generator)
        n_obs = float((resampled * np.arange(1, resampled.size + 1)).sum())
        if n_obs <= 0:
            continue
        depths = reads / n_obs
        fraction = None
        if np.any(depths > 1):
            try:
                fraction, sound_t = _fit_histogram(
                    resampled, float(depths.max()), max_terms, min_terms
                )
            except ComplexityError:
                continue
            if sound_t < float(depths.max()):
                continue
        replicates.append(_yields(resampled, fraction, depths))
    if not replicates:
        raise ComplexityError(
            f"none of {10 * bootstraps} bootstrap replicates could be"
            " extrapolated; the complexity curve has no confidence interval"
        )
    if len(replicates) < bootstraps:
        warnings.warn(
            f"only {len(replicates)} of {bootstraps} bootstrap replicates"
            " could be extrapolated",
            RuntimeWarning,
        )
    return np.stack(replicates)


def complexity_curve(
    counts: Array,
    freqs: Array,
    max_extrapolation: float = 10.0,
    steps: int = 20,
    bootstraps: int = 100,
    confidence: float = 0.95,
    max_terms: int = MAX_TERMS,
    min_terms: int = MIN_TERMS,
    seed: int | None = 0,
) -> DataFrame:
    """Predict the complexity curve of a sequencing library.

    The curve gives the expected number of distinct molecules observed as a
    function of the number of reads sequenced. It is exact at or below the
    observed depth and extrapolated by rational approximation above it.
    Confidence intervals are obtained by resampling species with
    replacement and refitting each replicate, so the reported bootstrap
    median differs slightly from the estimate of the observed data.

    Args:
        counts: A float Array of frequency counts.
        freqs: An int Array of their corresponding frequencies.
        max_extrapolation: The largest depth to predict, in multiples of the
            initial experiment.
        steps: The number of depths at which to predict.
        bootstraps: The number of bootstrap replicates, or 0 for none.
        confidence: The confidence level of the confidence interval.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.
        seed: Seed for the random generator, for reproducible intervals.

    Returns:
        A DataFrame of expected yields indexed by total reads. The number of
        terms selected and bootstrap replicates used are recorded in its
        `attrs`.
    """
    if max_extrapolation <= 0:
        raise ValueError("max_extrapolation must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    histogram = _dense_histogram(counts, freqs)
    n_obs = float((histogram * np.arange(1, histogram.size + 1)).sum())
    reads = np.union1d(
        np.linspace(
            n_obs * max_extrapolation / steps, n_obs * max_extrapolation, steps
        ),
        [n_obs],
    )
    depths = reads / n_obs
    fraction = None
    sound_t = float(depths.max())
    if max_extrapolation > 1:
        fraction, sound_t = _fit_histogram(
            histogram, float(depths.max()), max_terms, min_terms
        )
        if sound_t < float(depths.max()):
            warnings.warn(
                f"this library supports extrapolation to only {sound_t:.4g}"
                f" times the initial experiment, not {depths.max():.4g};"
                " the complexity curve is truncated there",
                RuntimeWarning,
            )
            reads, depths = reads[depths <= sound_t], depths[depths <= sound_t]
    curve = DataFrame(
        {"Expected Distinct": _yields(histogram, fraction, depths)},
        index=pd.Index(reads, name="total_reads"),
    )

    replicates = 0
    median = np.full(reads.size, np.nan)
    error = np.full(reads.size, np.nan)
    lower, upper = median.copy(), median.copy()
    if bootstraps > 0:
        estimates = _bootstrap_yields(
            histogram, reads, bootstraps, max_terms, min_terms, seed
        )
        replicates = int(estimates.shape[0])
        median = np.median(estimates, axis=0)
        error = np.std(estimates, axis=0, ddof=1 if replicates > 1 else 0)
        sigmas = cast(float, scipy.stats.norm.interval(confidence)[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            multiplier = np.exp(
                sigmas * np.sqrt(np.log1p(np.square(error / median)))
            )
        lower, upper = median / multiplier, median * multiplier
    curve["Bootstrap Median"] = median
    curve["S.E."] = error
    curve[f"{confidence*100:.0f}% Lower"] = lower
    curve[f"{confidence*100:.0f}% Upper"] = upper
    curve.attrs["terms"] = fraction.terms if fraction is not None else 0
    curve.attrs["bootstraps"] = replicates
    curve.attrs["max_sound_t"] = sound_t
    return curve


def complexity_metrics(
    frequencies: Series[int],
    max_extrapolation: float = 10.0,
    steps: int = 20,
    bootstraps: int = 100,
    confidence: float = 0.95,
    max_terms: int = MAX_TERMS,
    min_terms: int = MIN_TERMS,
    seed: int | None = 0,
) -> tuple[DataFrame, DataFrame]:
    """Predicts the complexity curve of a library from its frequencies.

    Args:
        frequencies: A Series of molecule frequencies.
        max_extrapolation: The largest depth to predict, in multiples of the
            initial experiment.
        steps: The number of depths at which to predict.
        bootstraps: The number of bootstrap replicates, or 0 for none.
        confidence: The confidence level of the confidence interval.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.
        seed: Seed for the random generator, for reproducible intervals.

    Returns:
        A tuple (statistics, curve), where statistics is a DataFrame of
        library statistics and curve is a DataFrame of expected yields
        indexed by total reads.
    """
    counts, freqs = get_frequency_counts(frequencies)
    curve = complexity_curve(
        counts,
        freqs,
        max_extrapolation=max_extrapolation,
        steps=steps,
        bootstraps=bootstraps,
        confidence=confidence,
        max_terms=max_terms,
        min_terms=min_terms,
        seed=seed,
    )
    histogram = _dense_histogram(counts, freqs)
    n_obs = float((histogram * np.arange(1, histogram.size + 1)).sum())
    S_obs = float(histogram.sum())
    empty = np.flatnonzero(histogram <= 0)
    usable = int(empty[0]) if empty.size else int(histogram.size)
    statistics = pd.DataFrame.from_dict(
        {
            "# sequenced reads": ["n_obs", n_obs],
            "# distinct molecules": ["S_obs", S_obs],
            "# molecules seen once": ["n_1", float(histogram[0])],
            "duplication rate": ["1-S_obs/n_obs", 1 - (S_obs / n_obs)],
            "# usable frequency classes": ["J", usable],
            "# terms in approximation": ["terms", curve.attrs["terms"]],
            "# bootstrap replicates": ["B", curve.attrs["bootstraps"]],
            "extrapolation limit": [
                "t",
                f"{curve.attrs['max_sound_t']:.4g}x"
                f" ({n_obs * curve.attrs['max_sound_t']:.0f} reads)",
            ],
        },
        orient="index",
        columns=["Variable", "Value"],
        dtype=object,
    )
    return statistics, curve


def complexity_string(
    frequencies: Series[int],
    max_extrapolation: float = 10.0,
    steps: int = 20,
    bootstraps: int = 100,
    confidence: float = 0.95,
    max_terms: int = MAX_TERMS,
    min_terms: int = MIN_TERMS,
    seed: int | None = 0,
) -> str:
    """Predicts the complexity curve of a library from its frequencies.

    Args:
        frequencies: A Series of molecule frequencies.
        max_extrapolation: The largest depth to predict, in multiples of the
            initial experiment.
        steps: The number of depths at which to predict.
        bootstraps: The number of bootstrap replicates, or 0 for none.
        confidence: The confidence level of the confidence interval.
        max_terms: The largest number of power series terms to use.
        min_terms: The smallest number of power series terms to use.
        seed: Seed for the random generator, for reproducible intervals.

    Returns:
        A string containing library statistics and the complexity curve.
    """
    statistics, curve = complexity_metrics(
        frequencies,
        max_extrapolation=max_extrapolation,
        steps=steps,
        bootstraps=bootstraps,
        confidence=confidence,
        max_terms=max_terms,
        min_terms=min_terms,
        seed=seed,
    )
    return cast(
        str,
        statistics.to_string(float_format=lambda x: f"{x:.3f}")
        + "\n\n"
        + curve.to_string(float_format=lambda x: f"{x:.1f}"),
    )

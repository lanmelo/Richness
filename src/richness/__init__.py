"""Nonparametric estimation of species richness from abundance/incidence data.

This module provides functions for nonparametric estimation of species richness
from both abundance and incidence data. Given abundance frequencies in a Pandas
Series, richness metrics can be calculated with

    abundance_richness_metrics(abundance_frequencies)

Alternatively, given incidence frequencies in a Series

    incidence_richness_metrics([incidence_frequencies])

or, with raw incidence data, with each sampling unit represented as a Series

    incidence_richness_metrics([incidence_1, incidence_2, ...])

Most other functions in this module work on a counts of frequencies.
Frequency data can be summarized into frequency counts, which encode the
number of species of a given frequency. For example, `f_0` is the count of
undetected species and `f_1` is the count of singleton species (species that
were only detected once). Frequency counts are stored as a float Array to allow
for gradient calculation.
"""

from __future__ import annotations

import gzip
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeAlias, cast

import jax
import jax.numpy as np
import jax.random
import pandas as pd
import scipy.special
import scipy.stats
from jax import Array
from pandas import DataFrame, Series

__version__ = "1.0.0"

float_type: TypeAlias = float | Array


def read_frequencies(path: str) -> "Series[int]":
    """Loads tab-delimited frequency data into a Pandas Series.

    The input data is assumed to have two columns separated by a tab character.
    Each row contains the species name and its corresponding frequency.
    Any initial lines lacking a tab character are skipped.

    Args:
        path: Path to the frequency TSV.

    Returns:
        A Series of frequencies, with species names stored in the index.
    """
    open_fn = gzip.open if os.path.splitext(path)[-1] == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8") as file_handle:  # type: ignore[operator]
        skiprows = 0
        while "\t" not in file_handle.readline():
            skiprows += 1
    return pd.read_csv(
        path,
        skiprows=skiprows,
        header=None,
        index_col=0,
        sep="\t",
        lineterminator="\n",
        encoding="utf-8",
        keep_default_na=False,
    ).iloc[:, 0]


def split_frequencies(
    abundance: "Series[int]", n: int = 2
) -> list["Series[int]"]:
    """Randomly sample a frequency Series into `n` Series."""
    out: list["Series[int]"] = []
    for _ in range(n - 1):
        new = cast(
            "Series[int]",
            pd.Series(
                data=scipy.stats.binom.rvs(abundance, 0.5),
                index=abundance.index,
            ),
        )
        abundance = abundance - new
        abundance = abundance[abundance > 0]
        new = new[new > 0]
        out.append(new)
    return [abundance] + out


def get_frequency_counts(frequencies: "Series[int]") -> tuple[Array, Array]:
    """Convert a frequency Series into a countsogram of frequency counts.

    Args:
        frequencies: A Series of species frequencies.

    Returns:
        A tuple (counts, freqs), where counts is a 1d float Array of frequency
        counts and freqs is a 1d int Array of their corresponding frequencies.
    """
    freqs, counts = np.unique(frequencies.to_numpy(), return_counts=True)
    return counts.astype(float), freqs


@jax.jit
def _frequency_count(
    counts: Array, freqs: Array, frequency: int
) -> float_type:  # typing issue: https://github.com/google/jax/issues/10311
    """Get count of species with a given frequency."""
    idx = np.searchsorted(freqs, frequency)
    concat = np.pad(counts, (0, 1))
    return concat[idx]


def raw_to_frequencies(
    incidence_series: Iterable[Mapping[Any, Any]] | "Sequence[Series[int]]"
) -> "Series[int]":
    """Converts raw incidence data to frequency data.

    Args:
        incidence_series: An Iterable where each element is a Mapping of
            species names to their presence in the respective sampling unit.

    Returns:
        A tuple containing, respectively, a Series of incidence frequencies
        and an int of the total number of sampling units.
    """
    frequencies: dict[Any, int] = {}
    units = 0
    for series in incidence_series:
        units += 1
        for species, presence in series.items():
            if presence == 0:
                continue
            if species in frequencies:
                frequencies[species] += 1
            else:
                frequencies[species] = 1
    return pd.Series(frequencies)


@dataclass
class Estimate:
    """Estimation attributes."""

    estimate: float
    standard_error: float
    lower_bound: float
    upper_bound: float


@dataclass
class _CoverageData:
    """Extends Estimate with coverage attributes."""

    cutoff: int
    C_rare: float
    CV_rare: float
    n_rare: int
    S_rare: int


@dataclass
class CoverageBasedEstimate(_CoverageData, Estimate):
    """Extends Estimate with coverage attributes."""


def abundance_richness_metrics(
    frequencies: "Series[int]",
    cutoff: int = 10,
    adjust_cutoff: bool = True,
    confidence: float = 0.95,
) -> tuple[DataFrame, DataFrame]:
    """Calculates all nonparametric richness metrics from abundance data.

    Args:
        frequencies: A Series of species abundance frequencies.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        adjust_cutoff: Whether to adjust cutoff in heterogeneous samples.
        confidence: The confidence level of the confidence interval.

    Returns:
        A tuple (statistics, estimators), where statistics is a DataFrame of
        richness statistics and estimators is a DataFrame of richness
        estimators, standard errors, and confidence intervals.
    """
    # Get basic statistics
    counts, freqs = get_frequency_counts(frequencies)
    S_obs, n_obs = len(frequencies), sum(frequencies)
    freq_arr = frequencies.to_numpy()
    _, coverage = _abundance(counts, freqs, cutoff=int(np.max(freqs) + 1))
    rel_freqs = freq_arr / n_obs
    shannon = float(-np.sum(rel_freqs * np.log(rel_freqs)))
    simpson = float(np.sum(np.square(rel_freqs)))

    # Get all estimates
    shannon_estimate = index_shannon(
        counts,
        freqs,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
    )
    simpson_estimate = index_simpson(counts, freqs, confidence=confidence)
    homogeneous_mle_estimate = richness_homogeneous_mle(
        counts, freqs, confidence=confidence
    )
    chao1_estimate = richness_chao(
        counts, freqs, bias_correction=False, confidence=confidence
    )
    chao1bc_estimate = richness_chao(
        counts, freqs, bias_correction=True, confidence=confidence
    )
    homogeneous_estimate = richness_coverage(
        counts,
        freqs,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
        homogeneous=True,
    )
    ace_estimate = richness_coverage(
        counts,
        freqs,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
    )
    ace1_estimate = richness_coverage(
        counts,
        freqs,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
        bias_corrected=True,
    )

    # Build statistics dataframe
    statistics = pd.DataFrame.from_dict(
        {
            "Sample Shannon entropy": ["H_sh", shannon],
            "Sample Shannon diversity": ["D1=e^H_sh", np.exp(shannon)],
            "Sample Simpson index": ["H_si", simpson],
            "Sample Simpson diversity": ["D2=1/H_si", 1 / simpson],
            "# detected individuals": ["n_obs", n_obs],
            "# detected species": ["S_obs", S_obs],
            "coverage estimate": ["C", coverage.C_rare],
            "coefficient of variation": ["CV", coverage.CV_rare],
            "cut-off point": ["cutoff", cutoff],
            "adjusted cut-off point": ["cutoff", ace_estimate.cutoff],
            "# individuals in rare group": ["n_rare", ace_estimate.n_rare],
            "# species in rare group": ["S_rare", ace_estimate.S_rare],
            "coverage estimate of rare group": ["C_rare", ace_estimate.C_rare],
            "CV estimate in ACE": ["CV_rare", ace_estimate.CV_rare],
            "CV1 estimate in ACE-1": ["CV1_rare", ace1_estimate.CV_rare],
            "# individuals in abundant group": [
                "n_abun",
                n_obs - ace_estimate.n_rare,
            ],
            "# species in abundant group": [
                "S_abun",
                S_obs - ace_estimate.S_rare,
            ],
        },
        orient="index",
        columns=["Variable", "Value"],
        dtype=object,
    )

    # Build results dataframe
    results = pd.DataFrame.from_dict(
        {
            "Shannon Index": shannon_estimate,
            "Simpson Index": simpson_estimate,
            "Homogeneous (MLE)": homogeneous_mle_estimate,
            "Chao1": chao1_estimate,
            "Chao1-bc": chao1bc_estimate,
            "Homogeneous Model": homogeneous_estimate,
            "ACE": ace_estimate,
            "ACE-1": ace1_estimate,
        },
        orient="index",
        columns=["estimate", "standard_error", "lower_bound", "upper_bound"],
    ).astype(float)
    results.rename(
        columns={
            "estimate": "Estimate",
            "standard_error": "S.E.",
            "lower_bound": f"{confidence*100:.0f}% Lower",
            "upper_bound": f"{confidence*100:.0f}% Upper",
        },
        inplace=True,
    )

    return statistics, results


def abundance_richness_string(
    frequencies: "Series[int]",
    cutoff: int = 10,
    adjust_cutoff: bool = True,
    confidence: float = 0.95,
) -> str:
    """Calculates all nonparametric richness metrics from abundance data.

    Args:
        frequencies: A Series of species abundance frequencies.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        adjust_cutoff: Whether to adjust cutoff in heterogeneous samples.
        confidence: The confidence level of the confidence interval.

    Returns:
        A string containing richness statistics, estimators, standard errors,
        and confidence intervals.
    """
    statistics, results = abundance_richness_metrics(
        frequencies,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
    )
    return (
        statistics.to_string(float_format=lambda x: f"{x:.3f}")
        + "\n\n"
        + results.to_string(float_format=lambda x: f"{x:.3f}")
    )


def incidence_richness_metrics(
    raw_incidence: Sequence["Series[int]"],
    n: int = 1,
    units: int | None = None,
    cutoff: int = 10,
    adjust_cutoff: bool = True,
    confidence: float = 0.95,
) -> tuple[DataFrame, DataFrame]:
    """Calculates all nonparametric richness metrics from incidence data.

    Args:
        raw_incidence: If multiple Series are provided, each Series maps the
            name of a species to its presence in that sampling unit.
            If only one Series is provided, it is interpreted as frequencies.
        n: If `n > 1`, `raw_incidence` is interpreted as abundance frequencies
            which will be sampled randomly into `n` units.
        units: The number of sampling units. Used only if a single incidence
            frequency Series is provided and `n == 1`.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        adjust_cutoff: Whether to adjust cutoff in heterogeneous samples.
        confidence: The confidence level of the confidence interval.

    Returns:
        A tuple (statistics, estimators), where statistics is a DataFrame of
        richness statistics and estimators is a DataFrame of richness
        estimators, standard errors, and confidence intervals.
    """
    # Parse input
    if n > 1:
        raw_incidence = split_frequencies(raw_incidence[0], n)
        units = n
    if len(raw_incidence) > 1:
        frequencies = raw_to_frequencies(raw_incidence)
        units = len(raw_incidence)
    else:
        if units is None:
            raise ValueError(
                "If a single incidence frequency Series is provided,"
                " the number of units must also be specified."
            )
        frequencies = raw_incidence[0]

    # Get basic statistics
    counts, freqs = get_frequency_counts(frequencies)
    S_obs, n_obs = len(frequencies), sum(frequencies)
    _, coverage = _incidence(
        counts, freqs, units, cutoff=int(np.max(freqs) + 1)
    )

    # Get all estimates
    homogeneous_estimate = richness_coverage(
        counts,
        freqs,
        units=units,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
        homogeneous=True,
    )
    chao2_estimate = richness_chao(
        counts,
        freqs,
        units=units,
        bias_correction=False,
        confidence=confidence,
    )
    chao2bc_estimate = richness_chao(
        counts, freqs, units=units, bias_correction=True, confidence=confidence
    )
    ice_estimate = richness_coverage(
        counts,
        freqs,
        units=units,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
    )
    ice1_estimate = richness_coverage(
        counts,
        freqs,
        units=units,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
        bias_corrected=True,
    )

    # Build statistics dataframe
    statistics = pd.DataFrame.from_dict(
        {
            "# detected individuals": ["n_obs", n_obs],
            "# detected species": ["S_obs", S_obs],
            "# sampling units": ["T", units],
            "coverage estimate": ["C", coverage.C_rare],
            "coefficient of variation": ["CV", coverage.CV_rare],
            "cut-off point": ["cutoff", cutoff],
            "adjusted cut-off point": ["cutoff", ice_estimate.cutoff],
            "# individuals in infrequent group": [
                "n_infreq",
                ice_estimate.n_rare,
            ],
            "# species in infrequent group": ["S_infreq", ice_estimate.S_rare],
            "coverage estimate of infrequent group": [
                "C_infreq",
                ice_estimate.C_rare,
            ],
            "CV estimate in ICE": ["CV_infreq", ice_estimate.CV_rare],
            "CV1 estimate in ICE-1": ["CV1_infreq", ice1_estimate.CV_rare],
            "# individuals in frequent group": [
                "n_freq",
                n_obs - ice_estimate.n_rare,
            ],
            "# species in frequent group": [
                "S_freq",
                S_obs - ice_estimate.S_rare,
            ],
        },
        orient="index",
        columns=["Variable", "Value"],
        dtype=object,
    )

    # Build results dataframe
    results_dict = {
        "Homogeneous Model": homogeneous_estimate,
        "Chao2": chao2_estimate,
        "Chao2-bc": chao2bc_estimate,
        "ICE": ice_estimate,
        "ICE-1": ice1_estimate,
    }
    if len(raw_incidence) == 2:
        results_dict["Chapman"] = richness_chapman(
            raw_incidence[0], raw_incidence[1], confidence=confidence
        )
    results = pd.DataFrame.from_dict(
        results_dict,
        orient="index",
        columns=["estimate", "standard_error", "lower_bound", "upper_bound"],
    ).astype(float)
    results.rename(
        columns={
            "estimate": "Estimate",
            "standard_error": "S.E.",
            "lower_bound": f"{confidence*100:.0f}% Lower",
            "upper_bound": f"{confidence*100:.0f}% Upper",
        },
        inplace=True,
    )

    return statistics, results


def incidence_richness_string(
    raw_incidence: Sequence["Series[int]"],
    n: int = 1,
    units: int = 1,
    cutoff: int = 10,
    adjust_cutoff: bool = True,
    confidence: float = 0.95,
) -> str:
    """Calculates all nonparametric richness metrics from incidence data.

    Args:
        raw_incidence: If multiple Series are provided, each Series maps the
            name of a species to its presence in that sampling unit.
            If only one Series is provided, it is interpreted as frequencies.
        n: If `n > 1`, `raw_incidence` is interpreted as abundance frequencies
            which will be sampled randomly into `n` units.
        units: The number of sampling units. Used only if a single incidence
            frequency Series is provided and `n == 1`.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        adjust_cutoff: Whether to adjust cutoff in heterogeneous samples.
        confidence: The confidence level of the confidence interval.

    Returns:
        A string containing richness statistics, estimators, standard errors,
        and confidence intervals.
    """
    statistics, results = incidence_richness_metrics(
        raw_incidence,
        n=n,
        units=units,
        cutoff=cutoff,
        adjust_cutoff=adjust_cutoff,
        confidence=confidence,
    )
    return (
        statistics.to_string(float_format=lambda x: f"{x:.3f}")
        + "\n\n"
        + results.to_string(float_format=lambda x: f"{x:.3f}")
    )


def _confidence_interval(
    counts: Array,
    freqs: Array,
    S_est: float_type,
    S_var: float_type,
    confidence: float = 0.95,
    log_transform: bool = False,
) -> tuple[float_type, float_type]:
    """Calculates the confidence interval.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        S_est: The estimate.
        S_var: The variance of the estimate.
        confidence: The confidence level of the confidence interval.
        log_transform: Whether to use a log-transform in calculation.

    Returns:
        A tuple (lower, upper), which are the bounds of the interval.
    """
    sigmas = cast(float, scipy.stats.norm.interval(confidence)[1])
    S_obs = np.sum(counts)

    if log_transform:
        T = S_est - S_obs
        K = math.exp(sigmas * math.sqrt(math.log(1 + (S_var / np.square(T)))))
        return S_obs + (T / K), S_obs + (T * K)

    P = np.sum(counts * np.exp(-freqs)) / S_obs
    S_corr = S_obs / (1 - P)
    error = math.sqrt(S_var) / (1 - P)
    lower = S_corr - (sigmas * error)
    return np.maximum(S_obs, lower), S_corr + (sigmas * error)


def richness_chapman(
    frequencies_1: "Series[int]",
    frequencies_2: "Series[int]",
    confidence: float = 0.95,
) -> Estimate:
    """Computes Chapman estimator of richness from capture-recapture.

    Args:
        frequencies_1: A Series mapping species names to their presence in the
            respective sampling unit.
        frequencies_2: A Series mapping species names to their presence in the
            respective sampling unit.
        confidence: The confidence level of the confidence interval.

    Returns:
        Dataclass with the estimate, standard error, and confidence interval.
    """
    frequencies_1 = frequencies_1[frequencies_1 > 0]
    frequencies_2 = frequencies_2[frequencies_2 > 0]
    n, K = len(frequencies_1.index), len(frequencies_2.index)
    k = len(frequencies_1.index.intersection(frequencies_2.index))
    S_est = ((n + 1) * ((K + 1) / (k + 1))) - 1
    S_var = (
        ((n - k) * ((K - k) / (k + 2)))
        * ((n + 1) / (k + 1))
        * ((K + 1) / (k + 1))
    )

    sigmas = cast(float, scipy.stats.norm.interval(confidence)[1])
    theta_var = (
        (1 / (k + 0.5))
        + (1 / (K - k + 0.5))
        + (1 / (n - k + 0.5))
        + ((k + 0.5) / ((n - k + 0.5) * (K - k + 0.5)))
    )
    lower = (K + n - k - 0.5) + (
        (((K - k + 0.5) * (n - k + 0.5)) / (k + 0.5))
        * np.exp(-sigmas * np.sqrt(theta_var))
    )
    upper = (K + n - k - 0.5) + (
        (((K - k + 0.5) * (n - k + 0.5)) / (k + 0.5))
        * np.exp(sigmas * np.sqrt(theta_var))
    )
    return Estimate(
        float(S_est), float(np.sqrt(S_var)), float(lower), float(upper)
    )


def richness_homogeneous_mle(
    counts: Array, freqs: Array, confidence: float = 0.95
) -> Estimate:
    r"""Computes richness assuming equally abundant species.

    .. math::
        S_{obs} = S_{est}[1 - \exp(-n/S_{est})]

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        confidence: The confidence level of the confidence interval.

    Returns:
        Dataclass with the estimate, standard error, and confidence interval.
    """
    S_obs, n_obs = np.sum(counts), np.sum(freqs * counts)
    S_est = np.real(
        n_obs
        / (
            scipy.special.lambertw((-n_obs / S_obs) * np.exp(-n_obs / S_obs))
            + (n_obs / S_obs)
        )
    )

    error = S_obs - S_est * (1 - np.exp(-n_obs / S_est))
    if error > 1e-5:
        warnings.warn(
            f"Failed to find MLE solution, with an error of {error}",
            RuntimeWarning,
        )

    if abs(S_est - S_obs) >= 1e-5:
        S_var = np.exp(
            np.log(S_est)
            - np.logaddexp(
                -1, np.logaddexp(n_obs / S_est, np.log(n_obs / S_est))
            )
        )
    else:
        S_var = np.sum(counts * (np.exp(-freqs) - np.exp(-2 * freqs))) - (
            np.square(np.sum(freqs * np.exp(-freqs) * counts)) / n_obs
        )

    lower, upper = _confidence_interval(
        counts, freqs, S_est, S_var, confidence
    )
    return Estimate(
        float(S_est), float(np.sqrt(S_var)), float(lower), float(upper)
    )


def richness_chao(
    counts: Array,
    freqs: Array,
    units: int = 1,
    bias_correction: bool = True,
    confidence: float = 0.95,
) -> Estimate:
    """Computes Chao1/2-bc estimators of richness for abundance/incidence data.

    While derived as a universally valid lower bound, they approximate richness
    when undetected species have equal detection rates to singletons (CV>0.5).
    Bias correction is necessary if sampling rates are homogeneous.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        units: The number of sampling units in the incidence data.
            If `units` == 1, interpreted as abundance data.
        bias_correction: Whether to use Chao1/2-bc instead of Chao1/2.
        confidence: The confidence level of the confidence interval.

    Returns:
        Dataclass with the estimate, standard error, and confidence interval.
    """
    S_obs = np.sum(counts)
    n_obs = np.sum(freqs * counts) if units == 1 else units
    k = (n_obs - 1) / n_obs  # type: ignore[operator]

    c_1 = cast(float_type, _frequency_count(counts, freqs, 1))
    c_2 = cast(float_type, _frequency_count(counts, freqs, 2))

    # Get S_est
    if c_1 > 0 and c_2 > 0 and not bias_correction:
        # Eq. 1
        S_est = S_obs + ((k * c_1**2) / (2 * c_2))
    elif (c_1 > 0 and c_2 > 0 and bias_correction) or (c_1 > 1 and c_2 == 0):
        # Eq. 2
        S_est = S_obs + ((k * c_1 * (c_1 - 1)) / (2 * (c_2 + 1)))
    else:
        S_est = S_obs

    # Get S_var
    if c_1 > 0 and c_2 > 0 and not bias_correction:
        # Eq. 5
        S_var = c_2 * (
            ((k / 2) * np.square(c_1 / c_2))
            + (np.square(k) * np.power(c_1 / c_2, 3))
            + ((np.square(k) / 4) * np.power(c_1 / c_2, 4))
        )
    elif c_1 > 0 and c_2 > 0 and bias_correction:
        # Eq. 6
        S_var = (
            (k * (c_1 / 2) * ((c_1 - 1) / (c_2 + 1)))
            + (np.square(k) * (c_1 / 4) * np.square((2 * c_1 - 1) / (c_2 + 1)))
            + (
                np.square(k)
                * (c_2 / 4)
                * np.square(c_1 / (c_2 + 1))
                * np.square((c_1 - 1) / (c_2 + 1))
            )
        )
    elif c_1 > 1 and c_2 == 0:
        # Eq. 7
        S_var = (
            ((k / 2) * c_1 * (c_1 - 1))
            + ((np.square(k) / 4) * c_1 * (2 * c_1 - 1) ** 2)
            - ((np.square(k) / 4) * (c_1**4 / S_est))
        )
    else:
        # Eq. 8
        S_var = np.sum(counts * (np.exp(-freqs) - np.exp(-2 * freqs))) - (
            np.square(np.sum(freqs * np.exp(-freqs) * counts)) / n_obs
        )

    # Get CI
    if c_1 == 0 or (c_1 == 1 and c_2 == 0):
        # Eq. 14
        lower, upper = _confidence_interval(
            counts, freqs, S_est, S_var, confidence, log_transform=False
        )
    else:
        # Eq. 13
        lower, upper = _confidence_interval(
            counts, freqs, S_est, S_var, confidence, log_transform=True
        )

    return Estimate(
        float(S_est), float(np.sqrt(S_var)), float(lower), float(upper)
    )


def _abundance(
    counts: Array,
    freqs: Array,
    cutoff: int = 10,
    bias_corrected: bool = False,
    homogeneous: bool = False,
) -> tuple[float_type, _CoverageData]:
    """Computes ACE(-1) estimator of richness for abundance data.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        bias_corrected: Whether to use ACE-1 instead of ACE.
        homogeneous: Whether to use homogeneous assumption.

    Returns:
        A tuple (estimate, coverage), where estimate is the estimated richness
        and coverage is a dataclass with coverage statistics.
    """
    f_1 = cast(float_type, _frequency_count(counts, freqs, 1))
    S_rare = np.sum(np.where(freqs <= cutoff, counts, 0))
    S_obs = np.sum(counts)
    S_abun = S_obs - S_rare
    n_rare = np.sum(np.where(freqs <= cutoff, freqs * counts, 0))

    C_rare = 1 - (f_1 / n_rare)
    S_est = S_abun + (S_rare / C_rare)
    coverage = _CoverageData(
        cutoff,
        float(jax.lax.stop_gradient(C_rare)),
        1.0,
        int(n_rare),
        int(S_rare),
    )

    if homogeneous:
        return S_est, coverage

    summation = np.sum(
        np.where(freqs <= cutoff, freqs * (freqs - 1) * counts, 0)
    )
    squareCV_rare = np.maximum(
        0.0, (S_rare / C_rare) * (summation / (n_rare * (n_rare - 1))) - 1
    )

    if bias_corrected:
        squareCV_rare = squareCV_rare * (
            1 + ((1 - C_rare) / C_rare) * (summation / (n_rare - 1))
        )

    S_est = S_abun + (S_rare / C_rare) + (f_1 * squareCV_rare / C_rare)
    coverage.CV_rare = float(jax.lax.stop_gradient(np.sqrt(squareCV_rare)))
    return S_est, coverage


_abundance_value_and_grad: Callable[
    [Array, Array, int, bool, bool],
    tuple[tuple[float_type, _CoverageData], Array],
] = jax.value_and_grad(_abundance, has_aux=True, allow_int=True)


def _incidence(
    counts: Array,
    freqs: Array,
    units: int,
    cutoff: int = 10,
    bias_corrected: bool = False,
    homogeneous: bool = False,
) -> tuple[float_type, _CoverageData]:
    """Computes ICE(-1) estimator of richness for abundance data.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        units: The number of sampling units in the incidence data.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        bias_corrected: Whether to use ICE-1 instead of ICE.
        homogeneous: Whether to use homogeneous assumption.

    Returns:
        A tuple (estimate, coverage), where estimate is the estimated richness
        and coverage is a dataclass with coverage statistics.
    """
    q_1 = cast(float_type, _frequency_count(counts, freqs, 1))
    q_2 = cast(float_type, _frequency_count(counts, freqs, 2))
    S_obs = np.sum(counts)
    S_infreq = np.sum(np.where(freqs <= cutoff, counts, 0))
    S_freq = S_obs - S_infreq
    n_infreq = np.sum(np.where(freqs <= cutoff, freqs * counts, 0))

    if q_1 > 0 and q_2 > 0:
        A = 2 * q_2 / ((units - 1) * q_1 + 2 * q_2)
    elif q_1 > 0 and q_2 == 0:
        A = 2 / ((units - 1) * (q_1 - 1) + 2)
    else:
        A = 1

    C_infreq = 1 - (q_1 / n_infreq) * (1 - A)
    S_est = S_freq + (S_infreq / C_infreq)

    coverage = _CoverageData(
        cutoff,
        float(jax.lax.stop_gradient(C_infreq)),
        1.0,
        int(n_infreq),
        int(S_infreq),
    )

    if homogeneous:
        return S_est, coverage

    summation = np.sum(
        np.where(freqs <= cutoff, freqs * (freqs - 1) * counts, 0)
    )
    squareCV_infreq = np.maximum(
        0.0,
        (S_infreq / C_infreq)
        * (units / (units - 1))
        * (summation / (n_infreq * (n_infreq - 1)))
        - 1,
    )

    if bias_corrected:
        squareCV_infreq = squareCV_infreq * (
            1
            + (q_1 / C_infreq)
            * (units / (units - 1))
            * ((summation / n_infreq) / (n_infreq - 1))
        )

    S_est = S_est + (q_1 * squareCV_infreq / C_infreq)
    coverage.CV_rare = float(jax.lax.stop_gradient(np.sqrt(squareCV_infreq)))
    return S_est, coverage


_incidence_value_and_grad: Callable[
    [Array, Array, int, int, bool, bool],
    tuple[tuple[float_type, _CoverageData], Array],
] = jax.value_and_grad(_incidence, has_aux=True, allow_int=True)


def richness_coverage(
    counts: Array,
    freqs: Array,
    units: int = 1,
    cutoff: int = 10,
    adjust_cutoff: bool = True,
    bias_corrected: bool = False,
    homogeneous: bool = False,
    confidence: float = 0.95,
) -> CoverageBasedEstimate:
    """Computes A/ICE(-1) estimators of richness for abundance/incidence data.

    The abundance/incidence-based coverage estimators utilize the Good-Turing
    estimator of sample coverage. A cutoff is used to select rare species for
    coverage estimation, and should be higher for heterogeneous samples.
    The heterogeneity is characterized by the coefficient of variation (CV).
    For highly heterogeneous samples, the A/ICE-1 estimater should be used.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        units: The number of sampling units in the incidence data.
            If `units` == 1, interpreted as abundance data.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        adjust_cutoff: Whether to adjust cutoff in heterogeneous samples.
        confidence: The confidence level of the confidence interval.
        bias_corrected: Whether to use A/ICE-1 instead of A/ICE.
        homogeneous: Whether to use homogeneous assumption.

    Returns:
        Dataclass with the estimate, standard error, and confidence interval.
    """
    if units == 1 and cutoff < 2:
        raise ValueError(
            "cutoff must be greater than 1"
            " for abundance-based coverage estimators"
        )
    if units > 1 and cutoff < 1:  # pylint: disable=chained-comparison
        raise ValueError(
            "cutoff must be greater than 0"
            " for incidence-based coverage estimators"
        )
    if len(counts) <= 1 or 1 not in freqs:
        return CoverageBasedEstimate(
            np.nan, np.nan, np.nan, np.nan, cutoff, np.nan, np.nan, 0, 0
        )
    S_obs, n_obs = np.sum(counts), np.sum(freqs * counts)
    if adjust_cutoff:
        S_obs, n_obs = np.sum(counts), np.sum(freqs * counts)
        cutoff = int(np.maximum(cutoff, n_obs // S_obs))
    if units == 1:
        (S_est, coverage), gradient = _abundance_value_and_grad(
            counts, freqs, cutoff, bias_corrected, homogeneous
        )
    else:
        (S_est, coverage), gradient = _incidence_value_and_grad(
            counts, freqs, units, cutoff, bias_corrected, homogeneous
        )
    cov = np.diag(counts) - (counts * counts[..., np.newaxis] / S_est)
    S_var = np.sum(gradient * gradient[..., np.newaxis] * cov)
    lower, upper = _confidence_interval(
        counts, freqs, S_est, S_var, confidence, log_transform=True
    )
    return CoverageBasedEstimate(
        float(S_est),
        float(np.sqrt(S_var)),
        float(lower),
        float(upper),
        cutoff=cutoff,
        C_rare=coverage.C_rare,
        CV_rare=coverage.CV_rare,
        n_rare=coverage.n_rare,
        S_rare=coverage.S_rare,
    )


def _shannon(counts: Array, freqs: Array) -> float_type:
    """Computes the Shannon diversity index (Shannon entropy).

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.

    Returns:
        The estimated Shannon diversity index.
    """
    n_obs = np.sum(freqs * counts)
    C = 1 - (cast(float_type, _frequency_count(counts, freqs, 1)) / n_obs)
    C = np.where(C == 0, 1 - ((n_obs - 1) / n_obs), C)
    rel_freqs = freqs * C / n_obs
    return np.sum(
        -jax.lax.stop_gradient(counts)
        * rel_freqs
        * np.log(rel_freqs)
        / (1 - np.power(1 - rel_freqs, n_obs))
    )


def index_shannon(
    counts: Array,
    freqs: Array,
    cutoff: int = 10,
    adjust_cutoff: bool = True,
    confidence: float = 0.95,
    mode: Literal["normal", "bias-corrected", "homogeneous"] = "normal",
) -> Estimate:
    """Computes an estimator of Shannon's index for abundance data.

    Shannon's index is the log of the Hill number of order 1. Calculated here
    using an estimator that in part utilizes the ACE estimator of richness.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        adjust_cutoff: Whether to adjust cutoff in heterogeneous samples.
        confidence: The confidence level of the confidence interval.
        mode: Whether to user `normal` (A/ICE), `bias-corrected` (A/ICE-1), or
            homogeneous estimators.

    Returns:
        Dataclass with the estimate, standard error, and confidence interval.
    """
    # Cannot get correct variance from Chao & Shen 2003
    # Using variance of biased estimator instead
    del cutoff, adjust_cutoff, mode
    I_est = _shannon(counts, freqs)
    n_obs = np.sum(freqs * counts)
    rel_freqs = freqs / n_obs
    I_var = (
        np.sum(counts * rel_freqs * np.square(np.log(rel_freqs)))
        - np.square(np.sum(counts * rel_freqs * np.log(rel_freqs)))
    ) / n_obs

    # S_est = richness_coverage(
    #     counts, freqs, cutoff=cutoff, adjust_cutoff=adjust_cutoff, mode=mode
    # ).estimate
    # I_est, gradient = jax.value_and_grad(_shannon, allow_int=True)(
    #     counts, freqs
    # )
    # cov = np.diag(counts) - (counts * counts[..., np.newaxis] / S_est)
    # I_var = np.sum(gradient * gradient[..., np.newaxis] * cov)

    I_se = np.sqrt(I_var)
    sigmas = cast(float, scipy.stats.norm.interval(confidence)[1])
    lower, upper = I_est - sigmas * I_se, I_est + sigmas * I_se
    return Estimate(float(I_est), float(I_se), float(lower), float(upper))


def index_simpson(
    counts: Array, freqs: Array, confidence: float = 0.95
) -> Estimate:
    """Computes an estimator of Simpson's index for abundance data.

    Simpson's index is the inverse of the Hill number of order 2.

    Args:
        counts: An int Array of frequency counts.
        freqs: A float Array of frequencies.
        confidence: The confidence level of the confidence interval.

    Returns:
        Dataclass with the estimate, standard error, and confidence interval.
    """
    n_obs = np.sum(freqs * counts)
    I_est = np.sum(counts * freqs * (freqs - 1)) / (n_obs * (n_obs - 1))
    T_est = np.sum(counts * freqs * (freqs - 1) * (freqs - 2)) / (
        n_obs * (n_obs - 1) * (n_obs - 2)
    )
    a = (4 * (n_obs - 2)) / ((n_obs) * (n_obs - 1))
    b = (2 * (2 * n_obs - 3)) / ((n_obs) * (n_obs - 1))
    c = 2 / ((n_obs) * (n_obs - 1))
    I_var = (
        (a / (1 - b)) * T_est
        - (b / (1 - b)) * np.square(I_est)
        + (c / (1 - b)) * I_est
    )
    I_se = np.sqrt(I_var)
    sigmas = cast(float, scipy.stats.norm.interval(confidence)[1])
    lower, upper = I_est - sigmas * I_se, I_est + sigmas * I_se
    return Estimate(float(I_est), float(I_se), float(lower), float(upper))

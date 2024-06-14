"""Nonparametric estimation of species richness from abundance/incidence data.

This module provides functions for nonparametric estimation of species richness
from both abundance and incidence data. Given abundance frequencies in a Pandas
Series, richness metrics can be calculated with

    abundance_richness_metrics(abundance_frequencies)

Alternatively, given incidence frequencies in a Series

    incidence_richness_metrics([incidence_frequencies])

or, with raw incidence data, with each sampling unit represented as a Series

    incidence_richness_metrics([incidence_1, incidence_2, ...])

Most other functions in this module work on a countsogram of frequencies.
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
from typing import Any, Literal, TypeAlias, cast, no_type_check

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


def log1mexp(x: jax.typing.ArrayLike) -> Array:
    r"""Computes the element-wise log1mexp in a numerically stable way.

    .. math::
        \log \left( 1 - e^{-x} \right)

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    return np.where(x > 0.693, np.log1p(-np.exp(-x)), np.log(-np.expm1(-x)))  # type: ignore[operator]


def read_frequencies(path: str) -> "Series[int]":
    """Loads tab-delimited frequency data into a Pandas Series.

    The input data is assumed to have two columns separated by a tab character.
    Each row contains the species name and its corresponding frequency.

    Args:
        path: Path to the frequency TSV.

    Returns:
        A Series of frequencies. Names are stored in the index.
    """
    open_fn = gzip.open if os.path.splitext(path)[-1] == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8") as file_handle:  # type: ignore[operator]
        first_line = file_handle.readline()
        skiprows = 0 if "\t" in first_line else 1
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
        A tuple containing, respectively, a 1d float Array of frequency counts
        and a 1d int Array of their corresponding frequencies.
    """
    freqs, counts = np.unique(frequencies.to_numpy(), return_counts=True)
    return counts.astype(float), freqs


@jax.jit
def __frequency_count(
    counts: Array, freqs: Array, frequency: int
) -> float_type:
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

    estimate: float_type
    standard_error: float_type
    lower_bound: float_type
    upper_bound: float_type


@dataclass
class CoverageBasedEstimate(Estimate):
    """Extends Estimate with coverage attributes."""

    cutoff: int
    C_rare: float_type
    CV_rare: float_type
    n_rare: int
    S_rare: int


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
        A tuple containing, respectively, a DataFrame of richness statistics
        and a DataFrame of richness estimators, standard errors, and confidence
        intervals.
    """
    counts, freqs = get_frequency_counts(frequencies)

    # Get all  estimates
    results = {
        "Shannon Index": index_shannon(
            counts,
            freqs,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
        ),
        "Simpson Index": index_simpson(counts, freqs, confidence=confidence),
        "Homogeneous (MLE)": richness_homogeneous_mle(
            counts, freqs, confidence=confidence
        ),
        "Chao1": richness_chao(
            counts, freqs, bias_correction=False, confidence=confidence
        ),
        "Chao1-bc": richness_chao(
            counts, freqs, bias_correction=True, confidence=confidence
        ),
        "Homogeneous Model": richness_coverage(
            counts,
            freqs,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
            homogeneous=True,
        ),
        "ACE": richness_coverage(
            counts,
            freqs,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
        ),
        "ACE-1": richness_coverage(
            counts,
            freqs,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
            bias_corrected=True,
        ),
    }

    # Record basic statistics
    S_obs, n_obs = len(frequencies), sum(frequencies)
    freq_arr = frequencies.to_numpy()
    _, auxiliary = __abundance(counts, freqs, cutoff=int(np.max(freqs) + 1))
    rel_freqs = freq_arr / n_obs
    shannon = -np.sum(rel_freqs * np.log(rel_freqs))
    simpson = np.sum(np.square(rel_freqs))
    statistics = pd.DataFrame.from_dict(  # type: ignore[call-overload]
        {
            "Sample Shannon entropy": {"Variable": "H_sh", "Value": shannon},
            "Sample Shannon diversity": {
                "Variable": "D1=e^H_sh",
                "Value": np.exp(shannon),
            },
            "Sample Simpson index": {"Variable": "H_si", "Value": simpson},
            "Sample Simpson diversity": {
                "Variable": "D2=1/H_si",
                "Value": 1 / simpson,
            },
            "# detected individuals": {"Variable": "n_obs", "Value": n_obs},
            "# detected species": {"Variable": "S_obs", "Value": S_obs},
            "coverage estimate": {
                "Variable": "C",
                "Value": auxiliary["C_rare"],
            },
            "coefficient of variation": {
                "Variable": "CV",
                "Value": auxiliary["CV_rare"],
            },
            "cut-off point": {"Variable": "cutoff", "Value": cutoff},
            "adjusted cut-off point": {
                "Variable": "cutoff",
                "Value": results["ACE"].cutoff,  # type: ignore[attr-defined]
            },
            "# individuals in rare group": {
                "Variable": "n_rare",
                "Value": results["ACE"].n_rare,  # type: ignore[attr-defined]
            },
            "# species in rare group": {
                "Variable": "S_rare",
                "Value": results["ACE"].S_rare,  # type: ignore[attr-defined]
            },
            "coverage estimate of rare group": {
                "Variable": "C_rare",
                "Value": results["ACE"].C_rare,  # type: ignore[attr-defined]
            },
            "CV estimate in ACE": {
                "Variable": "CV_rare",
                "Value": results["ACE"].CV_rare,  # type: ignore[attr-defined]
            },
            "CV1 estimate in ACE-1": {
                "Variable": "CV1_rare",
                "Value": results["ACE-1"].CV_rare,  # type: ignore[attr-defined]
            },
            "# individuals in abundant group": {
                "Variable": "n_abun",
                "Value": n_obs - results["ACE"].n_rare,  # type: ignore[attr-defined]
            },
            "# species in abundant group": {
                "Variable": "S_abun",
                "Value": S_obs - results["ACE"].S_rare,  # type: ignore[attr-defined]
            },
        },
        orient="index",
    )

    df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["estimate", "standard_error", "lower_bound", "upper_bound"],
    ).astype(float)
    confidence_str = f"{confidence*100}".rstrip("0").rstrip(".")
    df.rename(
        columns={
            "estimate": "Estimate",
            "standard_error": "S.E.",
            "lower_bound": f"{confidence_str}% Lower",
            "upper_bound": f"{confidence_str}% Upper",
        },
        inplace=True,
    )

    return statistics, df


def incidence_richness_metrics(
    raw_incidence: Sequence["Series[int]"],
    n: int = 1,
    units: int = 1,
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
        A tuple containing, respectively, a DataFrame of richness statistics
        and a DataFrame of richness estimators, standard errors, and confidence
        intervals.
    """
    if n > 1:
        raw_incidence = split_frequencies(raw_incidence[0], n)
    if len(raw_incidence) > 1:
        frequencies = raw_to_frequencies(raw_incidence)
        units = len(raw_incidence)
    else:
        frequencies = raw_incidence[0]
    counts, freqs = get_frequency_counts(frequencies)

    # Get all estimators
    results = {
        "Homogeneous Model": richness_coverage(
            counts,
            freqs,
            units=units,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
            homogeneous=True,
        ),
        "Chao2": richness_chao(
            counts,
            freqs,
            units=units,
            bias_correction=False,
            confidence=confidence,
        ),
        "Chao2-bc": richness_chao(
            counts,
            freqs,
            units=units,
            bias_correction=True,
            confidence=confidence,
        ),
        "ICE": richness_coverage(
            counts,
            freqs,
            units=units,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
        ),
        "ICE-1": richness_coverage(
            counts,
            freqs,
            units=units,
            cutoff=cutoff,
            adjust_cutoff=adjust_cutoff,
            confidence=confidence,
            bias_corrected=True,
        ),
    }
    if units == 2:
        results["Chapman"] = richness_chapman(
            raw_incidence[0], raw_incidence[1], confidence=confidence
        )

    # Record basic statistics
    S_obs, n_obs = len(frequencies), sum(frequencies)
    _, auxiliary = __incidence(
        counts, freqs, units, cutoff=int(np.max(freqs) + 1)
    )
    statistics = pd.DataFrame.from_dict(  # type: ignore[call-overload]
        {
            "# detected individuals": {"Variable": "n_obs", "Value": n_obs},
            "# detected species": {"Variable": "S_obs", "Value": S_obs},
            "# sampling units": {"Variable": "T", "Value": units},
            "coverage estimate": {
                "Variable": "C",
                "Value": auxiliary["C_rare"],
            },
            "coefficient of variation": {
                "Variable": "CV",
                "Value": auxiliary["CV_rare"],
            },
            "cut-off point": {"Variable": "cutoff", "Value": cutoff},
            "adjusted cut-off point": {
                "Variable": "cutoff",
                "Value": results["ICE"].cutoff,  # type: ignore[attr-defined]
            },
            "# individuals in infrequent group": {
                "Variable": "n_infreq",
                "Value": results["ICE"].n_rare,  # type: ignore[attr-defined]
            },
            "# species in infrequent group": {
                "Variable": "S_infreq",
                "Value": results["ICE"].S_rare,  # type: ignore[attr-defined]
            },
            "coverage estimate of infrequent group": {
                "Variable": "C_infreq",
                "Value": results["ICE"].C_rare,  # type: ignore[attr-defined]
            },
            "CV estimate in ICE": {
                "Variable": "CV_infreq",
                "Value": results["ICE"].CV_rare,  # type: ignore[attr-defined]
            },
            "CV1 estimate in ICE-1": {
                "Variable": "CV1_infreq",
                "Value": results["ICE-1"].CV_rare,  # type: ignore[attr-defined]
            },
            "# individuals in frequent group": {
                "Variable": "n_freq",
                "Value": n_obs - results["ICE"].n_rare,  # type: ignore[attr-defined]
            },
            "# species in frequent group": {
                "Variable": "S_freq",
                "Value": S_obs - results["ICE"].S_rare,  # type: ignore[attr-defined]
            },
        },
        orient="index",
    )

    df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["estimate", "standard_error", "lower_bound", "upper_bound"],
    ).astype(float)
    confidence_str = f"{confidence*100}".rstrip("0").rstrip(".")
    df.rename(
        columns={
            "estimate": "Estimate",
            "standard_error": "S.E.",
            "lower_bound": f"{confidence_str}% Lower",
            "upper_bound": f"{confidence_str}% Upper",
        },
        inplace=True,
    )

    return statistics, df


def __confidence_interval(
    counts: Array,
    freqs: Array,
    S_est: float_type,
    S_var: float_type,
    confidence: float = 0.95,
    log_transform: bool = False,
) -> tuple[float_type, float_type]:
    """Calculates the confidence interval.

    Args:
        counts: A float Array of frequency counts.
        freqs: A float Array of frequencies.
        S_est: The estimate.
        S_var: The variance of the estimate.
        confidence: The confidence level of the confidence interval.
        log_transform: Whether to use a log-transform in calculation.

    Returns:
        The lower and upper bounds of the confidence interval.
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
    return Estimate(S_est, np.sqrt(S_var), lower, upper)


def richness_homogeneous_mle(
    counts: Array, freqs: Array, confidence: float = 0.95
) -> Estimate:
    r"""Computes richness assuming equally abundant species.

    .. math::
        S_{obs} = S_{est}[1 - \exp(-n/S_{est})]

    Args:
        counts: A float Array of frequency counts.
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

    lower, upper = __confidence_interval(
        counts, freqs, S_est, S_var, confidence
    )
    return Estimate(S_est, np.sqrt(S_var), lower, upper)


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
        counts: A float Array of frequency counts.
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

    c_1 = __frequency_count(counts, freqs, 1)
    c_2 = __frequency_count(counts, freqs, 2)

    # Get S_est
    if c_1 > 0 and c_2 > 0 and not bias_correction:
        # Eq. 1
        S_est = S_obs + ((k * c_1**2) / (2 * c_2))
    elif (c_1 > 0 and c_2 > 0 and bias_correction) or (c_1 > 1 and c_2 == 0):
        # Eq. 2
        S_est = S_obs + ((k * c_1 * (c_1 - 1)) / (2 * (c_2 + 1)))
    elif c_1 == 0 or (c_1 == 1 and c_2 == 0):
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
        lower, upper = __confidence_interval(
            counts, freqs, S_est, S_var, confidence, log_transform=False
        )
    else:
        # Eq. 13
        lower, upper = __confidence_interval(
            counts, freqs, S_est, S_var, confidence, log_transform=True
        )

    return Estimate(S_est, np.sqrt(S_var), lower, upper)


def __abundance(
    counts: Array,
    freqs: Array,
    cutoff: int = 10,
    bias_corrected: bool = False,
    homogeneous: bool = False,
) -> tuple[float_type, dict[str, Any]]:
    """Computes ACE(-1) estimator of richness for abundance data.

    Args:
        counts: A float Array of frequency counts.
        freqs: A float Array of frequencies.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        bias_corrected: Whether to use ACE-1 instead of ACE.
        homogeneous: Whether to use homogeneous assumption.

    Returns:
        A tuple containing, respectively, the estimated richness and a
        dictionary of auxiliary data.
    """
    f_1 = __frequency_count(counts, freqs, 1)
    S_rare = np.sum(np.where(freqs <= cutoff, counts, 0))
    S_obs = np.sum(counts)
    S_abun = S_obs - S_rare
    n_rare = np.sum(np.where(freqs <= cutoff, freqs * counts, 0))

    C_rare = 1 - (f_1 / n_rare)
    S_est = S_abun + (S_rare / C_rare)
    auxiliary = {
        "C_rare": C_rare,
        "CV_rare": 1.0,
        "n_rare": n_rare.astype(int),
        "S_rare": S_rare.astype(int),
    }

    if homogeneous:
        return S_est, auxiliary

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
    auxiliary["CV_rare"] = np.sqrt(squareCV_rare)
    return S_est, auxiliary


def __incidence(
    counts: Array,
    freqs: Array,
    units: int,
    cutoff: int = 10,
    bias_corrected: bool = False,
    homogeneous: bool = False,
) -> tuple[float_type, dict[str, Any]]:
    """Computes ICE(-1) estimator of richness for abundance data.

    Args:
        counts: A float Array of frequency counts.
        freqs: A float Array of frequencies.
        units: The number of sampling units in the incidence data.
        cutoff: Frequency cutoff for rare species used for estimating coverage.
        bias_corrected: Whether to use ICE-1 instead of ICE.
        homogeneous: Whether to use homogeneous assumption.

    Returns:
        A tuple containing, respectively, the estimated richness and a
        dictionary of auxiliary data.
    """
    q_1 = __frequency_count(counts, freqs, 1)
    q_2 = __frequency_count(counts, freqs, 2)
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

    auxiliary = {
        "C_rare": C_infreq,
        "CV_rare": 1.0,
        "n_rare": n_infreq.astype(int),
        "S_rare": S_infreq.astype(int),
    }

    if homogeneous:
        return S_est, auxiliary

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
    auxiliary["CV_rare"] = np.sqrt(squareCV_infreq)
    return S_est, auxiliary


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
        counts: A float Array of frequency counts.
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
        (S_est, auxiliary), gradient = (
            jax.value_and_grad(__abundance, has_aux=True, allow_int=True)
        )(counts, freqs, cutoff, bias_corrected, homogeneous)
    else:
        (S_est, auxiliary), gradient = jax.value_and_grad(
            __incidence, has_aux=True, allow_int=True
        )(counts, freqs, units, cutoff, bias_corrected, homogeneous)
    cov = np.diag(counts) - (counts * counts[..., np.newaxis] / S_est)
    S_var = np.sum(gradient * gradient[..., np.newaxis] * cov)
    lower, upper = __confidence_interval(
        counts, freqs, S_est, S_var, confidence, log_transform=True
    )
    return CoverageBasedEstimate(
        S_est, np.sqrt(S_var), lower, upper, cutoff, **auxiliary
    )


def __shannon(counts: Array, freqs: Array) -> float_type:
    """Computes the Shannon diversity index (Shannon entropy).

    Args:
        counts: A float Array of frequency counts.
        freqs: A float Array of frequencies.

    Returns:
        The estimated Shannon diversity index.
    """
    n_obs = np.sum(freqs * counts)
    C = 1 - (__frequency_count(counts, freqs, 1) / n_obs)
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
        counts: A float Array of frequency counts.
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
    I_est = __shannon(counts, freqs)
    n_obs = np.sum(freqs * counts)
    rel_freqs = freqs / n_obs
    I_var = (
        np.sum(counts * rel_freqs * np.square(np.log(rel_freqs)))
        - np.square(np.sum(counts * rel_freqs * np.log(rel_freqs)))
    ) / n_obs

    # S_est = richness_coverage(
    #     counts, freqs, cutoff=cutoff, adjust_cutoff=adjust_cutoff, mode=mode
    # ).estimate
    # I_est, gradient = jax.value_and_grad(__shannon, allow_int=True)(
    #     counts, freqs
    # )
    # cov = np.diag(counts) - (counts * counts[..., np.newaxis] / S_est)
    # I_var = np.sum(gradient * gradient[..., np.newaxis] * cov)

    I_se = np.sqrt(I_var)
    sigmas = cast(float, scipy.stats.norm.interval(confidence)[1])
    lower, upper = I_est - sigmas * I_se, I_est + sigmas * I_se
    return Estimate(I_est, I_se, lower, upper)


def index_simpson(
    counts: Array, freqs: Array, confidence: float = 0.95
) -> Estimate:
    """Computes an estimator of Simpson's index for abundance data.

    Simpson's index is the inverse of the Hill number of order 2.

    Args:
        counts: A float Array of frequency counts.
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
    return Estimate(I_est, I_se, lower, upper)

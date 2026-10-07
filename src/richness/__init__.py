"""Nonparametric estimation of species richness and library complexity.

This package answers two related questions about a sample. The estimators in
:mod:`richness.richness` are asymptotic: they estimate how many species
exist, detected or not. Given abundance frequencies in a Pandas Series, all
of them are calculated with

    abundance_richness_metrics(abundance_frequencies)

or, for incidence data recorded across sampling units,

    incidence_richness_metrics([incidence_1, incidence_2, ...])

The complexity curve in :mod:`richness.complexity` instead predicts how many
distinct species, or molecules, would be observed if sampling continued to a
stated depth, which is the question posed by a sequencing experiment

    complexity_metrics(frequencies)

Because a yield at a stated depth is not an asymptotic richness estimate,
the two are reported separately. Everything public in both modules is
re-exported here.
"""

from __future__ import annotations

from .complexity import (
    MAX_TERMS,
    MIN_TERMS,
    ComplexityError,
    ContinuedFraction,
    complexity_curve,
    complexity_metrics,
    complexity_string,
    fit_rational,
    rarefaction_yield,
    yield_rational,
)
from .richness import (
    CoverageBasedEstimate,
    Estimate,
    abundance_richness_metrics,
    abundance_richness_string,
    float_type,
    get_frequency_counts,
    incidence_richness_metrics,
    incidence_richness_string,
    index_shannon,
    index_simpson,
    log1mexp,
    logexpm1,
    logsubexp,
    raw_to_frequencies,
    read_frequencies,
    richness_chao,
    richness_chapman,
    richness_coverage,
    richness_homogeneous_mle,
    split_frequencies,
)

__version__ = "1.1.0"

__all__ = [
    "MAX_TERMS",
    "MIN_TERMS",
    "ComplexityError",
    "ContinuedFraction",
    "CoverageBasedEstimate",
    "Estimate",
    "abundance_richness_metrics",
    "abundance_richness_string",
    "complexity_curve",
    "complexity_metrics",
    "complexity_string",
    "fit_rational",
    "float_type",
    "get_frequency_counts",
    "incidence_richness_metrics",
    "incidence_richness_string",
    "index_shannon",
    "index_simpson",
    "log1mexp",
    "logexpm1",
    "logsubexp",
    "rarefaction_yield",
    "raw_to_frequencies",
    "read_frequencies",
    "richness_chao",
    "richness_chapman",
    "richness_coverage",
    "richness_homogeneous_mle",
    "split_frequencies",
    "yield_rational",
]

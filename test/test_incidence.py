# type: ignore
# pylint: disable=missing-function-docstring, missing-module-docstring
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

import richness

from . import check_estimate, check_estimates

spader = rpackages.importr("SpadeR")
incidence_raw = (
    rpackages.data(spader)
    .fetch("ChaoSharedData")["ChaoSharedData"]
    .rx2("Inci_raw")
)

spader_stats, _, spader_estimates = spader.ChaoSpecies(
    incidence_raw, datatype="incidence_raw", k=10, conf=0.95
)

with (ro.default_converter + pandas2ri.converter).context():
    incidence_raw_df = ro.conversion.get_conversion().rpy2py(incidence_raw)
    incidence_raw_series = [incidence_raw_df[col] for col in incidence_raw_df]

richness_stats, richness_estimates = richness.incidence_richness_metrics(
    incidence_raw_series, cutoff=10, adjust_cutoff=False, confidence=0.95
)
richness_estimates = richness_estimates.round(decimals=3)


def test_units() -> None:
    check_estimate(
        richness_stats.loc["# sampling units", "Value"],
        spader_stats.rx("    Number of sampling units", "Value"),
    )


def test_coverage() -> None:
    check_estimate(
        richness_stats.loc["coverage estimate", "Value"].round(3),
        spader_stats.rx("    Coverage estimate for entire dataset", "Value"),
    )


def test_cv() -> None:
    check_estimate(
        richness_stats.loc["coefficient of variation", "Value"].round(3),
        spader_stats.rx("    CV for entire dataset", "Value"),
    )


def test_coverage_ice() -> None:
    check_estimate(
        richness_stats.loc[
            "coverage estimate of infrequent group", "Value"
        ].round(2),
        spader_stats.rx(
            "    Estimated sample coverage for infrequent group", "Value"
        ),
    )


def test_cv_ice() -> None:
    check_estimate(
        richness_stats.loc["CV estimate in ICE", "Value"].round(3),
        spader_stats.rx(
            "    Estimated CV for infrequent group in ICE", "Value"
        ),
    )


def test_cv_ice1() -> None:
    check_estimate(
        richness_stats.loc["CV1 estimate in ICE-1", "Value"].round(3),
        spader_stats.rx(
            "    Estimated CV1 for infrequent group in ICE-1", "Value"
        ),
    )


def test_n_rare() -> None:
    check_estimate(
        richness_stats.loc["# individuals in infrequent group", "Value"],
        spader_stats.rx(
            "    Total number of incidences in infrequent group", "Value"
        ),
    )


def test_S_rare() -> None:
    check_estimate(
        richness_stats.loc["# species in infrequent group", "Value"],
        spader_stats.rx(
            "    Number of observed species for infrequent group", "Value"
        ),
    )


def test_homogeneous() -> None:
    check_estimates(
        richness_estimates.loc["Homogeneous Model"],
        spader_estimates.rx("    Homogeneous Model", True),
    )


def test_chao2() -> None:
    check_estimates(
        richness_estimates.loc["Chao2"],
        spader_estimates.rx("    Chao2 (Chao, 1987)", True),
    )


def test_chao2_bc() -> None:
    check_estimates(
        richness_estimates.loc["Chao2-bc"],
        spader_estimates.rx("    Chao2-bc", True),
    )


def test_ice() -> None:
    check_estimates(
        richness_estimates.loc["ICE"],
        spader_estimates.rx("    ICE (Lee & Chao, 1994)", True),
    )


def test_ice_1() -> None:
    check_estimates(
        richness_estimates.loc["ICE-1"],
        spader_estimates.rx("    ICE-1 (Lee & Chao, 1994)", True),
    )

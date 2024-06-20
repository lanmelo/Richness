# type: ignore
# pylint: disable=missing-function-docstring, missing-module-docstring
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

import richness

from . import check_estimate, check_estimates

spader = rpackages.importr("SpadeR")
abundance = (
    rpackages.data(spader)
    .fetch("ChaoSpeciesData")["ChaoSpeciesData"]
    .rx2("Abu")
)

spader_diversity = spader.Diversity(abundance, datatype="abundance")
spader_stats, _, spader_estimates = spader.ChaoSpecies(
    abundance, datatype="abundance", k=10, conf=0.95
)

with (ro.default_converter + pandas2ri.converter).context():
    abundance_series = (
        ro.conversion.get_conversion().rpy2py(abundance).iloc[:, 0]
    )

richness_stats, richness_estimates = richness.abundance_richness_metrics(
    abundance_series, cutoff=10, adjust_cutoff=False, confidence=0.95
)


def test_coverage() -> None:
    check_estimate(
        round(richness_stats.loc["coverage estimate", "Value"], 3),
        spader_stats.rx("    Coverage estimate for entire dataset", "Value"),
    )


def test_cv() -> None:
    check_estimate(
        round(richness_stats.loc["coefficient of variation", "Value"], 3),
        spader_stats.rx("    CV for entire dataset", "Value"),
    )


def test_coverage_ace() -> None:
    check_estimate(
        round(
            richness_stats.loc["coverage estimate of rare group", "Value"], 3
        ),
        spader_stats.rx(
            "    Estimate of the sample coverage for rare group", "Value"
        ),
    )


def test_cv_ace() -> None:
    check_estimate(
        round(richness_stats.loc["CV estimate in ACE", "Value"], 3),
        spader_stats.rx("    Estimate of CV for rare group in ACE", "Value"),
    )


def test_cv_ace1() -> None:
    check_estimate(
        round(richness_stats.loc["CV1 estimate in ACE-1", "Value"], 2),
        spader_stats.rx(
            "    Estimate of CV1 for rare group in ACE-1", "Value"
        ),
    )


def test_n_rare() -> None:
    check_estimate(
        richness_stats.loc["# individuals in rare group", "Value"],
        spader_stats.rx(
            "    Number of observed individuals for rare group", "Value"
        ),
    )


def test_S_rare() -> None:
    check_estimate(
        richness_stats.loc["# species in rare group", "Value"],
        spader_stats.rx(
            "    Number of observed species for rare group", "Value"
        ),
    )


def test_shannon() -> None:
    check_estimates(
        richness_estimates.loc["Shannon Index"].round(3),
        spader_diversity.rx2("Shannon_index").rx("     Chao & Shen", True),
        abs_tol=1e-2,
    )


def test_simpson() -> None:
    check_estimates(
        richness_estimates.loc["Simpson Index"].round(5),
        spader_diversity.rx2("Simpson_index").rx("     MVUE", True),
        abs_tol=5e-3,
    )


def test_homogeneous() -> None:
    check_estimates(
        richness_estimates.loc["Homogeneous Model"].round(3),
        spader_estimates.rx("    Homogeneous Model", True),
    )


def test_homogeneous_mle() -> None:
    check_estimates(
        richness_estimates.loc["Homogeneous (MLE)"].round(3),
        spader_estimates.rx("    Homogeneous (MLE)", True),
    )


def test_chao1() -> None:
    check_estimates(
        richness_estimates.loc["Chao1"].round(3),
        spader_estimates.rx("    Chao1 (Chao, 1984)", True),
    )


def test_chao1_bc() -> None:
    check_estimates(
        richness_estimates.loc["Chao1-bc"].round(3),
        spader_estimates.rx("    Chao1-bc", True),
    )


def test_ace() -> None:
    check_estimates(
        richness_estimates.loc["ACE"].round(3),
        spader_estimates.rx("    ACE (Chao & Lee, 1992)", True),
    )


def test_ace_1() -> None:
    check_estimates(
        richness_estimates.loc["ACE-1"].round(3),
        spader_estimates.rx("    ACE-1 (Chao & Lee, 1992)", True),
    )

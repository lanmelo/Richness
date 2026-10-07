# pylint: disable=missing-function-docstring, missing-module-docstring
from __future__ import annotations

import jax.numpy as np
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from pandas import DataFrame, Series
from pytest import approx

if not rpackages.isinstalled("SpadeR"):
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)
    utils.install_packages("SpadeR")

# preseqR is the reference implementation of the complexity curve, but it was
# removed from CRAN, so it has to be installed from the CRAN archive. Tests
# that compare against it are skipped if that install is unavailable.
PRESEQR_ARCHIVE = (
    "https://cran.r-project.org/src/contrib/Archive"
    "/preseqR/preseqR_4.0.0.tar.gz"
)


def _install_preseqr() -> bool:
    if rpackages.isinstalled("preseqR"):
        return True
    try:
        rpackages.importr("utils").install_packages(
            PRESEQR_ARCHIVE, repos=ro.NULL, type="source"
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return bool(rpackages.isinstalled("preseqR"))


preseqr_installed = _install_preseqr()


def check_estimate(
    richness_estimate: float,
    spader_estimate: ro.RObject,
    abs_tol: float | None = None,
) -> None:
    assert richness_estimate == approx(
        np.array(spader_estimate[0], dtype=float), abs=abs_tol
    )


def check_estimates(
    richness_estimate: "Series[int]" | DataFrame,
    spader_estimate: ro.RObject,
    abs_tol: float | None = None,
) -> None:
    for richness_val, spader_val in zip(
        richness_estimate, np.array(spader_estimate, dtype=float)
    ):
        assert richness_val == approx(spader_val, abs=abs_tol)

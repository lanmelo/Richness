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

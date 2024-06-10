# SpeciesRichness

Nonparametric estimation of species richness from abundance/incidence data.

Richness refers to the number of species (detected and undetected) in an area.
Data can be recorded as either 1) abundance, where the frequency of individuals
of a species is tallied, and 2) incidence, where the (binary) presence of a
species is tallied across multiple sampling units.

While it is easy to calculate under the assumption of homogeneity
(in which each species is equally abundant),
overcoming this assumption requires different estimators.
This module provides functions for nonparametric estimation of species richness
from both abundance and incidence data.

## Installation and Usage
Install with `pip install git+https://github.com/lanmelo/SpeciesRichness.git`

Accessible in the command line with the `richness` command

## Statistics
* Note: Species richness is also the Hill number of order 0.
    * A Hill number is expressed in units of "effective numbers of species",
    the number of species needed to achieve the same diversity under a homogeneous assumption
* Shannon entropy
    * In information theory, quantifies the level of uncertainty in species abundance
    * Logarithm of Hill number of order 1
* Simpson's index
    * The probability that two individuals taken at random are from the same species
    * Reciprocal of Hill number of order 2
* Coverage estimate (`C`)
    * The proportion of individuals that belong to a species in the sample
    * Equal to the proportion of counts that come from non-singleton reads
    * The higher the coverage, the more of the population you have sampled
* Coefficient of variation (`CV`)
    * Related to the sample standard deviation of species frequencies
    * Represents the heterogeneity in counts between species

## Abundance-based metrics
Abundance data counts the frequency of individuals from each species in an area.
* Homogeneous-MLE
    * Approximate MLE under a homogeneous assumption (`CV = 0`)
* Homogeneous
    * Equivalent to ACE under a homogeneous assumption (`CV = 0`)
* Chao1
    * Universally valid **lower bound** that approximates richness if coverage is high (`C > 0.5`)
    * Bias-corrected form Chao1-bc should be used for homogeneous (`CV ≈ 0`) data
* Abundance-based Coverage Estimator (ACE)
    * Utilizes a frequency cutoff to calculate Good-Turing sample coverage
    * Bias-corrected form ACE-1 should be used for heterogenous (`CV > 2`) data

## Incidence-based metrics
Incidence data uses the (binary) presence of a species across multiple sampling units.
Note that if the number of sampling units should be high to get reliable estimates.
* Chapman Estimator
    * Bias-corrected form of Lincoln-Petersen index, which is the MLE solution
    * Essentially the MLE estimator under a homogeneous assumption (`CV = 0`)
* Homogeneous
    * Equivalent to ICE under a homogeneous assumption (`CV = 0`)
* Chao2
    * Universally valid **lower bound**
    * Bias-corrected form Chao2-bc should be used for homogeneous (`CV ≈ 0`) data
    * If coverage is high (`C > 0.5`), it approximates the estimated richness
* Incidence-based Coverage Estimator (ICE)
    * Utilizes a frequency cutoff to calculate Good-Turing sample coverage
    * Bias-corrected form ICE-1 should be used for heterogenous (`CV > 2`) data

## References:
* Gotelli N. J. and Chao A. (2013) Measuring and Estimating Species Richness, Species Diversity, and Biotic Similarity from Sampling Data. Levin S.A. (ed.) Encyclopedia of Biodiversity, second edition, Volume 5, pp. 195- 211.
    * Provides an overview on all of these metrics
* Chao, A. and Chiu, C. H. (2016) Species richness: estimation and comparison. Wiley StatsRef: Statistics Reference Online. 1-26
    * Provides a deeper insight into each metric, their benefits, and their drawbacks
* Chao, A. and Yang, M. C. K. (1993) Stopping Rules and Estimation for Recapture Debugging with Unequal Failure Rates. Biometrika, Vol. 80, No. 1, pp. 193-201
* Chao, A. and Lee, S. M. (1992). Estimating the Number of Classes via Sample Coverage. Journal of the American Statistical Association, Vol. 87, No. 417, pp. 210-217
    * These two papers cover ACE and ACE-1
* Chao, A. and Shen, T. J. (2003) Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample. Environmental and Ecological Statistics 10, pp. 429-443
* Tiffeau-Mayer, A. (2024) Unbiased estimation of sampling variance for Simpson's diversity index. arXiv:2310.03439
    * Covers Shannon/Simpson diversity estimation
* Colwell, R. (2013) EstimateS: Statistical estimation of species richness and shared species from samples. Version 9 and earlier. User’s Guide and application.
* Chao, A. and Ma, K. and Hsieh, T. C. and Chiu, C. H. (2016) User’s Guide for Online Program SpadeR (Species-richness Prediction And Diversity Estimation in R)
    * These two programs offer reference implementations for these metrics
* Chao, A. and Gotelli, N. J. and Hsieh, T. C. and Sander, E. L. and Ma, K. H. and Colwell, R. K. and Ellison, A. M. (2014) Rarefaction and extrapolation with Hill numbers: a framework for sampling and estimation in species diversity studies. Ecological Monographs, 84(1), pp. 45–67
    * Covers the relationship between Hill numbers, richness, and Simpson's and Shannon's indices
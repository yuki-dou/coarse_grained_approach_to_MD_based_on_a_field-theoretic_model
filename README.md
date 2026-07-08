# Coarse-grained approach to MD based on a field-theoretic model

This repository accompanies the research paper on a field-theoretic coarse-grained model for molecular dynamics simulations of knot unknotting transitions. The main research paper is located in the `report/` directory.

## Data

Data obtained from the results of the model

## Results 

Data obtained by statistical analysis

## Scripts

Analysis scripts for processing simulation results and computing transition metrics.

### `CDF.py`

Extracts knot populations from aggregated simulation data and fits a logistic decay function in logarithmic temperature scale to the knot 5₂ population curves. Performs bootstrap uncertainty estimation for transition temperatures. Outputs a summary table with T₁₀, T₉₀, ΔT, and T₅₀ values with uncertainties for each simulation mode.

### `PDF.py`

Reconstructs individual unknotting temperatures from aggregated population data by tracking the decline of knot 5₂ across temperature steps. Fits a logistic distribution (CDF/PDF) to the raw unknotting temperatures with bootstrap error estimation. Outputs summary statistics including median T₅₀, fitted T₅₀ with uncertainty, mean, and standard deviation.

### `assessment_of_significance.py`

Statistical test for knot intermediacy. For each simulation mode, tests the null hypothesis that the knot 3₁ population is noise (binomial with specified probability) against the alternative of a significant intermediate state. Uses Wilson score confidence intervals and binomial tests with Bonferroni correction for multiple comparisons. Outputs a summary verdict table classifying each mode as NOISE or POSSIBLE intermediate.

### `extrapolation.py`

Extrapolates the equilibrium unknotting temperature T_eq from T₅₀ vs 1/N using weighted least squares linear regression. Reads transition metrics from the CDF output table, excludes the outlier at 20M steps and plateau points (N ≥ 30M). Reports T_eq with uncertainty, R², and plateau value.

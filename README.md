# GPverdict

**Choose genomic prediction methods for selection — from your own cross-validation.**

Web version (runs in your browser; your data are not uploaded): **https://nblvguohao.github.io/gpverdict/**

Given cross-validated predictions of several methods in several environments, GPverdict answers the
questions a breeding programme asks before adopting a prediction model for within-environment
truncation selection:

1. **Which accuracy metrics can rank methods for this decision?** A metric represents selection within an
   environment only if order-preserving transformations of the predictions leave it unchanged and reversing
   their order changes it (Tier I: rank correlation, realised selection differential, top-*k* hit rate, NDCG).
   Pearson *r* is unmoved by calibration only (Tier II); RMSE, MAE, *R*² and pooled metrics are neither
   (Tier III). GPverdict checks this on your own predictions.
2. **Which method leads, and is its lead real?** Methods ranked by the within-environment rank and Pearson
   correlations (both suit this decision; neither consistently selected better in published tests), with 95 %
   rank intervals, and the methods your trial cannot tell apart from the leaders: those within *k*/√*N* of the
   best on either correlation, where *N* is the number of genotype–environment cells (Gaussian lower bound).
3. **What would RMSE have chosen?** The method an error-magnitude ranking picks, the share of method pairs
   RMSE and Pearson *r* order oppositely, and an out-of-sample test of how much of the attainable selection
   gain each rule's choice recovers (environments split in halves; genotypes split within environments when
   there are fewer than eight environments).
4. **How large must a trial be?** Genotype–environment cells needed to resolve a given accuracy gain.

## Use

```bash
pip install git+https://github.com/nblvguohao/gpverdict
gpverdict predictions.csv --frac 0.10 --out report.html
```

```python
import gpverdict as gv
v = gv.verdict("predictions.csv", frac=0.10)
open("report.html", "w").write(gv.render_html(v))
```

Input: a CSV with one row per environment, genotype and method and the columns `environment`, `genotype`,
`observed`, `predicted`, `method` (`Env`, `k`, `y`, `p` are also accepted). Environments with fewer than
`--min-genotypes` genotypes (default 10) are skipped. Dependencies: numpy and pandas.

### From rrBLUP, BGLR or any other software

Write one row per environment and genotype with the observed value and one column of cross-validated
predictions per method, then either reshape it in R (`examples/export_from_R.R` shows rrBLUP) or let
GPverdict do it:

```python
import pandas as pd, gpverdict as gv
wide = pd.read_csv("cv_predictions_wide.csv")        # environment, genotype, observed, rrBLUP, BayesB, ...
v = gv.verdict(gv.from_wide(wide, "environment", "genotype", "observed"))
```

## Validation

GPverdict reproduces the published results of Lv and Gu (2026) exactly; `pytest tests` checks the reversal
rate (25.0 % of 136 method pairs) and the out-of-sample recoveries (58.7 %, 78.5 % and 37.9 %) on the spring
wheat example, and that the empirical tiers follow the criterion.

Example data: cross-validated predictions of 17 methods for Fusarium head blight resistance in the Uniform
Regional Scab Nursery of spring wheat (109 environments), built from data released under CC0
(Brault et al. 2025, Plant Methods 21; data doi:10.5061/dryad.wstqjq2z0).

## Citation

G. Lv, L. Gu, A decision-based criterion and an open tool for choosing genomic prediction methods across
environments (2026). Analysis code for the paper: https://github.com/nblvguohao/gp-decision-criterion

Licence: MIT.

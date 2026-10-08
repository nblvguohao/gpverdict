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
5. **Does a method still lead in a new year?** *(forward evaluation, new in 1.2.0)* Given predictions of each
   target year made from earlier years only, each method is compared with a reference method within
   environments, the differences are pooled over target years (DerSimonian–Laird with a floor on the
   standard errors; Hartung–Knapp alongside), and a difference is called resolved only when it is at least
   3/√*N* and its 95 % interval excludes zero; otherwise it is reported as detectable but below resolution,
   or tied. New and old lines are reported separately when the file says which lines are new.

## Use

### Methods compared within trials

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

### Forward evaluation over target years

```bash
gpverdict forward forward_predictions.csv --reference cell_reml --out forward.html
```

```python
import gpverdict as gv
f = gv.forward_verdict("forward_predictions.csv", reference="cell_reml", min_genotypes=25)
open("forward.html", "w").write(gv.forward_html(f))
```

Input: one row per environment, genotype and method with the columns `environment`, `year`, `genotype`,
`observed`, `predicted`, `method`; optional `dataset` (target years of each dataset are pooled together and
reported by dataset), and `new` (1 = line first tested in the target year) or `first_year` (first year the
line was tested), which add separate verdicts for new and old lines. Environments with fewer than
`--min-genotypes` lines (default 25) are skipped. The environment-cluster bootstrap uses `--bootstrap`
replicates (default 2,000) and `--seed`. `examples/forward_MU_SOY.csv` holds the forward predictions of 11
methods for the 2020 soybean trials of the benchmark (Canella Vieira et al. 2022; data doi:10.5061/dryad.z8w9ghxf9, CC0). If you already have
year-level differences and standard errors, `gv.pool_year_effects(table, resolution=...)` pools them directly.

### A complete R example

`examples/rrblup_to_gpverdict.R` fits GBLUP, rrBLUP marker regression and a Gaussian kernel with
rrBLUP in five-fold leave-genotypes-out cross-validation on the spring wheat data in `examples/`
(384 lines, 2,302 markers, 109 environments; CC0), writes `predictions_long.csv` and is checked end
to end: `Rscript rrblup_to_gpverdict.R && gpverdict predictions_long.csv --min-genotypes 12`.

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

GPverdict reproduces the published results of Lv et al. (2026) exactly; `pytest tests` checks the reversal
rate (25.0 % of 136 method pairs) and the out-of-sample recoveries (58.7 %, 78.5 % and 37.9 %) on the spring
wheat example, and that the empirical tiers follow the criterion. `tests/test_forward.py` checks that the
forward module reproduces the pooled estimates, both intervals and the verdicts of the forward benchmark
(Lv and Gu 2026, manuscript in preparation; 48 target years) to 1e-9 from the year-level tables
archived with that paper (`tests/data`), and
that per-cell entry reproduces the benchmark's environment-level differences for the soybean example (to
1e-12; to 5e-4 for the two kNN methods, because the archived predictions are stored in single precision
and near-ties change a few ranks). The *t* quantile for the Hartung–Knapp interval is computed without
SciPy and agrees with `scipy.stats.t.ppf` to 3e-10.

Example data: cross-validated predictions of 17 methods for Fusarium head blight resistance in the Uniform
Regional Scab Nursery of spring wheat (109 environments), built from data released under CC0
(Brault et al. 2025, Plant Methods 21; data doi:10.5061/dryad.wstqjq2z0).

## Citation

G. Lv, R. Zheng, L. Gu, A decision-based criterion and a resolution threshold for ranking genomic prediction
models within trials (2026). Analysis code for the paper: https://github.com/nblvguohao/gp-decision-criterion

Forward evaluation: G. Lv, L. Gu, Genomic prediction for selection in a new year: a forward
benchmark across 48 target years of maize, wheat and soybean trials (2026), manuscript in preparation. Analysis
code for the paper: https://github.com/nblvguohao/gxe-forward-evaluation

Licence: MIT. The pooling functions of `gpverdict/forward.py` (`floor_se`, `dersimonian_laird`,
`hartung_knapp`) are copied from `src/dartgxe/forward/pool.py` of gxe-forward-evaluation (CC BY-NC 4.0 there;
`floor_se` and `dersimonian_laird` as in public commit d3c2e43, `hartung_knapp` as in public commit
be88a7e) and are relicensed under MIT by their copyright holders, G. Lv and L. Gu.

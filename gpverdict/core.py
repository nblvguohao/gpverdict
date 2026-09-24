"""Core computations of GPverdict (numpy and pandas only, so it also runs in a browser).

Input: one row per (environment, genotype, method) with the observed and the predicted
value, typically from cross-validation. Every statistic is computed within environments
and then averaged over environments, following Lv and Gu (2026):
  - a metric can rank methods for within-environment truncation selection only if it is
    unchanged by every strictly increasing transformation of the predictions within an
    environment and changes when their order is reversed (Tier I); Pearson r is unmoved by
    calibration (affine maps) only (Tier II); error-magnitude and pooled metrics are neither
    (Tier III);
  - the accuracy gap a trial of N genotype-environment cells resolves is at least
    k(f)/sqrt(N) in within-environment correlation (Gaussian design grid);
  - whether a rule picks methods that select better material is tested out of sample.
"""
import itertools
from statistics import NormalDist

import numpy as np
import pandas as pd

TIE = 1e-9
# Gaussian design grid (Lv and Gu 2026, Supplementary Table S11): Delta r* ~ k / sqrt(cells)
K_GAUSS = {0.05: 4.2883, 0.10: 2.9657, 0.20: 2.0927}
# the three calibrated panels of the paper needed 1.5-4.2 times the Gaussian value
PANEL_FACTOR = (1.5, 4.2)

METRICS = {
    # key: (label, tier, direction)   direction +1 higher is better, -1 lower is better, 0 closer to 1
    "spearman": ("mean within-environment rank correlation", "I", +1),
    "sel_diff": ("realised selection differential at f", "I", +1),
    "hit_rate": ("top-k hit rate (coincidence of selected sets)", "I", +1),
    "ndcg": ("NDCG at k", "I", +1),
    "pearson": ("mean within-environment Pearson r", "II", +1),
    "rmse": ("mean within-environment RMSE", "III", -1),
    "mae": ("mean within-environment MAE", "III", -1),
    "r2_score": ("mean within-environment R² score", "III", +1),
    "slope": ("mean calibration slope", "III", 0),
    "pooled_pearson": ("Pearson r pooled across environments", "III", +1),
    "pooled_rmse": ("RMSE pooled across environments", "III", -1),
}

ALIASES = {"env": "Env", "environment": "Env", "trial": "Env", "site": "Env",
           "genotype": "k", "k": "k", "gid": "k", "line": "k", "entry": "k", "hybrid": "k",
           "observed": "y", "y": "y", "obs": "y", "phenotype": "y",
           "predicted": "p", "p": "p", "pred": "p", "prediction": "p",
           "method": "method", "model": "method"}


def load(data):
    """Read a CSV path or DataFrame and return columns Env, k, y, p, method."""
    df = pd.read_csv(data) if isinstance(data, (str, bytes)) or hasattr(data, "read") else data.copy()
    ren = {c: ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in ALIASES}
    df = df.rename(columns=ren)
    miss = [c for c in ("Env", "k", "y", "p", "method") if c not in df.columns]
    if miss:
        raise ValueError("missing columns: " + ", ".join(miss) +
                         " (expected environment, genotype, observed, predicted, method)")
    df = df[["Env", "k", "y", "p", "method"]].dropna()
    for c in ("Env", "k", "method"):
        df[c] = df[c].astype(str)
    df["y"] = df.y.astype(float); df["p"] = df.p.astype(float)
    return df.reset_index(drop=True)


def n_select(n, frac):
    return max(1, int(round(frac * n)))


def _ranks(a):
    return pd.Series(a).rank(method="average").to_numpy()


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _ndcg(y, p, k):
    order = np.argsort(-p, kind="stable")[:k]
    g = y[order] - y.min()
    dcg = np.sum(g / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(y)[::-1][:k] - y.min()
    idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))
    return float(dcg / idcg) if idcg > 0 else np.nan


def env_metrics(y, p, frac):
    """All metrics of one prediction vector in one environment."""
    e = p - y; sy = y.std(); k = n_select(len(y), frac)
    out = dict(n=len(y), rmse=float(np.sqrt(np.mean(e * e))), mae=float(np.mean(np.abs(e))),
               r2_score=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)))
    if np.unique(p).size > 1:
        out["pearson"] = _corr(y, p)
        out["spearman"] = _corr(_ranks(y), _ranks(p))
        pc = p - p.mean()
        out["slope"] = float(np.sum(pc * (y - y.mean())) / np.sum(pc * pc))
        top = np.argsort(-p, kind="stable")[:k]
        out["sel_diff"] = float((y[top].mean() - y.mean()) / sy)
        best = np.argsort(-y, kind="stable")[:k]
        out["hit_rate"] = len(set(top.tolist()) & set(best.tolist())) / k
        out["ndcg"] = _ndcg(y, p, k)
    else:
        for c in ("pearson", "spearman", "slope", "sel_diff", "hit_rate", "ndcg"):
            out[c] = np.nan
    return out


def usable_envs(df, min_n):
    g = df.groupby(["method", "Env"])
    s = g.agg(n=("y", "size"), ny=("y", "nunique"), npred=("p", "nunique")).reset_index()
    return s[(s.n >= min_n) & (s.ny > 1) & (s.npred > 1)][["method", "Env"]]


def env_table(df, frac=0.10, min_n=10):
    """Long table: one row per (method, environment) with every within-environment metric."""
    ok = usable_envs(df, min_n)
    d = df.merge(ok, on=["method", "Env"])
    rows = []
    for (m, e), g in d.groupby(["method", "Env"], sort=True):
        rows.append(dict(method=m, Env=e, **env_metrics(g.y.to_numpy(), g.p.to_numpy(), frac)))
    return pd.DataFrame(rows)


def summarise(df, ET):
    """Methods x metrics: within-environment metrics averaged over environments, plus pooled."""
    S = ET.drop(columns=["Env", "n"]).groupby("method").mean()
    S["environments"] = ET.groupby("method").Env.nunique()
    S["cells"] = ET.groupby("method").n.sum()
    for m, g in df.groupby("method"):
        S.loc[m, "pooled_pearson"] = _corr(g.y.to_numpy(), g.p.to_numpy())
        S.loc[m, "pooled_rmse"] = float(np.sqrt(np.mean((g.p - g.y) ** 2)))
    return S


def oriented(S, key):
    v = S[key].to_numpy(float); d = METRICS[key][2]
    return v if d > 0 else (-v if d < 0 else -np.abs(v - 1.0))


def ranks(S, key):
    o = np.argsort(-oriented(S, key), kind="stable"); r = np.empty(len(S), int); r[o] = np.arange(1, len(S) + 1)
    return pd.Series(r, index=S.index)


def reversal(S, a="rmse", b="pearson"):
    """Share of method pairs that two metrics order in opposite directions (ties excluded)."""
    x, y = oriented(S, a), oriented(S, b); rev = tot = 0
    for i, j in itertools.combinations(range(len(S)), 2):
        d1, d2 = x[i] - x[j], y[i] - y[j]
        if not (np.isfinite(d1) and np.isfinite(d2)) or abs(d1) < TIE or abs(d2) < TIE:
            continue
        tot += 1; rev += d1 * d2 < 0
    return (rev / tot if tot else np.nan), tot


def _transform_env(df, kind, rng):
    out = df.copy()
    for e, idx in out.groupby("Env").groups.items():
        p = out.loc[idx, "p"].to_numpy(float); z = (p - p.mean()) / (p.std() or 1.0)
        if kind == "affine":
            q = rng.uniform(0.5, 2.0) * p + rng.normal(0, 1) * (p.std() or 1.0)
        elif kind == "monotone":
            c = rng.uniform(0.3, 1.5); q = np.exp(c * z) * rng.uniform(0.5, 2.0) + rng.normal(0, 1)
        else:
            q = -p
        out.loc[idx, "p"] = q
    return out


def invariance_check(df, frac=0.10, min_n=10, methods=None, reps=3, seed=0):
    """Median relative change of each metric under decision-preserving transformations of the
    predictions within environments (affine; strictly increasing) and under order reversal.
    Returns the empirical tier of every metric on the user's own predictions."""
    rng = np.random.default_rng(seed)
    meths = methods or sorted(df.method.unique())[:5]
    base = df[df.method.isin(meths)]
    S0 = summarise(base, env_table(base, frac, min_n))
    keys = list(METRICS)
    def drift(kind, n):
        ch = []
        for _ in range(n):
            T = _transform_env(base, kind, rng)
            S1 = summarise(T, env_table(T, frac, min_n))
            ch.append((S1[keys] - S0[keys]).abs().to_numpy() / np.maximum(S0[keys].abs().to_numpy(), 1e-12))
        return np.nanmedian(np.vstack(ch), axis=0)
    aff, mono, rev = drift("affine", reps), drift("monotone", reps), drift("reverse", 1)
    tier = []
    for a, m, r in zip(aff, mono, rev):
        tier.append("I" if (m < 1e-6 and r > 1e-3) else ("II" if (a < 1e-6 and r > 1e-3) else "III"))
    return pd.DataFrame(dict(metric=[METRICS[k][0] for k in keys], key=keys, affine_drift=aff,
                             monotone_drift=mono, reversal_change=rev, tier=tier))


def rank_intervals(ET, key="spearman", B=400, seed=0):
    """95 % intervals of each method's rank under `key`, resampling environments."""
    W = ET.pivot_table(index="Env", columns="method", values=key)
    rng = np.random.default_rng(seed); M = W.shape[1]; d = METRICS[key][2]
    R = np.empty((B, M))
    for b in range(B):
        mu = W.iloc[rng.integers(0, len(W), len(W))].mean().to_numpy()
        v = mu if d > 0 else (-mu if d < 0 else -np.abs(mu - 1))
        o = np.argsort(-v, kind="stable"); r = np.empty(M); r[o] = np.arange(1, M + 1); R[b] = r
    lo, hi = np.percentile(R, [2.5, 97.5], axis=0)
    return pd.DataFrame(dict(rank_lo=lo.round().astype(int), rank_hi=hi.round().astype(int)), index=W.columns)


def k_gauss(frac):
    """Constant of the Gaussian cells law, interpolated log-log between f = 0.05, 0.10, 0.20."""
    fs = np.array(sorted(K_GAUSS)); ks = np.array([K_GAUSS[f] for f in fs])
    return float(np.exp(np.interp(np.log(frac), np.log(fs), np.log(ks))))


def resolvable_gap(cells, frac=0.10):
    """Smallest accuracy gap a trial of `cells` genotype-environment cells resolves (Gaussian bound)."""
    return k_gauss(frac) / np.sqrt(cells)


def cells_needed(gap, frac=0.10):
    """Genotype-environment cells needed to resolve an accuracy gap (Gaussian bound)."""
    return (k_gauss(frac) / gap) ** 2


def selection_intensity(frac):
    nd = NormalDist(); x = nd.inv_cdf(1 - frac)
    return nd.pdf(x) / frac


def outcome_test(df, frac=0.10, min_n=10, B=200, seed=0, rules=("spearman", "pearson", "rmse")):
    """Out-of-sample check that a rule picks methods that select better material.

    With eight or more environments the environments are split in halves; with fewer,
    genotypes are split within each environment. Each rule ranks the methods on half A;
    the realised selection differential of its first choice is measured on half B.
    Recovery = (rule - average method) / (best method on half B - average method)."""
    rng = np.random.default_rng(seed)
    ET = env_table(df, frac, min_n); meths = sorted(ET.method.unique()); envs = sorted(ET.Env.unique())
    by_env = env_split = len(envs) >= 8
    res = {r: [] for r in rules}; orc, avg, rel = [], [], []
    if by_env:
        W = {k: ET.pivot_table(index="Env", columns="method", values=k).reindex(columns=meths) for k in set(rules) | {"sel_diff"}}
        for _ in range(B):
            e = rng.permutation(envs); A, Bh = e[:len(e) // 2], e[len(e) // 2:]
            gB = W["sel_diff"].loc[Bh].mean().to_numpy()
            gA = W["sel_diff"].loc[A].mean().to_numpy()
            for r in rules:
                sA = W[r].loc[A].mean().to_numpy(); d = METRICS[r][2]
                v = sA if d > 0 else -sA
                top = np.flatnonzero(np.abs(v - np.nanmax(v)) <= TIE * max(1.0, abs(np.nanmax(v))))
                res[r].append(np.nanmean(gB[top]))
            orc.append(np.nanmax(gB)); avg.append(np.nanmean(gB))
            ok = np.isfinite(gA) & np.isfinite(gB)
            if ok.sum() > 5: rel.append(_corr(_ranks(gA[ok]), _ranks(gB[ok])))
    else:
        d = df.merge(usable_envs(df, 2 * min_n), on=["method", "Env"])
        groups = {(m, e): (g.k.to_numpy(), g.y.to_numpy(), g.p.to_numpy()) for (m, e), g in d.groupby(["method", "Env"])}
        genos = {e: np.array(sorted(d[d.Env == e].k.unique())) for e in envs}
        for _ in range(B):
            half = {e: set(rng.permutation(genos[e])[:len(genos[e]) // 2].tolist()) for e in envs}
            scoreA = {r: {} for r in rules}; gA = {}; gB = {}
            for m in meths:
                accA = {r: [] for r in rules}; ga = []; gb = []
                for e in envs:
                    if (m, e) not in groups: continue
                    k_, y, p = groups[(m, e)]; inA = np.array([x in half[e] for x in k_])
                    ma = env_metrics(y[inA], p[inA], frac); mb = env_metrics(y[~inA], p[~inA], frac)
                    for r in rules: accA[r].append(ma[r])
                    ga.append(ma["sel_diff"]); gb.append(mb["sel_diff"])
                for r in rules: scoreA[r][m] = np.nanmean(accA[r])
                gA[m] = np.nanmean(ga); gB[m] = np.nanmean(gb)
            gBv = np.array([gB[m] for m in meths]); gAv = np.array([gA[m] for m in meths])
            for r in rules:
                v = np.array([scoreA[r][m] for m in meths]) * (1 if METRICS[r][2] > 0 else -1)
                top = np.flatnonzero(np.abs(v - np.nanmax(v)) <= TIE * max(1.0, abs(np.nanmax(v))))
                res[r].append(np.nanmean(gBv[top]))
            orc.append(np.nanmax(gBv)); avg.append(np.nanmean(gBv))
            ok = np.isfinite(gAv) & np.isfinite(gBv)
            if ok.sum() > 5: rel.append(_corr(_ranks(gAv[ok]), _ranks(gBv[ok])))
    o, a = np.mean(orc), np.mean(avg)
    rec = {r: float((np.mean(v) - a) / (o - a)) if o > a else np.nan for r, v in res.items()}
    raw = {r: float(np.mean(v)) for r, v in res.items()}
    return dict(design="environment split" if by_env else "genotype split within environments",
                splits=B, recovery=rec, realised=raw, best=float(o), average=float(a),
                reliability=float(np.mean(rel)) if rel else np.nan)


def verdict(data, frac=0.10, min_n=10, B=400, seed=0, check_invariance=True, run_outcome=True):
    """Everything the report needs, as plain Python objects."""
    df = load(data)
    ET = env_table(df, frac, min_n)
    if ET.empty:
        raise ValueError("no environment has at least %d genotypes with varying predictions" % min_n)
    df = df.merge(ET[["method", "Env"]].drop_duplicates(), on=["method", "Env"])
    S = summarise(df, ET)
    R = pd.DataFrame({k: ranks(S, k) for k in ("spearman", "pearson", "rmse", "sel_diff")})
    RI = rank_intervals(ET, "spearman", B, seed) if ET.Env.nunique() >= 8 else None
    cells = float(S.cells.median()); gap = resolvable_gap(cells, frac)
    order = S.spearman.sort_values(ascending=False)
    lead = order.index[0]
    tied = [m for m in order.index if order[lead] - order[m] <= gap]
    margin = float(order.iloc[0] - order.iloc[1]) if len(order) > 1 else np.nan
    rmse_pick = S.rmse.idxmin(); pearson_pick = S.pearson.idxmax()
    rev, npairs = reversal(S, "rmse", "pearson")
    out = dict(
        settings=dict(frac=frac, min_genotypes=min_n, bootstrap=B, seed=seed),
        data=dict(methods=int(S.shape[0]), environments=int(ET.Env.nunique()),
                  genotypes=int(df.k.nunique()), cells=int(cells)),
        summary=S, ranks=R, rank_intervals=RI,
        leader=lead, rmse_pick=rmse_pick, pearson_pick=pearson_pick,
        reversal_rmse_pearson=rev, method_pairs=npairs,
        resolution=dict(gap=float(gap), panel_range=(float(gap * PANEL_FACTOR[0]), float(gap * PANEL_FACTOR[1])),
                        leader_margin=margin, tied_with_leader=tied,
                        cells_to_resolve_margin=float(cells_needed(margin, frac)) if margin and margin > 0 else np.inf),
        intensity=selection_intensity(frac),
    )
    out["projection_gap"] = float((out["intensity"] * S.pearson - S.sel_diff).mean())
    out["invariance"] = invariance_check(df, frac, min_n) if check_invariance else None
    out["outcome"] = outcome_test(df, frac, min_n, B=min(B, 200), seed=seed) if run_outcome and S.shape[0] >= 3 else None
    return out

"""Forward evaluation: compare prediction methods with a reference method over target years, within environments.

This module implements the scoring of Lv, Zheng and Gu (2026), "Genomic prediction for selection in a new year: a
forward benchmark across 48 target years of maize, wheat and soybean trials":

1. In every environment of every target year, the Spearman correlation between observed and predicted values is
   computed for each method on the cells where the method and the reference both have a prediction.
2. Method minus reference, averaged over the environments of a target year, is the year effect; its standard error
   comes from an environment bootstrap.
3. Year effects are pooled over target years by DerSimonian-Laird random effects after raising each year's standard
   error to the median of the positive standard errors of its dataset (the "SE floor"); a Hartung-Knapp interval is
   reported alongside.
4. A difference is "resolved" if it is at least 3/sqrt(N), N the number of scored genotype-environment cells, and its
   interval excludes zero; "detectable, below resolution" if only the interval condition holds; "tied" otherwise.

Provenance and licence. floor_se, dersimonian_laird and hartung_knapp are copied from src/dartgxe/forward/pool.py of
the analysis code of that paper (https://github.com/nblvguohao/gxe-forward-evaluation, released there under CC BY-NC
4.0): floor_se and dersimonian_laird as in public commit d3c2e43, hartung_knapp as in public commit be88a7e
(unchanged from the authors' analysis commit 7ba0319 used for GPverdict 1.2.0). The copyright holders of that code (G. Lv and L. Gu) relicense
these functions under the MIT licence for GPverdict. The only change is that normal and t quantiles come from the
Python standard library instead of scipy (agreement with scipy.stats.t.ppf within 3e-10), so the module runs in a
browser with numpy and pandas only. tests/test_forward.py checks the module against the year-effect tables and
pooled results archived with that paper (tests/data)."""
import math
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

from .core import _corr, _ranks

ALIASES = {"env": "environment", "environment": "environment", "trial": "environment", "site": "environment",
           "year": "year", "target": "year", "target_year": "year", "season": "year",
           "genotype": "genotype", "k": "genotype", "gid": "genotype", "line": "genotype", "entry": "genotype", "hybrid": "genotype",
           "observed": "observed", "y": "observed", "obs": "observed", "phenotype": "observed",
           "predicted": "predicted", "p": "predicted", "pred": "predicted", "prediction": "predicted",
           "method": "method", "model": "method", "dataset": "dataset", "new": "new", "new_line": "new", "first_year": "first_year"}
VERDICTS = ("resolved gain", "resolved loss", "detectable, below resolution", "tied")


# ------------------------------------------------------------------ quantiles (standard library only)
def norm_ppf(q):
    return NormalDist().inv_cdf(q)


def _betacf(a, b, x):
    """Continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        de = d * c; h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1 - p if t > 0 else p


def t_ppf(q, df):
    """Quantile of Student's t by bisection on t_cdf (agrees with scipy.stats.t.ppf to about 1e-12)."""
    if q == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q:
        lo *= 2
    while t_cdf(hi, df) < q:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, abs(mid)):
            break
    return (lo + hi) / 2


# ------------------------------------------------------------------ pooling (from gxe-forward-evaluation, pool.py; MIT, see module docstring)
def floor_se(ye: pd.DataFrame) -> pd.DataFrame:
    ye = ye.copy()
    pos_all = ye.loc[ye["se"] > 0, "se"]
    ye["se_zero"] = ye["se"] <= 0
    for _, g in ye.groupby("dataset"):
        pos = g.loc[g["se"] > 0, "se"]
        fl = pos.median() if len(pos) else (pos_all.median() if len(pos_all) else np.nan)
        ye.loc[g.index, "se"] = np.maximum(g["se"], fl)
    return ye


def dersimonian_laird(ye: pd.DataFrame, level: float = 0.95) -> dict:
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    w = 1 / v
    mu = (w * d).sum() / w.sum()
    Q = (w * (d - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (len(d) - 1)) / (w.sum() - (w * w).sum() / w.sum())) if len(d) > 1 else 0.0
    ws = 1 / (v + tau2)
    est, se = (ws * d).sum() / ws.sum(), 1 / np.sqrt(ws.sum())
    z = norm_ppf(1 - (1 - level) / 2)
    return {"est": float(est), "se": float(se), "lo": float(est - z * se), "hi": float(est + z * se), "tau2": float(tau2),
            "k": int(len(d)), "n_years_se_zero_floored": int(ye["se_zero"].sum()) if "se_zero" in ye else 0}


def hartung_knapp(ye: pd.DataFrame, level: float = 0.95) -> dict:
    r = dersimonian_laird(ye, level=level)
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    k = len(d)
    if k < 2:
        return {**r, "hk_lo": float("nan"), "hk_hi": float("nan"), "hk_se": float("nan")}
    w = 1 / (v + r["tau2"])
    se = float(np.sqrt((w * (d - r["est"]) ** 2).sum() / ((k - 1) * w.sum())))
    q = float(t_ppf(1 - (1 - level) / 2, k - 1))
    return {**r, "hk_se": se, "hk_lo": r["est"] - q * se, "hk_hi": r["est"] + q * se}


def verdict_of(est, lo, hi, resolution):
    if lo > 0 and est >= resolution:
        return "resolved gain"
    if hi < 0 and -est >= resolution:
        return "resolved loss"
    if lo > 0 or hi < 0:
        return "detectable, below resolution"
    return "tied"


# ------------------------------------------------------------------ input
def load_forward(data):
    """Long table, one row per (environment, genotype, method): columns environment, year, genotype, observed, predicted,
    method; optional dataset, new (1 = new line) or first_year (first year in which the line was phenotyped)."""
    df = pd.read_csv(data) if isinstance(data, (str, bytes)) or hasattr(data, "read") else data.copy()
    df = df.rename(columns={c: ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in ALIASES})
    need = ["environment", "year", "genotype", "observed", "predicted", "method"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError("missing column(s): " + ", ".join(miss) + ". Expected one row per environment, genotype and method with "
                         "columns environment, year, genotype, observed, predicted, method (optional: dataset, new, first_year).")
    if "dataset" not in df.columns:
        df["dataset"] = "all"
    if "new" not in df.columns and "first_year" in df.columns:
        df["new"] = (pd.to_numeric(df["first_year"]) >= pd.to_numeric(df["year"])).astype(int)
    keep = need + ["dataset"] + (["new"] if "new" in df.columns else [])
    df = df[keep].dropna(subset=need)
    for c in ("environment", "genotype", "method", "dataset"):
        df[c] = df[c].astype(str)
    df["year"] = pd.to_numeric(df["year"]).astype(int)
    df["observed"] = df["observed"].astype(float); df["predicted"] = df["predicted"].astype(float)
    return df.reset_index(drop=True)


def _spearman(y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return _corr(_ranks(p), _ranks(y))


# ------------------------------------------------------------------ scoring
def env_differences(df, reference, min_genotypes=25, scope="all"):
    """Per (dataset, year, environment, method): Spearman of method and reference on their common cells, the
    difference, and the number of cells. scope: 'all', 'new' or 'old' lines (needs the column new)."""
    if scope != "all":
        if "new" not in df.columns:
            raise ValueError("scope 'new' or 'old' needs a column new (or first_year)")
        df = df[df["new"].astype(int) == (1 if scope == "new" else 0)]
    W = df.pivot_table(index=["dataset", "year", "environment", "genotype"], columns="method", values="predicted", aggfunc="first")
    y = df.drop_duplicates(["dataset", "year", "environment", "genotype"]).set_index(["dataset", "year", "environment", "genotype"])["observed"]
    if reference not in W.columns:
        raise ValueError(f"reference method {reference!r} not found; methods: {sorted(W.columns)}")
    rows = []
    for (d, Y, e), g in W.groupby(level=[0, 1, 2]):
        yy_all = y.reindex(g.index)
        for m in W.columns.drop(reference):
            ok = g[[m, reference]].dropna()
            if len(ok) < max(3, min_genotypes):
                continue
            yy = yy_all.reindex(ok.index).to_numpy(float)
            sa, sb = _spearman(yy, ok[m].to_numpy(float)), _spearman(yy, ok[reference].to_numpy(float))
            rows.append({"dataset": d, "year": int(Y), "environment": e, "method": m, "n": len(ok), "spearman": sa,
                         "spearman_reference": sb, "d": sa - sb})
    return pd.DataFrame(rows)


def year_effects(E, B=2000, seed=0):
    """Mean environment difference per (dataset, year, method) with an environment-bootstrap standard error."""
    rng = np.random.default_rng(seed)
    rows = []
    for (d, Y, m), g in E.groupby(["dataset", "year", "method"]):
        v = g["d"].dropna().to_numpy()
        if not len(v):
            continue
        se = float(v[rng.integers(0, len(v), (B, len(v)))].mean(1).std(ddof=1)) if len(v) > 1 else 0.0
        rows.append({"dataset": d, "year": int(Y), "method": m, "d": float(v.mean()), "se": se if se > 1e-12 else 0.0,
                     "n_env": len(v), "cells": int(g.loc[g["d"].notna(), "n"].sum())})
    return pd.DataFrame(rows)


def pool(YE, level=0.95, by_dataset=True):
    """Pool year effects per method over all datasets and within each dataset; N is the number of scored cells (column
    cells of YE). Returns one row per (method, range)."""
    out = []
    for m, g in YE.groupby("method"):
        ranges = [("all", g)] + ([(d, gd) for d, gd in g.groupby("dataset")] if by_dataset and g["dataset"].nunique() > 1 else [])
        for rg, sub in ranges:
            h = hartung_knapp(floor_se(sub[["dataset", "d", "se"]].reset_index(drop=True)), level=level)
            N = float(sub["cells"].sum()) if "cells" in sub else np.nan
            res = 3 / math.sqrt(N) if N and N > 0 else np.nan
            out.append({"method": m, "range": rg, "target_years": h["k"], "cells": int(N) if N == N else None, "resolution": res,
                        "est": h["est"], "lo": h["lo"], "hi": h["hi"], "tau2": h["tau2"], "hk_lo": h["hk_lo"], "hk_hi": h["hk_hi"],
                        "verdict": verdict_of(h["est"], h["lo"], h["hi"], res),
                        "verdict_hk": verdict_of(h["est"], h["hk_lo"], h["hk_hi"], res) if h["k"] > 1 else "n/a (one year)"})
    return pd.DataFrame(out)


def forward_verdict(data, reference, min_genotypes=25, B=2000, seed=0, level=0.95):
    """Full forward evaluation: environment differences, year effects and pooled verdicts for all lines and, when the
    input marks new lines, for new and old lines separately."""
    df = load_forward(data)
    res = {"settings": {"reference": reference, "min_genotypes": min_genotypes, "bootstrap": B, "seed": seed, "level": level},
           "data": {"rows": len(df), "datasets": sorted(df["dataset"].unique()), "years": sorted(int(y) for y in df["year"].unique()),
                    "methods": sorted(df["method"].unique()), "has_new": "new" in df.columns}}
    scopes = ["all"] + (["new", "old"] if "new" in df.columns else [])
    for sc in scopes:
        E = env_differences(df, reference, min_genotypes, sc)
        if E.empty:
            continue
        YE = year_effects(E, B, seed)
        res[sc] = {"environments": E, "year_effects": YE, "pooled": pool(YE, level=level)}
    return res


def pool_year_effects(YE, resolution=None, cells=None, level=0.95):
    """Second entry point: pool a table of year effects (columns dataset, d, se; optional method) without per-cell
    data, e.g. the year-effect tables archived with a paper. Give either the resolution or the number of scored cells."""
    YE = YE.copy()
    if "method" not in YE.columns:
        YE["method"] = "contrast"
    if resolution is None and cells is not None:
        resolution = 3 / math.sqrt(cells)
    out = []
    for m, g in YE.groupby("method"):
        h = hartung_knapp(floor_se(g[["dataset", "d", "se"]].reset_index(drop=True)), level=level)
        out.append({"method": m, "target_years": h["k"], "resolution": resolution, "est": h["est"], "lo": h["lo"], "hi": h["hi"],
                    "tau2": h["tau2"], "hk_lo": h["hk_lo"], "hk_hi": h["hk_hi"],
                    "verdict": verdict_of(h["est"], h["lo"], h["hi"], resolution) if resolution else None,
                    "verdict_hk": (verdict_of(h["est"], h["hk_lo"], h["hk_hi"], resolution) if resolution and h["k"] > 1 else None)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ report
def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{d}f}".replace("-", "−")


def forward_markdown(v):
    s = v["settings"]
    lines = [f"# Forward evaluation against `{s['reference']}`", "",
             f"Within-environment Spearman correlation, method minus reference, pooled over {len(v['data']['years'])} target year(s) "
             f"and {len(v['data']['datasets'])} dataset(s) (DerSimonian–Laird with the SE floor; Hartung–Knapp alongside). "
             f"A difference is resolved if it is at least 3/√N and its {100 * s['level']:.0f} % interval excludes zero.", ""]
    for sc, title in (("all", "All lines"), ("new", "New lines"), ("old", "Old lines")):
        if sc not in v:
            continue
        P = v[sc]["pooled"]
        lines += [f"## {title}", "", "| Method | Range | Years | Cells | 3/√N | Δ [interval] | Hartung–Knapp | Verdict |", "|---|---|---|---|---|---|---|---|"]
        for r in P.sort_values(["range", "est"], ascending=[True, False]).itertuples():
            lines.append(f"| {r.method} | {r.range} | {r.target_years} | {r.cells:,} | {r.resolution:.4f} | {_f(r.est, 4)} [{_f(r.lo, 4)}, {_f(r.hi, 4)}] | "
                         f"[{_f(r.hk_lo, 4)}, {_f(r.hk_hi, 4)}] | {r.verdict} |")
        lines.append("")
    return "\n".join(lines)


def _forest_svg(P, width=620):
    P = P[P["range"] == "all"].sort_values("est")
    if P.empty:
        return ""
    lo = min(P["lo"].min(), P["hk_lo"].min(skipna=True) if P["hk_lo"].notna().any() else 0, -P["resolution"].max(), 0)
    hi = max(P["hi"].max(), P["hk_hi"].max(skipna=True) if P["hk_hi"].notna().any() else 0, P["resolution"].max(), 0)
    pad = 0.05 * (hi - lo or 1); lo -= pad; hi += pad
    left, right, rowh = 180, 20, 22
    h = rowh * len(P) + 40
    X = lambda x: left + (x - lo) / (hi - lo) * (width - left - right)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" font-family="Arial" font-size="11">']
    for i, r in enumerate(P.itertuples()):
        y = 15 + i * rowh
        out.append(f'<rect x="{X(-r.resolution):.1f}" y="{y - 7}" width="{X(r.resolution) - X(-r.resolution):.1f}" height="14" fill="#e5e5e5"/>')
        if math.isfinite(r.hk_lo):
            out.append(f'<line x1="{X(r.hk_lo):.1f}" x2="{X(r.hk_hi):.1f}" y1="{y + 5}" y2="{y + 5}" stroke="#E69F00" stroke-width="1.5"/>')
        col = "#0072B2" if r.verdict == "resolved gain" else ("#D55E00" if r.verdict == "resolved loss" else "#7f7f7f")
        out.append(f'<line x1="{X(r.lo):.1f}" x2="{X(r.hi):.1f}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="1.5"/>')
        out.append(f'<circle cx="{X(r.est):.1f}" cy="{y}" r="3.5" fill="{col}"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4}" text-anchor="end">{r.method}</text>')
    out.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="5" y2="{h - 25}" stroke="#222"/>')
    for t in np.linspace(lo, hi, 5):
        out.append(f'<text x="{X(t):.1f}" y="{h - 8}" text-anchor="middle">{t:+.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def forward_html(v):
    import html as _h
    md = forward_markdown(v)
    body = []
    for ln in md.split("\n"):
        if ln.startswith("# "):
            body.append("<h1>" + re.sub(r"`([^`]*)`", r"<code>\1</code>", _h.escape(ln[2:])) + "</h1>")
        elif ln.startswith("## "):
            body.append(f"<h2>{_h.escape(ln[3:])}</h2>")
            sc = {"All lines": "all", "New lines": "new", "Old lines": "old"}[ln[3:]]
            body.append(_forest_svg(v[sc]["pooled"]))
        elif ln.startswith("|---"):
            continue
        elif ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            tag = "th" if cells[0] == "Method" else "td"
            body.append("<tr>" + "".join(f"<{tag}>{_h.escape(c).replace('`', '')}</{tag}>" for c in cells) + "</tr>")
        elif ln:
            body.append("<p>" + re.sub(r"`([^`]*)`", r"<code>\1</code>", _h.escape(ln)) + "</p>")
    html_ = "\n".join(body)
    html_ = html_.replace("<tr><th>", "<table><tr><th>").replace("</td></tr>\n<h2>", "</td></tr></table>\n<h2>")
    if html_.count("<table>") > html_.count("</table>"):
        html_ += "</table>"
    return ("<!doctype html><meta charset='utf-8'><title>GPverdict forward evaluation</title>"
            "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;font-size:13px}"
            "td,th{border:1px solid #ccc;padding:3px 6px}</style>" + html_)


def forward_json(v):
    import json
    out = {"settings": v["settings"], "data": v["data"]}
    for sc in ("all", "new", "old"):
        if sc in v:
            out[sc] = {"pooled": v[sc]["pooled"].to_dict("records"), "year_effects": v[sc]["year_effects"].to_dict("records")}
    return json.dumps(out, indent=1, default=float)

"""Rolling-origin cross-validation, accuracy measures and forecast comparison tests."""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import style
from .models import COLS, ensemble


def origins(index, h, folds, min_train):
    n = len(index)
    starts = [n - h * (folds - i) for i in range(folds)]
    starts = [s for s in starts if s >= min_train]
    return [index[s - 1] for s in starts]  # last training date of each fold


def run_cv(ctx, models: dict, h: int, folds: int, min_train: int, log=print):
    """Expanding-window evaluation. Returns long DataFrame of forecasts vs actuals."""
    y = ctx.p.y
    rows = []
    timings = {m: 0.0 for m in models}
    for k, origin in enumerate(origins(y.index, h, folds, min_train)):
        fut = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
        fcs = {}
        for name, fn in models.items():
            t0 = time.time()
            try:
                fcs[name] = fn(ctx, origin, fut)
            except Exception as exc:  # a model failing in one fold must not kill the run
                log(f"   fold {k + 1}: {name} failed ({exc.__class__.__name__}: {exc})")
            timings[name] += time.time() - t0
        if any(m in fcs for m in ("SARIMA", "ETS", "Gradient boosting")):
            fcs["Ensemble"] = ensemble(fcs)
        for name, fc in fcs.items():
            df = fc[COLS].copy()
            df["actual"] = y.reindex(fut).to_numpy()
            df["model"] = name
            df["fold"] = k + 1
            df["origin"] = origin
            df["step"] = np.arange(1, h + 1)
            df["scale"] = _mase_scale(ctx, origin)
            rows.append(df)
        log(f"   fold {k + 1}: origin {origin.date()}, {len(fcs)} models")
    cv = pd.concat(rows).rename_axis("date").reset_index()
    # Irregular, non-holiday closures cannot be predicted by any model: excluded from scoring.
    cv["evaluated"] = ~cv["date"].map(ctx.closed_nh).fillna(0).astype(bool).to_numpy()
    return cv, timings


def _mase_scale(ctx, origin):
    y = ctx.p.y_clean.loc[:origin]
    return float((y - y.shift(ctx.s)).abs().mean())


def metrics(df: pd.DataFrame) -> dict:
    d = df[df.evaluated]
    e = d["actual"] - d["mean"]
    pos = d["actual"] > 0
    pe = (e[pos] / d.loc[pos, "actual"]) * 100
    out = {
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MASE": (e.abs() / d["scale"]).mean(),
        "WAPE %": e.abs().sum() / d["actual"].abs().sum() * 100,
        "MAPE %": pe.abs().mean(),
        "RMSPE %": np.sqrt((pe ** 2).mean()),
        "Bias": -e.mean(),
    }
    if d["lo95"].notna().all():
        out["Cov80 %"] = ((d.actual >= d.lo80) & (d.actual <= d.hi80)).mean() * 100
        out["Cov95 %"] = ((d.actual >= d.lo95) & (d.actual <= d.hi95)).mean() * 100
        o = d[pos]
        out["Width95"] = (o.hi95 - o.lo95).mean()
    else:
        out["Cov80 %"] = out["Cov95 %"] = out["Width95"] = np.nan
    return out


def leaderboard(cv: pd.DataFrame) -> pd.DataFrame:
    tab = pd.DataFrame({m: metrics(g) for m, g in cv.groupby("model")}).T
    per_fold = cv[cv.evaluated].assign(ae=lambda d: (d.actual - d["mean"]).abs() / d.scale)
    sd = per_fold.groupby(["model", "fold"]).ae.mean().groupby("model").std()
    tab["MASE sd (folds)"] = sd
    return tab.sort_values("MASE")


def diebold_mariano(cv, model_a, model_b, h):
    """DM test (squared loss) with Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic: model_a is more accurate."""
    a = cv[(cv.model == model_a) & cv.evaluated].set_index(["fold", "step"])
    b = cv[(cv.model == model_b) & cv.evaluated].set_index(["fold", "step"])
    j = a.index.intersection(b.index)
    d = ((a.loc[j, "actual"] - a.loc[j, "mean"]) ** 2 - (b.loc[j, "actual"] - b.loc[j, "mean"]) ** 2)
    d = d.sort_index().to_numpy()
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean()
    lag = min(h - 1, T // 4)
    gam = [np.mean((d[k:] - dbar) * (d[:T - k] - dbar)) for k in range(lag + 1)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    if var <= 0:  # HAC estimate can be negative in small samples: fall back to iid variance
        var = gam[0] / T
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T) if T > 2 * h else 1.0
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)


def empirical_intervals(cv, model, h):
    """CV-calibrated interval offsets per horizon step (open days only)."""
    d = cv[(cv.model == model) & cv.evaluated & (cv.actual > 0)]
    e = d["actual"] - d["mean"]
    probs = [0.025, 0.1, 0.9, 0.975]
    pooled = e.quantile(probs)
    # Errors are pooled by forecast week (steps 1-7, 8-14, ...) so every quantile rests on
    # many errors; a week with fewer than 30 errors uses the quantiles of all steps.
    week = (d["step"] - 1) // 7
    rows = {}
    for w, g in e.groupby(week):
        rows[w] = g.quantile(probs) if len(g) >= 30 else pooled
    q = pd.DataFrame({s: rows.get((s - 1) // 7, pooled) for s in range(1, h + 1)}).T
    q.columns = probs
    return q


# --------------------------------------------------------------------------- figures
def fig_error_by_step(cv, path, models):
    d = cv[cv.evaluated & cv.model.isin(models)].copy()
    d["ae"] = (d.actual - d["mean"]).abs() / d.scale
    g = d.groupby(["model", "step"]).ae.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in models:
        if m in g:
            sm = g[m].rolling(7, center=True, min_periods=1).mean()
            ax.plot(sm.index, sm, color=style.MODEL_COLORS.get(m, style.TEXT_2), lw=2, label=m)
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel("Scaled absolute error (MASE units)")
    ax.set_title("Accuracy by horizon (7-day smoothed, all CV folds)")
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    style.save(fig, path)


def fig_last_fold(cv, ctx, path, models):
    last = cv.fold.max()
    d = cv[cv.fold == last]
    y = ctx.p.y
    start = d.date.min() - pd.Timedelta(days=42)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    hist = y.loc[start:d.date.max()]
    ax.plot(hist.index, hist, color=style.MUTED, lw=1.2, label="Actual")
    for m in models:
        g = d[d.model == m]
        if len(g):
            ax.plot(g.date, g["mean"], color=style.MODEL_COLORS.get(m, style.TEXT_2), lw=2, label=m)
    ax.axvline(d.date.min() - pd.Timedelta(hours=12), color=style.TEXT_2, lw=0.8, ls="--")
    ax.set_title(f"Last CV fold: forecasts from {d.origin.iloc[0].date()} vs actual")
    ax.set_ylabel("Income")
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    style.save(fig, path)

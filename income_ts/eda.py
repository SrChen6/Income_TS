"""Exploratory analysis and Box-Jenkins identification."""

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from . import style
from .data import WEEKDAYS, Prepared

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_dates(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


# --------------------------------------------------------------------------- summary
def summary(p: Prepared) -> dict:
    y, open_ = p.y, p.open_mask
    yo = y[open_]
    out = {
        "mean_all": float(y.mean()),
        "mean_open": float(yo.mean()),
        "median_open": float(yo.median()),
        "sd_open": float(yo.std()),
        "cv_open": float(yo.std() / yo.mean()) if yo.mean() else np.nan,
        "min_open": float(yo.min()),
        "max_open": float(yo.max()),
        "max_date": yo.idxmax().date().isoformat(),
        "total": float(y.sum()),
        "skew_open": float(stats.skew(yo)),
        "kurt_open": float(stats.kurtosis(yo)),
    }
    # Trend of monthly mean (OLS with HAC errors) expressed per year.
    m = y.resample("ME").mean()
    if len(m) >= 6:
        import statsmodels.api as sm
        X = sm.add_constant(np.arange(len(m)) / 12.0)
        fit = sm.OLS(m.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        out["trend_per_year"] = float(fit.params[1])
        out["trend_per_year_pct"] = float(fit.params[1] / m.mean() * 100)
        out["trend_p"] = float(fit.pvalues[1])
    # Yearly table and last-12-months growth.
    yr = y.groupby(y.index.year).agg(["sum", "mean", "count"])
    out["yearly"] = [
        {"year": int(k), "total": float(r["sum"]), "mean_daily": float(r["mean"]),
         "days": int(r["count"]), "complete": bool(r["count"] >= 365)}
        for k, r in yr.iterrows()
    ]
    if len(y) >= 730:
        last, prev = y.iloc[-365:].sum(), y.iloc[-730:-365].sum()
        out["growth_last12"] = float((last / prev - 1) * 100) if prev else np.nan
    return out


# --------------------------------------------------------------------------- figures
def fig_overview(p: Prepared, path):
    y = p.y
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(y.index, y, color=style.MUTED, lw=0.7, label="Daily income")
    win = 28 if len(y) > 120 else 7
    ax.plot(y.index, y.rolling(win, center=True, min_periods=win // 2).mean(),
            color=style.BLUE, lw=2, label=f"{win}-day centred mean")
    cl = p.closed[p.closed == 1].index
    if len(cl):
        ax.scatter(cl, np.zeros(len(cl)), s=14, color=style.RED, zorder=3,
                   label="Irregular closure (0 income)")
    if p.imputed.any():
        im = p.imputed[p.imputed].index
        ax.scatter(im, y[im], s=12, facecolor="none", edgecolor=style.ORANGE, zorder=3,
                   label="Imputed (missing day)")
    ax.set_title("Daily store income")
    ax.set_ylabel("Income")
    _fmt_dates(ax)
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    style.save(fig, path)


def fig_rolling(p: Prepared, path, window=28):
    yo = p.y.where(p.open_mask)
    rm = yo.rolling(window, min_periods=window // 2).mean()
    rs = yo.rolling(window, min_periods=window // 2).std()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    a1.plot(rm.index, rm, color=style.BLUE, lw=2)
    a1.set_title(f"{window}-day rolling mean (open days)")
    a1.set_ylabel("Income")
    a2.plot(rs.index, rs, color=style.ORANGE, lw=2)
    a2.set_title(f"{window}-day rolling standard deviation (open days)")
    a2.set_ylabel("Income")
    _fmt_dates(a2)
    style.save(fig, path)


def fig_profiles(p: Prepared, path):
    """Weekday and month distributions (open days)."""
    df = pd.DataFrame({"y": p.y, "open": p.open_mask})
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [7, 12]})
    for ax, key, labels, title in (
        (a1, "dow", [w[:3] for w in WEEKDAYS], "By weekday"),
        (a2, "month", MONTHS, "By month"),
    ):
        groups = [df.loc[df.open & (df[key] == k), "y"].to_numpy() for k in range(len(labels))
                  ] if key == "dow" else [df.loc[df.open & (df[key] == k + 1), "y"].to_numpy()
                                         for k in range(12)]
        pos = [i for i, g in enumerate(groups) if len(g)]
        bp = ax.boxplot([groups[i] for i in pos], positions=pos, widths=0.55, patch_artist=True,
                        showfliers=True, medianprops={"color": style.TEXT, "lw": 1.5},
                        flierprops={"marker": "o", "ms": 3, "mfc": style.MUTED, "mec": "none"})
        for b in bp["boxes"]:
            b.set(facecolor="#d6e5f7", edgecolor=style.BLUE)
        means = [groups[i].mean() for i in pos]
        ax.plot(pos, means, "o", color=style.BLUE, ms=6, label="Mean")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(f"Income distribution {title.lower()} (open days)")
        ax.set_ylabel("Income")
        ax.legend(loc="upper left")
    style.save(fig, path)


def fig_seasonal_years(p: Prepared, path):
    """Course-style seasonal plot: monthly mean, one line per year."""
    m = p.y.resample("ME").mean()
    years = sorted(set(m.index.year))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = style.SERIES
    for i, yv in enumerate(years[-len(cols):]):
        s = m[m.index.year == yv]
        ax.plot(s.index.month, s.to_numpy(), marker="o", ms=5, lw=2, color=cols[i], label=str(yv))
        ax.annotate(str(yv), (s.index.month[-1], s.iloc[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=style.TEXT_2)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_title("Seasonal plot: mean daily income per month, by year")
    ax.set_ylabel("Mean daily income")
    ax.legend(loc="upper left", ncols=min(len(years), 6))
    style.save(fig, path)


def fig_heatmap(p: Prepared, path):
    """Calendar heatmap: ISO week x weekday."""
    y = p.y
    df = pd.DataFrame({"y": y.to_numpy()}, index=y.index)
    df["week"] = (df.index - pd.to_timedelta(df.index.dayofweek, unit="D")).normalize()
    df["dow"] = df.index.dayofweek
    grid = df.pivot_table(index="dow", columns="week", values="y", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(7), [w[:3] for w in WEEKDAYS])
    n = grid.shape[1]
    step = max(1, n // 12)
    ax.set_xticks(range(0, n, step), [c.strftime("%Y-%m-%d") for c in grid.columns[::step]],
                  rotation=0, fontsize=8)
    ax.grid(False)
    ax.set_title("Calendar heatmap (week starting Monday x weekday)")
    fig.colorbar(im, ax=ax, label="Income", pad=0.01)
    style.save(fig, path)


# --------------------------------------------------------------------------- transformation
def guerrero_lambda(y: pd.Series, open_mask: pd.Series) -> float:
    """Guerrero (1993): lambda minimising the CV of sd_i / mean_i^(1-lambda) over months."""
    yo = y[open_mask & (y > 0)]
    g = yo.groupby([yo.index.year, yo.index.month])
    mu, sd = g.mean(), g.std()
    ok = (g.count() >= 5) & (mu > 0) & (sd > 0)
    mu, sd = mu[ok], sd[ok]
    if len(mu) < 3:
        return np.nan
    grid = np.linspace(-1, 2, 301)
    cv = [np.std(sd / mu ** (1 - lam)) / np.mean(sd / mu ** (1 - lam)) for lam in grid]
    return float(grid[int(np.argmin(cv))])


def transformation(p: Prepared, path) -> dict:
    yo = p.y[p.open_mask & (p.y > 0)]
    _, lam_ml, ci = stats.boxcox(yo.to_numpy(), alpha=0.05)
    lam_g = guerrero_lambda(p.y, p.open_mask)
    g = yo.groupby([yo.index.year, yo.index.month])
    mv = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.count()})
    mv = mv[(mv.n >= 5) & (mv.sd > 0)]
    slope = slope_p = np.nan
    if len(mv) >= 4:
        r = stats.linregress(np.log(mv["mean"]), np.log(mv["sd"]))
        slope, slope_p = float(r.slope), float(r.pvalue)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.scatter(mv["mean"], mv["sd"], s=42, color=style.BLUE, edgecolor=style.SURFACE, lw=1.5, zorder=3)
    if np.isfinite(slope):
        xs = np.linspace(mv["mean"].min(), mv["mean"].max(), 50)
        ax.plot(xs, np.exp(r.intercept) * xs ** slope, color=style.ORANGE, lw=2,
                label=f"sd ∝ mean^{slope:.2f}  (p = {slope_p:.3f})")
        ax.legend(loc="upper left")
    ax.set_xlabel("Monthly mean of daily income (open days)")
    ax.set_ylabel("Monthly standard deviation")
    ax.set_title("Mean-dispersion plot (one dot per month)")
    style.save(fig, path)

    lam_ref = lam_g if np.isfinite(lam_g) else lam_ml
    # Zeros (closures) rule out the log; sqrt is the usual zero-safe compromise.
    variance_depends_on_level = np.isfinite(slope_p) and slope_p < 0.05 and slope > 0.25
    if variance_depends_on_level and lam_ref < 0.75:
        choice = "sqrt"
    else:
        choice = "none"
    return {
        "lambda_ml": float(lam_ml), "lambda_ml_ci": [float(ci[0]), float(ci[1])],
        "lambda_guerrero": lam_g, "mv_slope": slope, "mv_slope_p": slope_p,
        "n_months_mv": int(len(mv)), "choice": choice,
    }


def apply_transform(x, kind):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.clip(x, 0, None)) if kind == "sqrt" else x


def invert_transform(x, kind):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 0, None) ** 2 if kind == "sqrt" else x


# --------------------------------------------------------------------------- stationarity
def _adf(x):
    try:
        r = adfuller(x, autolag="AIC")
        return float(r[0]), float(r[1])
    except Exception:
        return np.nan, np.nan


def _kpss(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = kpss(x, regression="c", nlags="auto")
            return float(r[0]), float(r[1])
        except Exception:
            return np.nan, np.nan


def stationarity(z: pd.Series, s: int = 7) -> dict:
    """ADF + KPSS on each differencing level, plus the course's variance check."""
    variants = {
        "W_t (levels)": z,
        "∇W_t": z.diff(),
        f"∇<sub>{s}</sub>W_t": z.diff(s),
        f"∇∇<sub>{s}</sub>W_t": z.diff(s).diff(),
        f"∇²∇<sub>{s}</sub>W_t": z.diff(s).diff().diff(),
    }
    rows = []
    for name, x in variants.items():
        x = x.dropna().to_numpy()
        adf_s, adf_p = _adf(x)
        kp_s, kp_p = _kpss(x)
        rows.append({"series": name, "variance": float(np.var(x, ddof=1)),
                     "adf_stat": adf_s, "adf_p": adf_p, "kpss_stat": kp_s, "kpss_p": kp_p})
    return {"table": rows}


def seasonal_strength(z: pd.Series, period: int = 7) -> dict:
    res = STL(z, period=period, robust=True).fit()
    var_r = np.var(res.resid)
    fs = max(0.0, 1 - var_r / np.var(res.seasonal + res.resid))
    ft = max(0.0, 1 - var_r / np.var(res.trend + res.resid))
    return {"F_seasonal": float(fs), "F_trend": float(ft)}


def choose_differencing(z: pd.Series, s: int = 7) -> dict:
    """D from seasonal strength (Hyndman-Wang rule F_S > 0.64), d from KPSS,
    then cross-checked against the variance criterion used in the course."""
    strength = seasonal_strength(z, s)
    D = 1 if strength["F_seasonal"] > 0.64 else 0
    base = z.diff(s).dropna() if D else z.dropna()
    _, kp_p = _kpss(base.to_numpy())
    d = 1 if (np.isfinite(kp_p) and kp_p < 0.05) else 0
    # Over-differencing guard: if the extra regular difference increases the variance, drop it.
    if d == 1 and np.var(base.diff().dropna()) > np.var(base):
        d_var_rule = 0
    else:
        d_var_rule = d
    return {"d": d_var_rule, "D": D, "d_kpss": d, **strength}


# --------------------------------------------------------------------------- decomposition
def decomposition(p: Prepared, path) -> dict:
    """Robust STL with period 7 and a 91-day trend. With >= 2 years, the annual cycle is
    split out of that trend by a Fourier regression (K = 3): an STL/MSTL seasonal smoother
    with period 365 only sees a handful of cycles and would interpolate every day-of-year."""
    y = p.y_clean
    n = len(y)
    use_annual = n >= 2 * 365
    res = STL(y, period=7, robust=True, trend=91 if n >= 3 * 91 else None).fit()
    if use_annual:
        t = np.arange(n)
        doy = (y.index - pd.Timestamp("2000-01-01")).days.to_numpy()
        F = np.column_stack([f(2 * np.pi * k * doy / 365.25) for k in (1, 2, 3) for f in (np.sin, np.cos)])
        X = np.column_stack([np.ones(n), t / 365.25, (t / 365.25) ** 2, F])
        beta, *_ = np.linalg.lstsq(X, res.trend.to_numpy(), rcond=None)
        annual = pd.Series(F @ beta[3:], index=y.index)
        comps = [("Trend", res.trend - annual), ("Weekly seasonality", res.seasonal),
                 ("Annual seasonality (Fourier, K = 3)", annual), ("Remainder", res.resid)]
    else:
        comps = [("Trend", res.trend), ("Weekly seasonality", res.seasonal), ("Remainder", res.resid)]
    use_mstl = use_annual
    fig, axes = plt.subplots(len(comps) + 1, 1, figsize=(11, 1.9 * (len(comps) + 1)), sharex=True)
    axes[0].plot(y.index, y, color=style.MUTED, lw=0.7)
    axes[0].set_title("Observed (irregular closures imputed)")
    colors = [style.BLUE, style.AQUA, style.VIOLET, style.TEXT_2]
    for ax, (name, c), col in zip(axes[1:], comps, colors):
        if name == "Remainder":
            ax.scatter(c.index, c, s=3, color=col)
            ax.axhline(0, color=style.TEXT_2, lw=0.8)
        else:
            ax.plot(c.index, c, color=col, lw=1.4 if "Weekly" in name else 2)
        ax.set_title(name)
    _fmt_dates(axes[-1])
    style.save(fig, path)
    resid = comps[-1][1]
    var_r = np.var(resid)
    out = {"method": "STL (period 7, 91-day trend) + annual Fourier split" if use_mstl
           else "STL (period 7)"}
    out["F_trend"] = float(max(0, 1 - var_r / np.var(comps[0][1] + resid)))
    out["F_weekly"] = float(max(0, 1 - var_r / np.var(comps[1][1] + resid)))
    if use_mstl:
        out["F_annual"] = float(max(0, 1 - var_r / np.var(comps[2][1] + resid)))
    wk = comps[1][1]
    prof = wk.groupby(wk.index.dayofweek).mean()
    out["weekly_profile"] = {WEEKDAYS[int(k)]: float(v) for k, v in prof.items()}
    out["resid"] = resid
    return out


def periodogram(p: Prepared, path) -> dict:
    y = p.y_clean
    x = (y - y.rolling(365 if len(y) > 400 else 29, center=True, min_periods=1).mean()
         if len(y) > 60 else y - y.mean()).to_numpy()
    f, pxx = signal.periodogram(x, detrend="linear", scaling="spectrum")
    f, pxx = f[1:], pxx[1:]
    period = 1 / f
    keep = period <= min(len(y) / 2, 400)
    period, pxx = period[keep], pxx[keep]
    idx = signal.find_peaks(pxx)[0]
    top = idx[np.argsort(pxx[idx])[::-1][:4]] if len(idx) else []
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.semilogx(period, pxx, color=style.BLUE, lw=1.4)
    for i in top:
        ax.annotate(f"{period[i]:.1f} d", (period[i], pxx[i]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9, color=style.TEXT)
    ax.set_xlabel("Period (days, log scale)")
    ax.set_ylabel("Power")
    ax.set_title("Periodogram (after removing the slow trend)")
    style.save(fig, path)
    return {"peaks": [float(period[i]) for i in top]}


def fig_acf_pacf(series_dict: dict, path, lags=56, s=7):
    n = len(series_dict)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n))
    axes = np.atleast_2d(axes)
    for row, (name, x) in zip(axes, series_dict.items()):
        x = pd.Series(x).dropna()
        lg = min(lags, len(x) // 2 - 1)
        plot_acf(x, ax=row[0], lags=lg, zero=False, color=style.BLUE, vlines_kwargs={"colors": style.BLUE})
        plot_pacf(x, ax=row[1], lags=lg, zero=False, method="ywm", color=style.BLUE,
                  vlines_kwargs={"colors": style.BLUE})
        for ax, kind in zip(row, ("ACF", "PACF")):
            for coll in ax.collections:
                if coll.__class__.__name__ == "PolyCollection":
                    coll.set_facecolor("#d6e5f7")
            for k in range(s, lg + 1, s):  # seasonal lags highlighted as in the course plots
                ax.axvline(k, color=style.ORANGE, lw=0.8, alpha=0.5, zorder=0)
            ax.set_title(f"{kind} of {name}")
            ax.set_ylim(-1, 1)
    style.save(fig, path)


def anomalies(p: Prepared, resid: pd.Series, z_crit: float = 3.5) -> pd.DataFrame:
    """Robust z-score of the decomposition remainder on normal open days."""
    r = resid[p.open_mask]
    mad = np.median(np.abs(r - r.median())) * 1.4826
    z = (r - r.median()) / mad if mad > 0 else r * 0
    out = pd.DataFrame({"income": p.y[z.index], "remainder": r, "robust_z": z})
    out = out[out.robust_z.abs() > z_crit].sort_values("robust_z", key=np.abs, ascending=False)
    return out

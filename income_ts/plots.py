"""Figures for model validation, outliers and forecasts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess

from . import style
from .eda import _fmt_dates


def _shade_ci(ax):
    for coll in ax.collections:
        if coll.__class__.__name__ == "PolyCollection":
            coll.set_facecolor("#d6e5f7")


def fig_diagnostics(val: dict, path, title, s=7, lags=56):
    e = val["resid"]
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(4, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(e.index, e, color=style.BLUE, lw=0.7)
    sd = e.std()
    for k in (-3, 3):
        ax.axhline(k * sd, color=style.RED, lw=0.8, ls="--")
    ax.axhline(0, color=style.TEXT_2, lw=0.8)
    ax.set_title("Residuals (dashed: ±3σ)")
    _fmt_dates(ax)

    ax = fig.add_subplot(gs[0, 1])
    r = np.sqrt(np.abs(val["resid_std"]))
    t = np.arange(len(r))
    ax.scatter(e.index, r, s=3, color=style.MUTED)
    sm = lowess(r, t, frac=0.15, return_sorted=False)
    ax.plot(e.index, sm, color=style.ORANGE, lw=2, label="LOWESS smoother")
    ax.set_title("√|standardized residuals| (constant variance check)")
    ax.legend(loc="upper left")
    _fmt_dates(ax)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(val["resid_std"], bins=40, density=True, color="#d6e5f7", edgecolor=style.BLUE, lw=0.6)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, stats.norm.pdf(xs), color=style.ORANGE, lw=2, label="N(0,1)")
    ax.set_title(f"Histogram (Shapiro p = {val['shapiro_p']:.3g}, JB p = {val['jb_p']:.3g})")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    (osm, osr), (sl, ic, _) = stats.probplot(val["resid_std"], dist="norm")
    ax.scatter(osm, osr, s=5, color=style.BLUE)
    ax.plot(osm, sl * osm + ic, color=style.ORANGE, lw=2)
    ax.set_title("Normal Q-Q plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")

    for j, (fn, name) in enumerate(((plot_acf, "ACF"), (plot_pacf, "PACF"))):
        ax = fig.add_subplot(gs[2, j])
        kw = {"method": "ywm"} if name == "PACF" else {}
        fn(e, ax=ax, lags=lags, zero=False, color=style.BLUE, vlines_kwargs={"colors": style.BLUE}, **kw)
        _shade_ci(ax)
        for k in range(s, lags + 1, s):
            ax.axvline(k, color=style.ORANGE, lw=0.8, alpha=0.5, zorder=0)
        ax.set_ylim(-0.3, 0.3)
        ax.set_title(f"{name} of residuals")

    ax = fig.add_subplot(gs[3, :])
    pv = val["lb_pvalues_by_lag"]
    ax.scatter(pv.index, pv.to_numpy(), s=16, color=style.BLUE, zorder=3)
    ax.axhline(0.05, color=style.RED, lw=1, ls="--", label="5 %")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Lag")
    ax.set_ylabel("p-value")
    ax.set_title("Ljung-Box p-values by lag (as in tsdiag)")
    ax.legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    style.save(fig, path)


def fig_psi(val, path, title):
    psi = np.asarray(val["psi"][1:])
    pi = -np.asarray(val["pi"][1:])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, w, name in ((a1, psi, "ψ weights (MA(∞) form)"), (a2, pi, "π weights (AR(∞) form)")):
        k = np.arange(1, len(w) + 1)
        ax.vlines(k, 0, w, color=style.BLUE, lw=2)
        ax.scatter(k, w, s=14, color=style.BLUE, zorder=3)
        ax.axhline(0, color=style.TEXT_2, lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Lag")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    style.save(fig, path)


def fig_outliers(y, lin, table, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index, y, color=style.MUTED, lw=0.7, label="Observed")
    ax.plot(lin.index, lin, color=style.BLUE, lw=0.9, label="Outlier-adjusted (linearized)")
    marks = {"AO": ("o", style.RED), "LS": ("s", style.VIOLET), "TC": ("^", style.ORANGE)}
    for kind, (mk, col) in marks.items():
        ds = [pd.Timestamp(r["date"]) for r in table if r["type"] == kind]
        if ds:
            ax.scatter(ds, y[ds], marker=mk, s=40, color=col, edgecolor=style.SURFACE, lw=1,
                       zorder=4, label=kind)
    ax.set_title("Automatic outlier detection (AO: additive, LS: level shift, TC: transitory)")
    ax.set_ylabel("Income")
    _fmt_dates(ax)
    ax.legend(loc="upper left", ncols=5, fontsize=9)
    style.save(fig, path)


def fig_forecast(y, fc, path, title, history_days=120):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    h = y.iloc[-history_days:]
    ax.plot(h.index, h, color=style.MUTED, lw=1.2, label="Observed")
    ax.fill_between(fc.index, fc.lo95, fc.hi95, color="#d6e5f7", lw=0, label="95 % interval")
    ax.fill_between(fc.index, fc.lo80, fc.hi80, color="#a9c9ee", lw=0, label="80 % interval")
    ax.plot(fc.index, fc["mean"], color=style.BLUE, lw=2, label="Forecast")
    ax.axvline(y.index[-1] + pd.Timedelta(hours=12), color=style.TEXT_2, lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_ylabel("Income")
    _fmt_dates(ax)
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    style.save(fig, path)


def fig_monthly(m, fit_index, fitted, fc, path, title):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(m.index, m, color=style.TEXT_2, lw=1.6, marker="o", ms=3, label="Monthly income")
    ax.fill_between(fc.index, fc.lo95, fc.hi95, color="#d6e5f7", lw=0, label="95 % interval")
    ax.plot(fc.index, fc["mean"], color=style.BLUE, lw=2, marker="o", ms=4, label="Forecast")
    ax.set_title(title)
    ax.set_ylabel("Monthly income")
    _fmt_dates(ax)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    style.save(fig, path)

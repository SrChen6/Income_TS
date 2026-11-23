"""Monthly view: the classic Box-Jenkins SARIMA(p,d,q)(P,D,Q)_12 workflow of the course,
applied to monthly income with a trading-day (open days) regressor."""

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

from . import plots
from .calendar import holiday_series


def _open_days(index_month_end, regular_dows, region, holiday_closed):
    out = []
    for me in index_month_end:
        days = pd.date_range(me.replace(day=1), me, freq="D")
        ok = ~np.isin(days.dayofweek, regular_dows)
        if holiday_closed:
            ok &= holiday_series(days, region).to_numpy() == 0
        out.append(int(ok.sum()))
    return pd.Series(out, index=index_month_end, name="open_days")


def analyse(ctx, fig_dir, horizon=12):
    p = ctx.p
    counts = p.y.resample("ME").count()
    m = p.y.resample("ME").sum()
    full = counts.to_numpy() == counts.index.days_in_month.to_numpy()
    m = m[full]
    if len(m) < 36:
        return {"available": False, "n_months": int(len(m))}
    m = m.asfreq("ME")
    if m.isna().any():
        return {"available": False, "n_months": int(len(m)), "reason": "gaps between complete months"}
    future = pd.date_range(m.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
    od = _open_days(m.index.append(future), p.regular_closed_dows, ctx.region, ctx.holiday_closed)
    use_td = od.loc[m.index].std() > 0

    stl = STL(m, period=12, robust=True).fit()
    fs = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
    D = 1 if fs > 0.64 else 0
    base = m.diff(12).dropna() if D else m
    try:
        d = 1 if kpss(base, regression="c", nlags="auto")[1] < 0.05 else 0
    except Exception:
        d = 1

    transforms = ["none"] + (["log"] if (m > 0).all() else [])
    rows = []
    for tr in transforms:
        z = np.log(m) if tr == "log" else m
        jac = 2 * np.log(m).sum() if tr == "log" else 0.0  # AIC on the original scale
        for p_, q_, P_, Q_ in itertools.product(range(3), range(3), range(2), range(2)):
            try:
                res = SARIMAX(z, exog=od.loc[m.index] if use_td else None, order=(p_, d, q_),
                              seasonal_order=(P_, D, Q_, 12), trend="c" if d + D == 0 else "n"
                              ).fit(disp=False, maxiter=200)
                rows.append({"transform": tr, "order": (p_, d, q_), "sorder": (P_, D, Q_, 12),
                             "AIC": res.aic + jac, "BIC": res.bic + jac})
            except Exception:
                pass
    tab = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best = tab.iloc[0]
    tr, order, sorder = best.transform, tuple(best.order), tuple(best.sorder)

    def fit(series, exog):
        z = np.log(series) if tr == "log" else series
        return SARIMAX(z, exog=exog, order=order, seasonal_order=sorder,
                       trend="c" if d + D == 0 else "n").fit(disp=False, maxiter=200)

    def fc_frame(res, idx, exog):
        f = res.get_forecast(len(idx), exog=exog)
        sf95, sf80 = f.summary_frame(alpha=0.05), f.summary_frame(alpha=0.2)
        out = pd.DataFrame(index=idx)
        g = np.exp if tr == "log" else (lambda v: v)
        out["mean"] = g(sf95["mean"].to_numpy())
        out["lo95"], out["hi95"] = g(sf95["mean_ci_lower"].to_numpy()), g(sf95["mean_ci_upper"].to_numpy())
        out["lo80"], out["hi80"] = g(sf80["mean_ci_lower"].to_numpy()), g(sf80["mean_ci_upper"].to_numpy())
        return out.clip(lower=0)

    # hold-out: last 12 months
    tr_m, te_m = m.iloc[:-12], m.iloc[-12:]
    ex = (lambda idx: od.loc[idx].to_frame()) if use_td else (lambda idx: None)
    hold = fc_frame(fit(tr_m, ex(tr_m.index)), te_m.index, ex(te_m.index))
    sn = tr_m.iloc[-12:].to_numpy()
    acc = {}
    for name, pred in (("SARIMA", hold["mean"].to_numpy()), ("Seasonal naive", sn)):
        e = te_m.to_numpy() - pred
        acc[name] = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                     "MAPE %": float(np.mean(np.abs(e / te_m.to_numpy())) * 100)}
    acc["SARIMA"]["Cov95 %"] = float(((te_m >= hold.lo95) & (te_m <= hold.hi95)).mean() * 100)

    res = fit(m, ex(m.index))
    fc = fc_frame(res, future, ex(future))
    plots.fig_monthly(m, m.index, None, fc, fig_dir / "monthly_forecast.png",
                      f"Monthly SARIMA{order}{sorder[:3]}₁₂"
                      f"{' on log scale' if tr == 'log' else ''}: {horizon}-month forecast")
    td_coef = None
    if use_td:
        td_coef = {"coef": float(res.params["open_days"]), "p": float(res.pvalues["open_days"])}
    return {
        "available": True, "n_months": int(len(m)), "F_seasonal": float(fs), "d": d, "D": D,
        "transform": tr, "order": order, "sorder": sorder, "top": tab.head(5),
        "holdout": acc, "forecast": fc, "trading_day": td_coef,
        "ljungbox_p24": float(res.test_serial_correlation("ljungbox", lags=[24])[0][1][-1]),
    }

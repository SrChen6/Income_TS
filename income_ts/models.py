"""Forecasting models.

Every model is a function ``f(ctx, train_end, future_index) -> DataFrame`` returning the
columns ``mean, lo80, hi80, lo95, hi95`` (intervals may be NaN) on the original scale.
``train_end`` is the last date the model is allowed to see.
"""

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima_process import arma2ar, arma2ma
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .calendar import holiday_series
from .data import Prepared
from .eda import apply_transform, invert_transform

warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

Z95, Z80 = 1.959964, 1.281552
COLS = ["mean", "lo80", "hi80", "lo95", "hi95"]


# --------------------------------------------------------------------------- context
@dataclass
class Context:
    p: Prepared
    region: str = "catalonia"
    transform: str = "none"
    s: int = 7
    d: int = 0
    D: int = 1
    fourier_K: int = 0
    sarima_order: tuple = (1, 0, 1)
    sarima_sorder: tuple = (0, 1, 1, 7)
    ets_damped: bool = True
    outliers: list = field(default_factory=list)  # [(Timestamp, type)]
    holiday_closed: bool = False

    def __post_init__(self):
        far = self.p.y.index.max() + pd.Timedelta(days=800)
        full = pd.date_range(self.p.y.index.min(), far, freq="D")
        hol = holiday_series(full, self.region)
        # a holiday on a regular closing day changes nothing: keep the dummy at 0 there
        self.holiday = hol.where(~np.isin(full.dayofweek, self.p.regular_closed_dows), 0)
        hol_hist = self.holiday.reindex(self.p.y.index)
        hol_days = hol_hist[hol_hist == 1].index
        hol_days = hol_days[~np.isin(hol_days.dayofweek, self.p.regular_closed_dows)]
        if len(hol_days) >= 2:
            self.holiday_closed = bool((self.p.y[hol_days] == 0).mean() >= 0.8)
        self.holiday_closed_rate = float((self.p.y[hol_days] == 0).mean()) if len(hol_days) else np.nan
        # an irregular closure that falls on a holiday is explained by the holiday dummy
        self.closed_nh = (self.p.closed.astype(bool) & ~hol_hist.astype(bool)).astype(int)

    # exogenous regressors ------------------------------------------------------
    def exog(self, index, fourier_K=None, outliers=None):
        K = self.fourier_K if fourier_K is None else fourier_K
        X = pd.DataFrame(index=index)
        X["holiday"] = self.holiday.reindex(index).fillna(0).to_numpy()
        X["closure"] = self.closed_nh.reindex(index).fillna(0).to_numpy()  # unknown future -> 0
        if K:
            t = (index - pd.Timestamp("2000-01-01")).days.to_numpy()
            for k in range(1, K + 1):
                X[f"sin{k}"] = np.sin(2 * np.pi * k * t / 365.25)
                X[f"cos{k}"] = np.cos(2 * np.pi * k * t / 365.25)
        for T, kind in (self.outliers if outliers is None else outliers):
            X[f"{kind}_{T.date()}"] = outlier_pattern(index, T, kind)
        return X

    def postprocess(self, fc: pd.DataFrame) -> pd.DataFrame:
        fc = fc.copy()
        fc[COLS] = fc[COLS].clip(lower=0)
        zero = np.isin(fc.index.dayofweek, self.p.regular_closed_dows)
        if self.holiday_closed:
            zero |= self.holiday.reindex(fc.index).fillna(0).to_numpy().astype(bool)
        fc.loc[zero, COLS] = 0.0
        return fc


def _usable(X_train, X_future):
    keep = [c for c in X_train.columns if X_train[c].std() > 0]
    if not keep:
        return None, None
    return X_train[keep], X_future[keep]


def outlier_pattern(index, T, kind, delta=0.7):
    k = (index - T).days.to_numpy()
    if kind == "AO":
        return (k == 0).astype(float)
    if kind == "LS":
        return (k >= 0).astype(float)
    if kind == "TC":
        return np.where(k >= 0, delta ** np.clip(k, 0, None), 0.0)
    raise ValueError(kind)


def aicc(res):
    k = len(res.params)
    n = res.nobs
    return res.aic + 2 * k * (k + 1) / max(n - k - 1, 1)


def _trend(d, D):
    return "c" if d + D == 0 else "n"


def _frame(mean, se, index, kind="none"):
    """Build interval frame from a mean and standard error on the model scale."""
    mean, se = np.asarray(mean, float), np.asarray(se, float)
    out = pd.DataFrame(index=index)
    if kind == "sqrt":
        out["mean"] = np.clip(mean, 0, None) ** 2 + se ** 2  # bias-adjusted back-transform
    else:
        out["mean"] = mean
    for z, lo, hi in ((Z80, "lo80", "hi80"), (Z95, "lo95", "hi95")):
        out[lo] = invert_transform(mean - z * se, kind)
        out[hi] = invert_transform(mean + z * se, kind)
    return out


# --------------------------------------------------------------------------- SARIMA
def fit_sarimax(z, X, order, sorder, trend):
    mod = SARIMAX(z, exog=X, order=order, seasonal_order=sorder, trend=trend)
    return mod.fit(disp=False, maxiter=300)


def select_sarima(ctx: Context, train_end=None, max_p=2, max_q=2, max_P=1, max_Q=1, fast=False):
    """Grid search by AICc over (p,q)(P,Q) with d, D fixed by the identification step."""
    y = ctx.p.y.loc[:train_end]
    z = pd.Series(apply_transform(y, ctx.transform), index=y.index)
    X = ctx.exog(y.index, fourier_K=0, outliers=[])
    X, _ = _usable(X, X)
    rows = []
    if fast:
        max_p, max_q = 1, 1
    for p_, q_, P_, Q_ in itertools.product(range(max_p + 1), range(max_q + 1),
                                            range(max_P + 1), range(max_Q + 1)):
        order, sorder = (p_, ctx.d, q_), (P_, ctx.D, Q_, ctx.s)
        try:
            res = fit_sarimax(z, X, order, sorder, _trend(ctx.d, ctx.D))
            if not np.isfinite(res.aic):
                continue
            rows.append({"order": order, "sorder": sorder, "AIC": res.aic, "BIC": res.bic,
                         "AICc": aicc(res), "k": len(res.params)})
        except Exception:
            continue
    tab = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    return tab


def pick_two_models(tab: pd.DataFrame):
    """Course-style: Model 1 = best AICc; Model 2 = best BIC among the other specifications
    (usually the more parsimonious alternative)."""
    m1 = tab.iloc[0]
    rest = tab.iloc[1:]
    m2 = rest.sort_values("BIC").iloc[0] if len(rest) else m1
    return (tuple(m1.order), tuple(m1.sorder)), (tuple(m2.order), tuple(m2.sorder))


def sarima_forecast(ctx: Context, train_end, future_index, order=None, sorder=None,
                    fourier_K=0, outliers=None, return_res=False):
    order = order or ctx.sarima_order
    sorder = sorder or ctx.sarima_sorder
    y = ctx.p.y.loc[:train_end]
    z = pd.Series(apply_transform(y, ctx.transform), index=y.index)
    outl = ctx.outliers if outliers is None else outliers
    X, Xf = _usable(ctx.exog(y.index, fourier_K, outl), ctx.exog(future_index, fourier_K, outl))
    res = fit_sarimax(z, X, order, sorder, _trend(order[1], sorder[1]))
    f = res.get_forecast(len(future_index), exog=Xf)
    fc = ctx.postprocess(_frame(f.predicted_mean, f.se_mean, future_index, ctx.transform))
    return (fc, res) if return_res else fc


def select_fourier_K(ctx: Context, train_end=None, Ks=(1, 2, 3, 4, 6)):
    y = ctx.p.y.loc[:train_end]
    if len(y) < 400:
        return 0, []
    z = pd.Series(apply_transform(y, ctx.transform), index=y.index)
    rows = []
    for K in Ks:
        X, _ = _usable(ctx.exog(y.index, K, []), ctx.exog(y.index, K, []))
        try:
            res = fit_sarimax(z, X, ctx.sarima_order, ctx.sarima_sorder,
                              _trend(ctx.d, ctx.D))
            rows.append({"K": K, "AICc": aicc(res)})
        except Exception:
            pass
    if not rows:
        return 0, []
    best = min(rows, key=lambda r: r["AICc"])
    return best["K"], rows


def dhr_forecast(ctx, train_end, future_index):
    return sarima_forecast(ctx, train_end, future_index, fourier_K=ctx.fourier_K)


# --------------------------------------------------------------------------- validation
def full_polynomials(res, order, sorder):
    """AR polynomial including differencing, and MA polynomial (statsmodels sign convention)."""
    ar = np.asarray(res.polynomial_reduced_ar, float)
    ma = np.asarray(res.polynomial_reduced_ma, float)
    d, D, s = order[1], sorder[1], sorder[3]
    full = ar.copy()
    for _ in range(d):
        full = np.convolve(full, [1, -1])
    for _ in range(D):
        sd = np.zeros(s + 1)
        sd[0], sd[-1] = 1, -1
        full = np.convolve(full, sd)
    return ar, ma, full


def validate(res, order, sorder, lb_max=56):
    d, D, s = order[1], sorder[1], sorder[3]
    burn = d + D * s
    e = np.asarray(res.resid)[burn:]
    e_std = (e - e.mean()) / e.std()
    k_arma = order[0] + order[2] + sorder[0] + sorder[2]
    lags = [lag for lag in (7, 14, 21, 28, 42, 56) if lag <= lb_max and lag > k_arma]
    lb = acorr_ljungbox(e, lags=lags, model_df=k_arma)
    lb_all = acorr_ljungbox(e, lags=range(1, lb_max + 1))
    lb_df_p = acorr_ljungbox(e, lags=range(k_arma + 1, lb_max + 1), model_df=k_arma)["lb_pvalue"]
    shapiro = stats.shapiro(e[:5000])
    jb = stats.jarque_bera(e)
    arch = het_arch(e, nlags=7)
    try:
        bv = res.test_heteroskedasticity("breakvar")[0]
        breakvar_p = float(bv[1])
    except Exception:
        breakvar_p = np.nan
    ar, ma, _ = full_polynomials(res, order, sorder)
    ar_roots = np.roots(ar[::-1]) if len(ar) > 1 else np.array([])
    ma_roots = np.roots(ma[::-1]) if len(ma) > 1 else np.array([])
    psi = arma2ma(ar, ma, lags=22)
    pi = arma2ar(ar, ma, lags=22)
    names = list(res.model.param_names)
    coefs = pd.DataFrame({"coef": res.params, "se": res.bse, "z": res.tvalues, "p": res.pvalues},
                         index=names)
    return {
        "resid": pd.Series(e, index=res.model._index[burn:]),
        "resid_std": e_std,
        "ljungbox": [{"lag": int(lag), "Q": float(q), "p": float(pv)}
                     for lag, q, pv in zip(lags, lb["lb_stat"], lb["lb_pvalue"])],
        "lb_pvalues_by_lag": lb_df_p,
        "lb_all": lb_all,
        "shapiro_p": float(shapiro.pvalue), "jb_p": float(jb.pvalue),
        "skew": float(stats.skew(e)), "kurtosis": float(stats.kurtosis(e)),
        "arch_p": float(arch[1]), "breakvar_p": breakvar_p,
        "ar_root_mod": sorted(np.abs(ar_roots).tolist()),
        "ma_root_mod": sorted(np.abs(ma_roots).tolist()),
        "causal": bool(np.all(np.abs(ar_roots) > 1.0)) if len(ar_roots) else True,
        "invertible": bool(np.all(np.abs(ma_roots) > 1.0)) if len(ma_roots) else True,
        "psi": psi.tolist(), "pi": pi.tolist(),
        "aic": float(res.aic), "bic": float(res.bic), "aicc": float(aicc(res)),
        "sigma2": float(res.params.get("sigma2", np.nan)) if hasattr(res.params, "get") else np.nan,
        "loglik": float(res.llf),
        "coefs": coefs,
        "n_obs": int(res.nobs),
    }


def stability(ctx: Context, order, sorder, h):
    """Course 'model stability' check: re-estimate without the last h days and compare."""
    y = ctx.p.y
    cut = y.index[-h - 1]
    _, r_full = sarima_forecast(ctx, y.index[-1], y.index[-1:] + pd.Timedelta(days=1),
                                order, sorder, return_res=True)
    _, r_cut = sarima_forecast(ctx, cut, y.index[-1:] + pd.Timedelta(days=1),
                               order, sorder, return_res=True)
    tab = pd.DataFrame({"full": r_full.params, "full_se": r_full.bse,
                        "without_last_h": r_cut.params})
    tab["shift_in_se"] = (tab["without_last_h"] - tab["full"]) / tab["full_se"]
    tab = tab.drop(index=[i for i in tab.index if i == "sigma2"], errors="ignore")
    return tab


# --------------------------------------------------------------------------- outliers
def detect_outliers(ctx: Context, train_end=None, crit=3.5, max_iter=10,
                    types=("AO", "LS", "TC"), delta=0.7):
    """Iterative Chen-Liu style detection of additive outliers (AO), level shifts (LS) and
    transitory changes (TC), the daily analogue of the course's ``outdetec``."""
    order, sorder = ctx.sarima_order, ctx.sarima_sorder
    y = ctx.p.y.loc[:train_end]
    idx = y.index
    n = len(y)
    z = pd.Series(apply_transform(y, ctx.transform), index=idx)
    found = []
    burn = order[1] + sorder[1] * sorder[3] + sorder[3]
    tail_guard = 14
    for _ in range(max_iter):
        X, _ = _usable(ctx.exog(idx, 0, found), ctx.exog(idx, 0, found))
        res = fit_sarimax(z, X, order, sorder, _trend(order[1], sorder[1]))
        e = np.array(res.resid, dtype=float, copy=True)
        e[:burn] = 0.0
        sigma = 1.4826 * np.median(np.abs(e[burn:] - np.median(e[burn:])))
        _, ma, full = full_polynomials(res, order, sorder)
        pi = arma2ar(full, ma, lags=n)
        best = (0.0, None, None)
        taken = {T for T, _ in found}
        for kind in types:
            if kind == "AO":
                c = pi
            elif kind == "LS":
                c = np.cumsum(pi)
            else:
                c = signal.lfilter([1.0], [1.0, -delta], pi)
            num = signal.correlate(e, c, mode="full")[n - 1:]
            den = np.cumsum(c[:n] ** 2)[n - 1 - np.arange(n)]  # den[T] = sum_{j < n-T} c_j^2
            tau = num / np.sqrt(den) / sigma
            tau[:burn] = 0
            if kind != "AO":
                tau[-tail_guard:] = 0
            for T in taken:
                tau[idx.get_loc(T)] = 0
            j = int(np.argmax(np.abs(tau)))
            if abs(tau[j]) > abs(best[0]):
                best = (float(tau[j]), idx[j], kind)
        if abs(best[0]) < crit:
            break
        found.append((best[1], best[2]))
    # final fit with all outliers to get their size
    X, _ = _usable(ctx.exog(idx, 0, found), ctx.exog(idx, 0, found))
    res = fit_sarimax(z, X, order, sorder, _trend(order[1], sorder[1]))
    table = []
    for T, kind in found:
        name = f"{kind}_{T.date()}"
        table.append({"date": T.date().isoformat(), "type": kind,
                      "effect": float(res.params[name]), "z": float(res.tvalues[name])})
    return found, table, res


def linearized(ctx: Context, res, outliers):
    """Series with the estimated outlier effects removed (course: ``lineal``)."""
    adj = np.zeros(len(ctx.p.y))
    for T, kind in outliers:
        adj += res.params[f"{kind}_{T.date()}"] * outlier_pattern(ctx.p.y.index, T, kind)
    z = apply_transform(ctx.p.y, ctx.transform) - adj
    return pd.Series(invert_transform(z, ctx.transform), index=ctx.p.y.index)


# --------------------------------------------------------------------------- ETS
def ets_fit(ctx, train_end, damped=None):
    y = ctx.p.y_clean.loc[:train_end]
    z = pd.Series(apply_transform(y, ctx.transform), index=y.index).asfreq("D")
    damped = ctx.ets_damped if damped is None else damped
    trend = "add" if damped is not False else None
    mod = ETSModel(z, error="add", trend=trend, damped_trend=bool(trend), seasonal="add",
                   seasonal_periods=ctx.s)
    return mod.fit(disp=False, maxiter=500)


def select_ets(ctx, train_end=None):
    out = {}
    for label, damped in (("A,Ad,A", True), ("A,N,A", False)):
        try:
            r = ets_fit(ctx, train_end, damped)
            out[label] = aicc(r)
        except Exception:
            pass
    best = min(out, key=out.get)
    return best == "A,Ad,A", out


def ets_forecast(ctx, train_end, future_index):
    res = ets_fit(ctx, train_end)
    n = res.nobs
    pr = res.get_prediction(start=n, end=n + len(future_index) - 1)
    sf = pr.summary_frame(alpha=0.05)
    mean = sf["mean"].to_numpy()
    se = (sf["pi_upper"].to_numpy() - mean) / Z95
    return ctx.postprocess(_frame(mean, se, future_index, ctx.transform))


# --------------------------------------------------------------------------- baselines
def _bl_frame(ctx, mean, se, future_index):
    return ctx.postprocess(_frame(mean, se, future_index))


def naive_forecast(ctx, train_end, future_index):
    y = ctx.p.y_clean.loc[:train_end]
    last = y[ctx.p.open_mask.loc[:train_end]].iloc[-1]
    e = y[ctx.p.open_mask.loc[:train_end]].diff().dropna()
    se = e.std() * np.sqrt(np.arange(1, len(future_index) + 1))
    return _bl_frame(ctx, np.full(len(future_index), last), se, future_index)


def snaive_forecast(ctx, train_end, future_index):
    y = ctx.p.y_clean.loc[:train_end]
    s = ctx.s
    last_week = y.iloc[-s:].to_numpy()
    k = np.arange(len(future_index))
    mean = last_week[k % s]
    sd = (y - y.shift(s)).dropna().std()
    se = sd * np.sqrt(k // s + 1)
    return _bl_frame(ctx, mean, se, future_index)


def smean_forecast(ctx, train_end, future_index, weeks=4):
    y = ctx.p.y_clean.loc[:train_end]
    s = ctx.s
    block = y.iloc[-s * weeks:].to_numpy().reshape(weeks, s)
    prof = block.mean(axis=0)
    k = np.arange(len(future_index))
    mean = prof[k % s]
    sd = (y - y.rolling(s * weeks).mean().shift(1)).dropna().std()
    return _bl_frame(ctx, mean, np.full(len(future_index), sd), future_index)


# --------------------------------------------------------------------------- gradient boosting
def _gbm_features(ctx, index, y_full, L1):
    f = pd.DataFrame(index=index)
    f["dow"] = index.dayofweek
    f["month"] = index.month
    f["dom"] = index.day
    t = index.dayofyear.to_numpy()
    for k in (1, 2, 3):
        f[f"s{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        f[f"c{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    f["holiday"] = ctx.holiday.reindex(index).fillna(0).to_numpy()
    f["closure"] = ctx.closed_nh.reindex(index).fillna(0).to_numpy()
    yy = y_full.reindex(y_full.index.union(index))
    for j in range(4):
        f[f"lag{L1 + 7 * j}"] = yy.shift(L1 + 7 * j).reindex(index).to_numpy()
    base = yy.shift(L1)
    f["roll7"] = base.rolling(7, min_periods=4).mean().reindex(index).to_numpy()
    f["roll28"] = base.rolling(28, min_periods=14).mean().reindex(index).to_numpy()
    f["roll91"] = base.rolling(91, min_periods=30).mean().reindex(index).to_numpy()
    if len(y_full) > 400:
        f["lag364"] = yy.shift(364).reindex(index).to_numpy()
    return f


def gbm_forecast(ctx, train_end, future_index, seed=0):
    """Direct multi-horizon gradient boosting: only lags >= horizon, so no recursion."""
    h = len(future_index)
    L1 = int(np.ceil(max(h, 1) / 7) * 7)
    y = ctx.p.y.loc[:train_end]
    Xtr = _gbm_features(ctx, y.index, y, L1)
    Xf = _gbm_features(ctx, future_index, y, L1)
    keep = Xtr.notna().mean() > 0.5
    Xtr, Xf = Xtr.loc[:, keep], Xf.loc[:, keep]
    cat = [c in ("dow", "month") for c in Xtr.columns]
    params = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=15,
                  l2_regularization=1.0, categorical_features=cat, random_state=seed)
    out = pd.DataFrame(index=future_index)
    out["mean"] = HistGradientBoostingRegressor(loss="squared_error", **params).fit(Xtr, y).predict(Xf)
    for q, col in ((0.1, "lo80"), (0.9, "hi80"), (0.025, "lo95"), (0.975, "hi95")):
        out[col] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **params).fit(Xtr, y).predict(Xf)
    qs = np.sort(out[["lo95", "lo80", "hi80", "hi95"]].to_numpy(), axis=1)
    out[["lo95", "lo80", "hi80", "hi95"]] = qs
    return ctx.postprocess(out)


# --------------------------------------------------------------------------- registry
def registry(ctx: Context, include_dhr=True):
    reg = {
        "Naive": naive_forecast,
        "Seasonal naive": snaive_forecast,
        "Seasonal mean": smean_forecast,
        "ETS": ets_forecast,
        "SARIMA": sarima_forecast,
    }
    if include_dhr and ctx.fourier_K:
        reg["DHR (SARIMA + Fourier)"] = dhr_forecast
    reg["Gradient boosting"] = gbm_forecast
    return reg


ENSEMBLE_MEMBERS = ("ETS", "SARIMA", "DHR (SARIMA + Fourier)", "Gradient boosting")


def ensemble(forecasts: dict) -> pd.DataFrame:
    members = [forecasts[m] for m in ENSEMBLE_MEMBERS if m in forecasts]
    out = pd.DataFrame(index=members[0].index)
    out["mean"] = np.mean([m["mean"].to_numpy() for m in members], axis=0)
    for c in COLS[1:]:
        out[c] = np.nan
    return out

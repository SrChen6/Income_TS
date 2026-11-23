"""Render REPORT.md. All interpretation sentences are generated from the computed
statistics, so the report stays correct when the data are updated and re-run."""

import math

import jinja2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- formatting
def num(x, dec=0):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{dec}f}"


def pval(x):
    if x is None or not math.isfinite(x):
        return "–"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def md_table(df: pd.DataFrame, fmt=None, index=True, index_name=""):
    fmt = fmt or {}
    cols = list(df.columns)
    head = ([index_name] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join([":--"] * (1 if index else 0) + ["--:"] * len(cols)) + "|"]
    for i, r in df.iterrows():
        cells = [str(i)] if index else []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if callable(f):
                cells.append(f(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(num(float(v), 2 if f is None else f))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sarima_name(o, so):
    return f"SARIMA({o[0]},{o[1]},{o[2]})({so[0]},{so[1]},{so[2]})<sub>{so[3]}</sub>"


def jsonable(R):
    def conv(v):
        if isinstance(v, pd.DataFrame):
            return v.reset_index().to_dict(orient="records")
        if isinstance(v, pd.Series):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    out = conv({k: v for k, v in R.items()})
    for tag in ("m1", "m2"):
        if "validation" in out and tag in out["validation"]:
            for k in ("resid", "resid_std", "lb_all", "lb_pvalues_by_lag"):
                out["validation"][tag].pop(k, None)
    return out


# --------------------------------------------------------------------------- narrative
def _narrative(R):
    N = {}
    s, dmeta = R["summary"], R["data"]

    # data
    bits = []
    if dmeta["regular_closed_days"]:
        bits.append(f"the store is regularly closed on **{', '.join(dmeta['regular_closed_days'])}** "
                    f"(zero or no income on ≥ 80 % of them); those days are set to 0 and forecast as 0")
    if dmeta["n_irregular_closures"]:
        bits.append(f"**{dmeta['n_irregular_closures']}** zero-income days on normally open weekdays "
                    "are treated as irregular closures (holidays, vacations) and modelled with dummies")
    if dmeta["n_imputed"]:
        bits.append(f"**{dmeta['n_imputed']}** missing open days were imputed from the same weekday "
                    "in neighbouring weeks")
    N["cleaning"] = ("; ".join(bits) + ".") if bits else "No gaps, closures or duplicates needed treatment."

    # trend
    if "trend_per_year" in s:
        sig = s["trend_p"] < 0.05
        direction = "upward" if s["trend_per_year"] > 0 else "downward"
        N["trend"] = (f"A linear fit to the monthly means gives an {direction} trend of "
                      f"{num(s['trend_per_year'], 1)} per day per year ({num(s['trend_per_year_pct'], 1)} % of the "
                      f"mean level, HAC p = {pval(s['trend_p'])}), which is "
                      f"{'statistically significant' if sig else 'not statistically significant'}.")
    else:
        N["trend"] = "The sample is too short to estimate a trend reliably."
    if s.get("growth_last12") is not None and np.isfinite(s.get("growth_last12", np.nan)):
        N["trend"] += (f" Income over the last 12 months is {num(s['growth_last12'], 1)} % "
                       f"{'above' if s['growth_last12'] >= 0 else 'below'} the previous 12 months.")

    dm = R["dow_means"]
    best_d, worst_d = max(dm, key=dm.get), min(dm, key=dm.get)
    N["weekly"] = (f"The busiest weekday on average is **{best_d}** ({num(dm[best_d])}) and the quietest open "
                   f"day is **{worst_d}** ({num(dm[worst_d])}), a ratio of {num(dm[best_d] / dm[worst_d], 2)}×."
                   if dm[worst_d] > 0 else "")
    mm = {k: v for k, v in R["month_means"].items()}
    if len(mm) >= 6:
        bm, wm = max(mm, key=mm.get), min(mm, key=mm.get)
        N["monthly"] = (f"By month, mean daily income peaks in **{bm}** ({num(mm[bm])}) and is lowest in "
                        f"**{wm}** ({num(mm[wm])}).")
    else:
        N["monthly"] = ""

    # transformation
    t = R["transform"]
    if t["choice"] == "sqrt":
        N["transform"] = ("The dispersion grows with the level (significant mean–sd slope) and the Guerrero "
                          "λ is well below 1, so the models are fitted on **√income** (the log is not possible "
                          "because closures produce zeros); forecasts are back-transformed with bias adjustment.")
    else:
        why = []
        if not (np.isfinite(t["mv_slope_p"]) and t["mv_slope_p"] < 0.05):
            why.append("the mean–sd relation is not significant")
        if np.isfinite(t["lambda_guerrero"]) and t["lambda_guerrero"] >= 0.75:
            why.append("the Guerrero λ is close to 1")
        N["transform"] = ("No variance-stabilising transformation is applied"
                          + (f" ({' and '.join(why)})" if why else "") + "; models work on the original scale.")

    # differencing
    dd = R["differencing"]
    N["diff"] = (f"Seasonal strength F<sub>S</sub> = {num(dd['F_seasonal'], 2)} "
                 f"{'>' if dd['D'] else '≤'} 0.64 → **D = {dd['D']}**; KPSS on the "
                 f"{'seasonally differenced ' if dd['D'] else ''}series → **d = {dd['d']}**"
                 + (" (KPSS suggested d = 1 but the extra difference increased the variance, "
                    "a sign of over-differencing, so it was dropped)" if dd["d_kpss"] != dd["d"] else "") + ".")

    # validation
    for tag in ("m1", "m2"):
        v = R["validation"][tag]
        lb_bad = [x["lag"] for x in v["ljungbox"] if x["p"] < 0.05]
        parts = [("causal (stationary)" if v["causal"] else "**not causal**"),
                 ("invertible" if v["invertible"] else "**not invertible**")]
        txt = f"The model is {' and '.join(parts)}: all AR/MA root moduli are " + (
            "greater than 1." if v["causal"] and v["invertible"] else "not all above 1.")
        if lb_bad:
            txt += (f" Ljung–Box rejects white noise at lag(s) {', '.join(map(str, lb_bad))}, so some "
                    "autocorrelation remains (common with long daily series; the forecast evaluation "
                    "below is what decides usefulness).")
        else:
            txt += " Ljung–Box does not reject white-noise residuals at any tested lag."
        txt += (" Residuals are " + ("approximately normal" if v["jb_p"] >= 0.05 else
                                     f"not normal (JB p = {pval(v['jb_p'])}, excess kurtosis "
                                     f"{num(v['kurtosis'], 2)}), so model-based intervals should be "
                                     "checked against their empirical coverage") + ".")
        txt += (" There is evidence of changing variance (ARCH LM p = " + pval(v["arch_p"]) + ")."
                if v["arch_p"] < 0.05 else " No evidence of conditional heteroskedasticity (ARCH LM).")
        st = v["stability"]
        mx = st["shift_in_se"].abs().max() if len(st) else 0
        txt += (f" Re-estimating without the last {R['cv']['h']} days moves the coefficients by at most "
                f"{num(mx, 2)} standard errors: the model is "
                + ("**stable**." if mx < 1 else "only moderately stable." if mx < 2 else "**not stable**."))
        N[f"val_{tag}"] = txt

    # CV
    lb = R["leaderboard"]
    best = R["best"]
    sn = lb.loc["Seasonal naive", "MASE"] if "Seasonal naive" in lb.index else np.nan
    gain = (1 - lb.loc[best, "MASE"] / sn) * 100 if np.isfinite(sn) else np.nan
    N["cv"] = (f"The most accurate method out of sample is **{best}** (MASE {num(lb.loc[best, 'MASE'], 3)}, "
               f"WAPE {num(lb.loc[best, 'WAPE %'], 1)} %), "
               + (f"{num(gain, 1)} % lower error than the seasonal naive benchmark." if np.isfinite(gain) else "."))
    sig = [d for d in R["dm"] if np.isfinite(d["p"]) and d["p"] < 0.05 and d["stat"] < 0]
    nsig = [d for d in R["dm"] if not (np.isfinite(d["p"]) and d["p"] < 0.05)]
    if sig:
        N["cv"] += (" The Diebold–Mariano test confirms it is significantly more accurate than "
                    + ", ".join(d["vs"] for d in sig) + ".")
    if nsig:
        N["cv"] += (" Its advantage over " + ", ".join(d["vs"] for d in nsig)
                    + " is not statistically significant, so those are practically equivalent alternatives.")
    if "SARIMA" in lb.index and "SARIMA (Model 2)" in lb.index:
        a, b = lb.loc["SARIMA", "MASE"], lb.loc["SARIMA (Model 2)", "MASE"]
        if abs(a - b) < 0.01:
            N["m1m2"] = (f"The two Box–Jenkins candidates forecast equally well (MASE {num(a, 3)} vs "
                         f"{num(b, 3)}), so the simpler one is preferable.")
        else:
            N["m1m2"] = (f"Between the two Box–Jenkins candidates, Model {'1' if a < b else '2'} forecasts "
                         f"better (MASE {num(min(a, b), 3)} vs {num(max(a, b), 3)}).")
    else:
        N["m1m2"] = ""
    covs = lb["Cov95 %"].dropna()
    if len(covs):
        under = [m for m, c in covs.items() if c < 90]
        N["coverage"] = ("Nominal 95 % intervals reach between "
                         f"{num(covs.min(), 0)} % and {num(covs.max(), 0)} % empirical coverage."
                         + (f" {', '.join(under)} under-cover (intervals too narrow)." if under else ""))
    else:
        N["coverage"] = ""

    # outliers
    oh = R["outlier_holdout"]
    w, wo = oh["With outlier treatment"]["MAE"], oh["Without outlier treatment"]["MAE"]
    N["outliers"] = (f"{len(R['outliers'])} outliers were detected on the full sample (|t| > "
                     f"{R['args']['outlier_crit']}). In the hold-out check (detection on the training part "
                     f"only, {R['outlier_holdout_n']} outliers) the treatment "
                     + (f"**improves** the {R['cv']['h']}-day forecast MAE from {num(wo)} to {num(w)}, so the "
                        "final SARIMA-type forecasts include the outlier regressors."
                        if R["use_outliers"] else
                        f"does not improve the forecast (MAE {num(wo)} without vs {num(w)} with), so the final "
                        "forecast does not use outlier regressors."))
    cal = R["calendar"]
    if cal["region"] == "none":
        N["calendar"] = "No holiday calendar was used (`--region none`)."
    else:
        N["calendar"] = (f"Public holidays ({cal['region'].title()} calendar, {cal['n_holidays_in_sample']} "
                         "in the sample) enter SARIMA, DHR and gradient boosting as a dummy. "
                         + (f"The store was closed on {num(cal['holiday_closed_rate'] * 100, 0)} % of holidays "
                            "falling on open weekdays, so future holidays are forecast as 0."
                            if cal["holiday_closed"] else
                            ("Holidays falling on open weekdays were closed only "
                             f"{num(cal['holiday_closed_rate'] * 100, 0)} % of the time, so their effect is "
                             "estimated rather than forced to 0."
                             if np.isfinite(cal["holiday_closed_rate"]) else "")))

    # forecast
    F = len(R["forecast_weekly"]) and sum(w_["days"] for w_ in R["forecast_weekly"])
    N["forecast"] = (f"Expected income over the next {F} days: **{num(R['forecast_total'])}** "
                     f"(80 % range of the daily sums: {num(R['forecast_total_lo'])} – {num(R['forecast_total_hi'])}), "
                     f"{num(R['forecast_vs_prev'], 1)} % vs the last {F} days")
    if R.get("forecast_vs_ly") is not None:
        N["forecast"] += f" and {num(R['forecast_vs_ly'], 1)} % vs the same period last year"
    N["forecast"] += "."
    N["intervals"] = ("Intervals are the model's own: their 95 % coverage in cross-validation was at least 85 %."
                      if R["interval_method"] == "model" else
                      "Intervals are **empirical**: quantiles of the cross-validation errors of this model at "
                      "each horizon, which guarantees realistic coverage even when residuals are not normal.")
    return N


# --------------------------------------------------------------------------- template
TEMPLATE = r"""
# {{ R.title }} — time-series analysis
{% if R.banner %}
> [!NOTE]
> {{ R.banner }}
{% endif %}
*Report generated automatically by `run_analysis.py` on {{ R.generated }} (run time {{ num(R.runtime_s) }} s).
Every number and sentence below is computed from the data; re-running the script refreshes the report.*

**Data:** {{ D.start }} → {{ D.end }} · {{ num(D.n_days) }} days ({{ D.n_years }} years) ·
**Horizon evaluated:** {{ R.cv.h }} days · **Best model:** {{ R.best }}

## Executive summary

- Mean income on an open day is **{{ num(S.mean_open) }}** (median {{ num(S.median_open) }},
  coefficient of variation {{ num(S.cv_open * 100, 0) }} %). Total income in the sample: {{ num(S.total) }}.
- {{ N.trend }}
- {{ N.weekly }} {{ N.monthly }}
- Seasonality: weekly strength {{ num(R.decomposition.F_weekly, 2) }}{% if R.decomposition.F_annual is defined %},
  annual strength {{ num(R.decomposition.F_annual, 2) }}{% endif %}, trend strength {{ num(R.decomposition.F_trend, 2) }}
  (0 = none, 1 = dominant).
- {{ N.cv }}
- {{ N.forecast }}

## Contents

1. [Data and cleaning](#1-data-and-cleaning)
2. [Exploratory analysis](#2-exploratory-analysis)
3. [Identification](#3-identification-box-jenkins)
4. [Estimation](#4-estimation)
5. [Validation](#5-validation-of-the-sarima-models)
6. [Calendar effects and outliers](#6-calendar-effects-and-outliers)
7. [Forecast evaluation](#7-forecast-evaluation-rolling-origin-cross-validation)
8. [Forecast](#8-forecast)
9. [Monthly view](#9-monthly-view-sarima-with-s--12)
10. [Conclusions](#10-conclusions-and-recommendations)
11. [Appendix: methods](#appendix-methods)

## 1. Data and cleaning

| Item | Value |
|:--|--:|
| Rows in file | {{ D.n_rows }} |
| Period | {{ D.start }} → {{ D.end }} |
| Calendar days | {{ D.n_days }} |
| Duplicate dates (summed) | {{ D.n_duplicate_dates }} |
| Missing calendar days | {{ D.n_missing_days }} |
| Imputed open days | {{ D.n_imputed }} |
| Zero-income days | {{ D.n_zero_days }} |
| Regular closing day(s) | {{ D.regular_closed_days | join(", ") or "none" }} |
| Irregular closures | {{ D.n_irregular_closures }} |
| Negative values | {{ D.n_negative }} |

{{ N.cleaning }}

## 2. Exploratory analysis

![Daily income](report/figures/01_series.png)

{{ N.trend }}

![Rolling mean and standard deviation](report/figures/02_rolling.png)

The rolling statistics show whether level and volatility drift over time: a moving mean
indicates trend (non-stationary in mean), a moving standard deviation that follows the mean
suggests a variance-stabilising transformation (section 3.1).

![Weekday and month profiles](report/figures/03_profiles.png)

{{ N.weekly }} {{ N.monthly }}

{{ dow_table }}

![Calendar heatmap](report/figures/04_heatmap.png)
{% if R.has_seasonal_years %}
![Seasonal plot by year](report/figures/05_seasonal_years.png)

{{ yearly_table }}
{% endif %}

## 3. Identification (Box–Jenkins)

### 3.1 Variance stabilisation

| Estimate | Value |
|:--|--:|
| Box–Cox λ (ML, open days) | {{ num(T.lambda_ml, 3) }} (95 % CI {{ num(T.lambda_ml_ci[0], 3) }} – {{ num(T.lambda_ml_ci[1], 3) }}) |
| Guerrero λ (monthly blocks) | {{ num(T.lambda_guerrero, 3) }} |
| Slope of log sd on log mean | {{ num(T.mv_slope, 3) }} (p = {{ pval(T.mv_slope_p) }}, {{ T.n_months_mv }} months) |

![Mean-dispersion plot](report/figures/06_mean_sd.png)

{{ N.transform }}
A slope of 0 means constant variance (no transform), 0.5 suggests √, 1 suggests log (λ ≈ 1 − slope).
The earlier notebook used Box–Cox on income + 1, which mixes closure zeros into the estimate;
here λ is estimated on open days only.

### 3.2 Stationarity and differencing

ADF tests H₀: unit root (non-stationary); KPSS tests H₀: stationary. A series is treated as
stationary when ADF rejects **and** KPSS does not. Following the course, the variance of each
differenced series is also reported: differencing should *decrease* it; an increase signals
over-differencing.

{{ stat_table }}

{{ N.diff }}

### 3.3 Decomposition

![Decomposition](report/figures/07_decomposition.png)

{{ R.decomposition.method }} decomposition (robust LOESS). Strength of trend
F<sub>T</sub> = {{ num(R.decomposition.F_trend, 2) }}, weekly seasonality
F<sub>S,7</sub> = {{ num(R.decomposition.F_weekly, 2) }}{% if R.decomposition.F_annual is defined %},
annual seasonality F<sub>S,365</sub> = {{ num(R.decomposition.F_annual, 2) }}{% endif %}.

### 3.4 Periodogram

![Periodogram](report/figures/08_periodogram.png)

Dominant periods (days): {{ peaks }}. A peak at 7 (and its harmonic 3.5) confirms the weekly
cycle that motivates s = 7.

### 3.5 ACF and PACF

![ACF and PACF](report/figures/09_acf_pacf.png)

Orange lines mark the seasonal lags (multiples of 7). In the levels the ACF decays slowly at
seasonal lags (seasonal non-stationarity). After ∇₇ the pattern of significant lags at 7, 14, …
in ACF vs PACF identifies the seasonal part (a single ACF spike at 7 → seasonal MA(1); decaying
PACF at 7, 14 → seasonal MA), and the first lags identify the regular part.

### 3.6 Anomalous days

{{ R.n_anomalies }} open days have a decomposition remainder with robust |z| > 3.5.
{% if R.anomalies %}
| Date | Weekday | Income | Robust z | Holiday |
|:--|:--|--:|--:|:--|
{% for a in R.anomalies %}| {{ a.date }} | {{ a.weekday }} | {{ num(a.income) }} | {{ num(a.z, 1) }} | {{ a.holiday }} |
{% endfor %}{% endif %}

## 4. Estimation

The orders (p, q)(P, Q) were selected by AICc over a grid of {{ R.sarima_grid_size }} models with
d = {{ R.differencing.d }}, D = {{ R.differencing.D }}, s = 7, using **only data before the first
cross-validation origin** ({{ R.cv.first_origin }}), so the later evaluation is genuinely out of
sample. Holiday and closure dummies are included as regressors (regression with SARIMA errors).

{{ grid_table }}

- **Model 1** (lowest AICc): {{ m1 }}
- **Model 2** (lowest BIC among the rest, the parsimonious alternative): {{ m2 }}

Coefficients estimated on the full sample:

**Model 1**

{{ coef_m1 }}

**Model 2**

{{ coef_m2 }}

**Other models.** ETS: additive error and seasonality, {{ "damped additive trend" if ets_damped else "no trend" }}
(chosen by AICc: {{ ets_aicc }}). Dynamic harmonic regression (DHR):
{% if R.fourier.K %}Model 1 errors plus K = {{ R.fourier.K }} annual Fourier pairs (K chosen by AICc){% else %}not used (needs > 400 days to estimate the annual cycle){% endif %}.
Gradient boosting: direct multi-horizon model on calendar features and lags ≥ horizon, with
quantile-loss models for the intervals.

## 5. Validation of the SARIMA models
{% for tag in ["m1", "m2"] %}{% set v = R.validation[tag] %}
### Model {{ tag[1] }}: {{ m1 if tag == "m1" else m2 }}

![Residual analysis](report/figures/10_diagnostics_{{ tag }}.png)

| Check | Result |
|:--|--:|
| AIC / BIC / AICc | {{ num(v.aic, 1) }} / {{ num(v.bic, 1) }} / {{ num(v.aicc, 1) }} |
| AR root moduli (min) | {{ num(v.ar_root_mod[0], 3) if v.ar_root_mod else "–" }} |
| MA root moduli (min) | {{ num(v.ma_root_mod[0], 3) if v.ma_root_mod else "–" }} |
| Ljung–Box p (lags {{ v.ljungbox | map(attribute="lag") | join(", ") }}) | {{ v.ljungbox | map(attribute="p") | map("pval") | join(", ") }} |
| Shapiro–Wilk p / Jarque–Bera p | {{ pval(v.shapiro_p) }} / {{ pval(v.jb_p) }} |
| Skewness / excess kurtosis | {{ num(v.skew, 2) }} / {{ num(v.kurtosis, 2) }} |
| ARCH LM p (7 lags) / break-variance p | {{ pval(v.arch_p) }} / {{ pval(v.breakvar_p) }} |

{{ N["val_" + tag] }}

![psi and pi weights](report/figures/11_weights_{{ tag }}.png)

The ψ weights (MA(∞) representation) show how a shock propagates into future income; the π weights
(AR(∞) representation) show how much each past value enters the one-step forecast. Both decay to
zero, as required for a causal and invertible model.

**Stability** (coefficients with all data vs without the last {{ R.cv.h }} days):

{{ stab[tag] }}
{% endfor %}

## 6. Calendar effects and outliers

{{ N.calendar }}

![Outliers](report/figures/12_outliers.png)
{% if R.outliers %}
| Date | Type | Estimated effect | t |
|:--|:--|--:|--:|
{% for o in R.outliers %}| {{ o.date }} | {{ o.type }} | {{ num(o.effect, 1) }} | {{ num(o.z, 1) }} |
{% endfor %}{% endif %}

{{ N.outliers }}

| Hold-out ({{ R.cv.h }} days) | MAE | RMSE |
|:--|--:|--:|
{% for k, v in R.outlier_holdout.items() %}| {{ k }} | {{ num(v.MAE, 1) }} | {{ num(v.RMSE, 1) }} |
{% endfor %}

## 7. Forecast evaluation (rolling-origin cross-validation)

Each model is re-estimated on an expanding window and forecasts the next {{ R.cv.h }} days, for
{{ R.cv.folds }} consecutive origins ({{ R.cv.first_origin }} → {{ R.cv.last_origin }}). This
tests the models the way they will be used, instead of judging them only by in-sample fit.
Irregular closures inside the test windows ({{ R.cv.n_excluded }} days) are not scored, because no
model can anticipate them. MAPE/RMSPE use days with positive income only.

{{ leaderboard }}

MASE < 1 means more accurate than the in-sample seasonal naive forecast. Cov80/Cov95 are
empirical coverages of the 80 % / 95 % intervals; Width95 is the mean width of the 95 % interval
(the course's precision measure).

{{ N.cv }} {{ N.m1m2 }} {{ N.coverage }}

**Diebold–Mariano tests** (squared loss, HLN correction; negative = {{ R.best }} better):

| {{ R.best }} vs | DM statistic | p-value |
|:--|--:|--:|
{% for d in R.dm %}| {{ d.vs }} | {{ num(d.stat, 2) }} | {{ pval(d.p) }} |
{% endfor %}

![Error by horizon](report/figures/13_error_by_horizon.png)

![Last fold](report/figures/14_last_fold.png)

## 8. Forecast

![Forecast](report/figures/15_forecast.png)

{{ N.forecast }} {{ N.intervals }}

| Week ending | Days | Forecast income |
|:--|--:|--:|
{% for w in R.forecast_weekly %}| {{ w.week_ending }} | {{ w.days }} | {{ num(w.forecast) }} |
{% endfor %}

Total over the horizon according to each model (a quick check on model agreement):
{% for k, v in R.forecast_alt.items() %}{{ k }} {{ num(v) }}{{ " · " if not loop.last }}{% endfor %}.

The daily values are written to `outputs/forecast.csv` (not committed).

## 9. Monthly view (SARIMA with s = 12)
{% if M.available %}
The classic course workflow on monthly totals ({{ M.n_months }} complete months), with the number
of open days in each month as a trading-day regressor. Seasonal strength
F<sub>S,12</sub> = {{ num(M.F_seasonal, 2) }} → D = {{ M.D }}; d = {{ M.d }}. The transformation
(none / log) and orders were chosen by AIC computed on the original scale (Jacobian-corrected).

Selected: **{{ msel }}**{{ " on log(income)" if M.transform == "log" else "" }}.
{% if M.trading_day %}Trading-day coefficient: {{ num(M.trading_day.coef, 2) }} per open day
(p = {{ pval(M.trading_day.p) }}).{% endif %} Ljung–Box p (lag 24) = {{ pval(M.ljungbox_p24) }}.

| Hold-out last 12 months | MAE | RMSE | MAPE % |
|:--|--:|--:|--:|
{% for k, v in M.holdout.items() %}| {{ k }} | {{ num(v.MAE) }} | {{ num(v.RMSE) }} | {{ num(v["MAPE %"], 1) }} |
{% endfor %}

![Monthly forecast](report/figures/monthly_forecast.png)

{{ monthly_fc }}
{% else %}
Skipped: needs at least 36 complete months (the data have {{ M.n_months }}). Re-run when more
history is available.
{% endif %}

## 10. Conclusions and recommendations

1. **Structure.** Income is dominated by the weekly cycle (F<sub>S,7</sub> = {{ num(R.decomposition.F_weekly, 2) }}).
   {{ N.weekly }} {{ N.trend }}
2. **Model.** {{ N.cv }} Use it for the operational {{ R.cv.h }}-day forecast; the ensemble and the
   SARIMA are robust fall-backs.
3. **Uncertainty.** {{ N.intervals }} Plan staffing and stock against the 80 % band, not the point
   forecast alone.
4. **Keep it updated.** Append new days to the CSV and re-run `python3 run_analysis.py`; orders are
   re-selected and all models re-evaluated automatically.
5. **Possible improvements.** Add a file of *known future closures* (vacations) and promotions as
   regressors; add weather or local events if they drive footfall; model the share of card vs
   cash payments if available.

## Appendix: methods

| Method | Role | Reference idea |
|:--|:--|:--|
| Box–Cox (ML) and Guerrero λ, mean–sd plot | Variance stabilisation | Box & Cox 1964; Guerrero 1993 |
| ADF, KPSS, variance of differences | Choice of d, D | Dickey–Fuller; Kwiatkowski et al. 1992 |
| STL / MSTL, seasonal strength | Decomposition, choice of D | Cleveland et al. 1990; Wang, Smith & Hyndman 2006 |
| Periodogram | Detect cycles | Spectral analysis |
| SARIMA with regressors | Box–Jenkins model (two candidates) | Box & Jenkins |
| Root moduli, ψ/π weights | Causality, invertibility | — |
| Ljung–Box, Shapiro–Wilk, Jarque–Bera, ARCH LM, √\|e\| smoother | Residual validation | — |
| AO / LS / TC detection | Outlier treatment (iterative, π-weighted) | Chen & Liu 1993 |
| Holiday dummies | Calendar effects | — |
| Dynamic harmonic regression | Annual cycle with Fourier terms + SARIMA errors | Hyndman & Athanasopoulos, FPP3 |
| ETS (A, Ad, A) | Exponential smoothing state-space model | Hyndman et al. 2008 |
| Gradient boosting (direct, quantile loss) | Machine-learning comparison | Friedman 2001 |
| Ensemble (mean of ETS, SARIMA, DHR, GBM) | Forecast combination | Bates & Granger 1969 |
| Rolling-origin CV, MASE, WAPE, coverage | Out-of-sample evaluation | Tashman 2000; Hyndman & Koehler 2006 |
| Diebold–Mariano (HLN) | Significance of accuracy differences | Diebold & Mariano 1995 |
"""


def render(R):
    N = _narrative(R)
    env = jinja2.Environment(undefined=jinja2.ChainableUndefined, trim_blocks=False)
    env.filters["pval"] = pval
    v1, v2 = R["validation"]["m1"], R["validation"]["m2"]
    st = pd.DataFrame(R["stationarity"]["table"]).set_index("series")
    st.columns = ["Variance", "ADF stat", "ADF p", "KPSS stat", "KPSS p"]
    grid = R["sarima_grid"].copy()
    grid["model"] = [sarima_name(o, s) for o, s in zip(grid.order, grid.sorder)]
    grid = grid.set_index("model")[["k", "AIC", "BIC", "AICc"]]
    lb = R["leaderboard"].copy()
    lb_cols = ["MASE", "MASE sd (folds)", "MAE", "RMSE", "WAPE %", "MAPE %", "RMSPE %", "Bias",
               "Cov80 %", "Cov95 %", "Width95", "Fit time (s/fold)"]
    dows = pd.DataFrame({"Mean income (open days)": R["dow_means"]})
    yearly = pd.DataFrame(R["summary"]["yearly"]).set_index("year")
    yearly.columns = ["Total", "Mean daily", "Days", "Complete year"]
    M = R["monthly"]
    ctx = dict(
        R=R, N=N, D=R["data"], S=R["summary"], T=R["transform"], M=M, num=num, pval=pval,
        m1=sarima_name(R["m1"]["order"], R["m1"]["sorder"]),
        m2=sarima_name(R["m2"]["order"], R["m2"]["sorder"]),
        stat_table=md_table(st, {"Variance": 1, "ADF stat": 2, "ADF p": pval, "KPSS stat": 3,
                                 "KPSS p": pval}, index_name="Series"),
        grid_table=md_table(grid, {"k": lambda v: str(int(v)), "AIC": 1, "BIC": 1, "AICc": 1}, index_name="Model"),
        coef_m1=md_table(v1["coefs"], {"coef": 4, "se": 4, "z": 2, "p": pval}, index_name="Parameter"),
        coef_m2=md_table(v2["coefs"], {"coef": 4, "se": 4, "z": 2, "p": pval}, index_name="Parameter"),
        stab={t: md_table(R["validation"][t]["stability"],
                          {"full": 4, "full_se": 4, "without_last_h": 4, "shift_in_se": 2},
                          index_name="Parameter") for t in ("m1", "m2")},
        leaderboard=md_table(lb[lb_cols], {"MASE": 3, "MASE sd (folds)": 3, "MAE": 1, "RMSE": 1,
                                           "WAPE %": 1, "MAPE %": 1, "RMSPE %": 1, "Bias": 1,
                                           "Cov80 %": 0, "Cov95 %": 0, "Width95": 0,
                                           "Fit time (s/fold)": 1}, index_name="Model"),
        dow_table=md_table(dows, {"Mean income (open days)": 0}, index_name="Weekday"),
        yearly_table=md_table(yearly, {"Total": 0, "Mean daily": 1, "Days": lambda v: str(int(v)), "Complete year":
                                       lambda b: "yes" if b else "no"}, index_name="Year"),
        peaks=", ".join(num(x, 1) for x in R["periodogram"]["peaks"]) or "none",
        ets_damped=R["args"] and R.get("ets_aicc") and min(R["ets_aicc"], key=R["ets_aicc"].get) == "A,Ad,A",
        ets_aicc=", ".join(f"{k} {num(v, 1)}" for k, v in R["ets_aicc"].items()),
    )
    if M.get("available"):
        ctx["msel"] = sarima_name(M["order"], M["sorder"])
        mf = M["forecast"].copy()
        mf.index = mf.index.strftime("%Y-%m")
        ctx["monthly_fc"] = md_table(mf[["mean", "lo80", "hi80", "lo95", "hi95"]],
                                     {c: 0 for c in mf.columns}, index_name="Month")
    md = env.from_string(TEMPLATE).render(**ctx)
    while "\n\n\n" in md:
        md = md.replace("\n\n\n", "\n\n")
    return md.strip() + "\n"

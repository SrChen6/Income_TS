#!/usr/bin/env python3
"""Full time-series analysis of the store's daily income -> REPORT.md + figures.

Usage:
    python3 run_analysis.py                      # uses income_data.csv
    python3 run_analysis.py --data other.csv --horizon 28 --folds 6 --region catalonia
    python3 run_analysis.py --fast               # smaller SARIMA grid, fewer folds

Outputs:
    REPORT.md               the report (GitHub renders it, figures included)
    report/figures/*.png    figures used by the report
    outputs/                forecast.csv, cv_forecasts.csv, results.json (git-ignored)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from income_ts import data, eda, evaluation, models, monthly, plots, report, style
from income_ts.calendar import holiday_names

ROOT = Path(__file__).resolve().parent


def log(msg):
    print(msg, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "income_data.csv"))
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--value-col", default="Income")
    ap.add_argument("--dayfirst", action="store_true", help="parse dates as DD/MM/YYYY")
    ap.add_argument("--horizon", type=int, default=28, help="forecast horizon evaluated in CV (days)")
    ap.add_argument("--folds", type=int, default=6, help="rolling-origin CV folds")
    ap.add_argument("--forecast-days", type=int, default=None, help="final forecast length (default: horizon)")
    ap.add_argument("--region", default="catalonia", choices=["spain", "catalonia", "barcelona", "none"],
                    help="public-holiday calendar used as calendar effect")
    ap.add_argument("--outlier-crit", type=float, default=3.5, help="|t| threshold for AO/LS/TC detection")
    ap.add_argument("--out", default=str(ROOT), help="where REPORT.md, report/ and outputs/ go")
    ap.add_argument("--title", default="Daily income of a local store")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--banner", default="", help="note shown at the top of the report (e.g. 'synthetic data')")
    a = ap.parse_args(argv)

    t_start = time.time()
    out = Path(a.out)
    fig_dir = out / "report" / "figures"
    res_dir = out / "outputs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    style.apply()
    h = a.horizon
    F = a.forecast_days or h
    folds = 3 if a.fast else a.folds
    R = {"title": a.title, "banner": a.banner, "args": vars(a), "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

    # ------------------------------------------------------------------ 1. data
    log("1/9 Loading and cleaning data")
    raw, load_info = data.load_raw(a.data, a.date_col, a.value_col, a.dayfirst)
    p = data.prepare(raw)
    R["data"] = {**load_info, **p.meta}
    if len(p.y) < 120:
        sys.exit("Need at least ~4 months of daily data for this analysis.")

    # ------------------------------------------------------------------ 2. EDA
    log("2/9 Exploratory analysis")
    R["summary"] = eda.summary(p)
    eda.fig_overview(p, fig_dir / "01_series.png")
    eda.fig_rolling(p, fig_dir / "02_rolling.png")
    eda.fig_profiles(p, fig_dir / "03_profiles.png")
    eda.fig_heatmap(p, fig_dir / "04_heatmap.png")
    R["has_seasonal_years"] = p.y.index.year.nunique() >= 2 and len(p.y) >= 400
    if R["has_seasonal_years"]:
        eda.fig_seasonal_years(p, fig_dir / "05_seasonal_years.png")
    dow_means = p.y[p.open_mask].groupby(p.y[p.open_mask].index.dayofweek).mean()
    R["dow_means"] = {data.WEEKDAYS[int(k)]: float(v) for k, v in dow_means.items()}
    mon_means = p.y.groupby(p.y.index.month).mean()
    R["month_means"] = {eda.MONTHS[int(k) - 1]: float(v) for k, v in mon_means.items()}

    # ------------------------------------------------------------------ 3. identification
    log("3/9 Identification: transformation, stationarity, decomposition")
    R["transform"] = eda.transformation(p, fig_dir / "06_mean_sd.png")
    kind = R["transform"]["choice"]
    w = pd.Series(eda.apply_transform(p.y_clean, kind), index=p.y.index)
    R["stationarity"] = eda.stationarity(w, 7)
    R["differencing"] = eda.choose_differencing(w, 7)
    d, D = R["differencing"]["d"], R["differencing"]["D"]
    dec = eda.decomposition(p, fig_dir / "07_decomposition.png")
    R["decomposition"] = {k: v for k, v in dec.items() if k != "resid"}
    R["periodogram"] = eda.periodogram(p, fig_dir / "08_periodogram.png")
    acf_series = {"W_t": w, "∇₇W_t": w.diff(7), "∇∇₇W_t": w.diff(7).diff()}
    eda.fig_acf_pacf(acf_series, fig_dir / "09_acf_pacf.png")
    an = eda.anomalies(p, dec["resid"])
    hn = holiday_names(an.index, a.region) if len(an) else pd.Series(dtype=str)
    R["anomalies"] = [{"date": i.date().isoformat(), "weekday": data.WEEKDAYS[i.dayofweek],
                       "income": float(r.income), "z": float(r.robust_z), "holiday": hn.get(i, "")}
                      for i, r in an.head(12).iterrows()]
    R["n_anomalies"] = int(len(an))

    # ------------------------------------------------------------------ 4. estimation
    log("4/9 Estimation: SARIMA order selection")
    min_train = max(180, 4 * h)
    max_folds = max(1, (len(p.y) - min_train) // h)
    folds = min(folds, max_folds)
    cv_origins = evaluation.origins(p.y.index, h, folds, min_train)
    sel_end = cv_origins[0]  # orders chosen with data before the first CV origin -> no leakage
    ctx = models.Context(p=p, region=a.region, transform=kind, d=d, D=D,
                         sarima_sorder=(0, D, 1, 7))
    R["calendar"] = {"region": a.region, "holiday_closed": ctx.holiday_closed,
                     "holiday_closed_rate": ctx.holiday_closed_rate,
                     "n_holidays_in_sample": int(ctx.holiday.reindex(p.y.index).sum())}
    tab = models.select_sarima(ctx, train_end=sel_end, fast=a.fast)
    (o1, s1), (o2, s2) = models.pick_two_models(tab)
    ctx.sarima_order, ctx.sarima_sorder = o1, s1
    R["sarima_grid"] = tab.head(10)
    R["sarima_grid_size"] = int(len(tab))
    R["m1"], R["m2"] = {"order": o1, "sorder": s1}, {"order": o2, "sorder": s2}
    ctx.ets_damped, R["ets_aicc"] = models.select_ets(ctx, sel_end)
    K, krows = models.select_fourier_K(ctx, sel_end)
    ctx.fourier_K, R["fourier"] = K, {"K": K, "grid": krows}
    log(f"   Model 1 SARIMA{o1}{s1}, Model 2 SARIMA{o2}{s2}, ETS damped={ctx.ets_damped}, Fourier K={K}")

    # ------------------------------------------------------------------ 5. validation
    log("5/9 Validation of the two SARIMA candidates")
    last = p.y.index[-1]
    nxt = pd.date_range(last + pd.Timedelta(days=1), periods=1, freq="D")
    R["validation"] = {}
    for tag, (o, so) in (("m1", (o1, s1)), ("m2", (o2, s2))):
        _, res = models.sarima_forecast(ctx, last, nxt, o, so, return_res=True)
        val = models.validate(res, o, so)
        plots.fig_diagnostics(val, fig_dir / f"10_diagnostics_{tag}.png",
                              f"Residual analysis - Model {tag[1]}: SARIMA{o}{so}")
        plots.fig_psi(val, fig_dir / f"11_weights_{tag}.png", f"Model {tag[1]}: SARIMA{o}{so}")
        val["stability"] = models.stability(ctx, o, so, h)
        R["validation"][tag] = val

    # ------------------------------------------------------------------ 6. outliers
    log("6/9 Outlier detection (AO / LS / TC)")
    found, otab, ores = models.detect_outliers(ctx, crit=a.outlier_crit, max_iter=6 if a.fast else 12)
    R["outliers"] = otab
    lin = models.linearized(ctx, ores, found) if found else p.y
    plots.fig_outliers(p.y, lin, otab, fig_dir / "12_outliers.png")
    # course-style check: does treating outliers improve the hold-out forecast?
    cut = p.y.index[-h - 1]
    hold_idx = p.y.index[-h:]
    found_tr, _, _ = models.detect_outliers(ctx, train_end=cut, crit=a.outlier_crit,
                                            max_iter=6 if a.fast else 12)
    f_plain = models.sarima_forecast(ctx, cut, hold_idx, outliers=[])
    f_out = models.sarima_forecast(ctx, cut, hold_idx, outliers=found_tr)
    ev = ~ctx.closed_nh.reindex(hold_idx).astype(bool)
    act = p.y[hold_idx]
    R["outlier_holdout"] = {
        name: {"MAE": float((act - f["mean"])[ev].abs().mean()),
               "RMSE": float(np.sqrt(((act - f["mean"])[ev] ** 2).mean()))}
        for name, f in (("Without outlier treatment", f_plain), ("With outlier treatment", f_out))
    }
    R["outlier_holdout_n"] = len(found_tr)
    use_outliers = (R["outlier_holdout"]["With outlier treatment"]["MAE"]
                    < R["outlier_holdout"]["Without outlier treatment"]["MAE"]) and bool(found)
    R["use_outliers"] = use_outliers

    # ------------------------------------------------------------------ 7. cross-validation
    log(f"7/9 Rolling-origin cross-validation ({folds} folds x {h} days)")
    reg = models.registry(ctx)
    reg["SARIMA (Model 2)"] = lambda c, te, fi: models.sarima_forecast(c, te, fi, o2, s2)
    style.MODEL_COLORS.setdefault("SARIMA (Model 2)", style.RED)
    cv, timings = evaluation.run_cv(ctx, reg, h, folds, min_train, log=log)
    cv.to_csv(res_dir / "cv_forecasts.csv", index=False)
    lb = evaluation.leaderboard(cv)
    lb["Fit time (s/fold)"] = [timings.get(m, np.nan) / folds for m in lb.index]
    R["leaderboard"] = lb
    R["cv"] = {"folds": int(cv.fold.max()), "h": h, "first_origin": cv_origins[0].date().isoformat(),
               "last_origin": cv_origins[-1].date().isoformat(),
               "n_excluded": int((~cv[cv.model == "SARIMA"].evaluated).sum())}
    best = lb.index[0]
    R["best"] = best
    dm = []
    for other in ("Seasonal naive", "SARIMA", "ETS", "Gradient boosting"):
        if other != best and other in lb.index:
            st, pv = evaluation.diebold_mariano(cv, best, other, h)
            dm.append({"vs": other, "stat": st, "p": pv})
    R["dm"] = dm
    top = [m for m in lb.index if m not in ("Naive", "Seasonal mean")][:5]
    if "Seasonal naive" not in top:
        top.append("Seasonal naive")
    evaluation.fig_error_by_step(cv, fig_dir / "13_error_by_horizon.png", top)
    evaluation.fig_last_fold(cv, ctx, fig_dir / "14_last_fold.png", top[:4])

    # ------------------------------------------------------------------ 8. final forecast
    log(f"8/9 Final {F}-day forecast with {best}")
    fut = pd.date_range(last + pd.Timedelta(days=1), periods=F, freq="D")
    if use_outliers:
        ctx.outliers = found
    finals = {}
    for m in set([best, "SARIMA", "ETS", "Gradient boosting"] + list(models.ENSEMBLE_MEMBERS)):
        if m in reg:
            finals[m] = reg[m](ctx, last, fut)
    finals["Ensemble"] = models.ensemble(finals)
    fc = finals[best].copy()
    cov = lb.loc[best, "Cov95 %"]
    R["interval_method"] = "model"
    if not np.isfinite(cov) or cov < 85:  # no model intervals, or they under-cover in CV
        q = evaluation.empirical_intervals(cv, best, h).reindex(range(1, F + 1)).ffill()
        fc["lo95"] = fc["mean"] + q[0.025].to_numpy()
        fc["lo80"] = fc["mean"] + q[0.1].to_numpy()
        fc["hi80"] = fc["mean"] + q[0.9].to_numpy()
        fc["hi95"] = fc["mean"] + q[0.975].to_numpy()
        fc = ctx.postprocess(fc)
        R["interval_method"] = "empirical"
    fc.rename_axis("Date").to_csv(res_dir / "forecast.csv")
    plots.fig_forecast(p.y, fc, fig_dir / "15_forecast.png",
                       f"{F}-day forecast - {best} (80 % / 95 % intervals)")
    weekly = fc["mean"].resample("W-SUN").agg(["sum", "count"])
    R["forecast_weekly"] = [{"week_ending": i.date().isoformat(), "days": int(r["count"]),
                             "forecast": float(r["sum"])} for i, r in weekly.iterrows()]
    R["forecast_total"] = float(fc["mean"].sum())
    R["forecast_total_lo"] = float(fc["lo80"].sum())
    R["forecast_total_hi"] = float(fc["hi80"].sum())
    prev = p.y.iloc[-F:].sum()
    R["forecast_vs_prev"] = float((fc["mean"].sum() / prev - 1) * 100) if prev else np.nan
    ly = p.y.reindex(fut - pd.Timedelta(days=364))
    R["forecast_vs_ly"] = (float((fc["mean"].sum() / ly.sum() - 1) * 100)
                           if ly.notna().all() and ly.sum() else None)
    R["forecast_alt"] = {m: float(f["mean"].sum()) for m, f in finals.items()}

    # ------------------------------------------------------------------ 9. monthly + report
    log("9/9 Monthly SARIMA (s = 12) and report")
    R["monthly"] = monthly.analyse(ctx, fig_dir)
    R["runtime_s"] = time.time() - t_start
    md = report.render(R)
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    (res_dir / "results.json").write_text(json.dumps(report.jsonable(R), indent=2, default=str))
    log(f"Done in {R['runtime_s']:.0f}s -> {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()

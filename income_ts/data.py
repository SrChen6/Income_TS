"""Loading and cleaning: build a complete, regular daily series.

Rules (all counts are reported so nothing happens silently):
* duplicate dates are summed;
* the calendar is completed to one row per day;
* a weekday that is zero/missing on >= ``closed_threshold`` of its occurrences is a
  *regular closing day* (e.g. Sundays): its missing days become 0 and its forecasts
  are forced to 0;
* a zero on a normally-open weekday is an *irregular closure* (holiday, vacation...).
  It is kept as 0 in ``y`` and flagged in ``closed`` so regression models can absorb
  it with a dummy; ``y_clean`` replaces it by a seasonal imputation for models
  without exogenous inputs;
* any other missing day is imputed from the same weekday in neighbouring weeks.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class Prepared:
    y: pd.Series            # daily income, complete (closures are 0)
    y_clean: pd.Series      # same, irregular closures replaced by imputation
    closed: pd.Series       # 1 = irregular closure day, else 0
    imputed: pd.Series      # True where the value was missing and imputed
    regular_closed_dows: list
    meta: dict = field(default_factory=dict)

    @property
    def open_mask(self) -> pd.Series:
        """Days on which the store is expected to be open."""
        reg = np.isin(self.y.index.dayofweek, self.regular_closed_dows)
        return pd.Series(~reg, index=self.y.index) & self.closed.eq(0)


def load_raw(path, date_col="Date", value_col="Income", dayfirst=False):
    df = pd.read_csv(path)
    missing_cols = [c for c in (date_col, value_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found; available: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col], dayfirst=dayfirst).dt.normalize()
    values = pd.to_numeric(df[value_col], errors="coerce")
    info = {
        "n_rows": int(len(df)),
        "n_duplicate_dates": int(dates.duplicated().sum()),
        "n_unparseable_values": int(values.isna().sum() - df[value_col].isna().sum()),
        "n_blank_values": int(df[value_col].isna().sum()),
    }
    s = pd.Series(values.to_numpy(dtype=float), index=dates)
    s = s.groupby(level=0).sum(min_count=1).sort_index()
    return s, info


def _seasonal_impute(s: pd.Series, need: pd.Series, open_only: pd.Series) -> pd.Series:
    """Mean of the same weekday 1-2 weeks before/after (open, observed days only)."""
    ref = s.where(open_only & s.notna())
    cands = pd.concat([ref.shift(k) for k in (7, -7, 14, -14, 21, -21)], axis=1)
    est = cands.mean(axis=1)
    dow_median = ref.groupby(ref.index.dayofweek).median()
    fallback = pd.Series(s.index.dayofweek, index=s.index).map(dow_median)
    est = est.fillna(fallback).fillna(ref.median())
    return est.where(need)


def prepare(raw: pd.Series, closed_threshold: float = 0.8) -> Prepared:
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq="D", name="Date")
    s = raw.reindex(idx)
    missing = s.isna()
    zero = s.eq(0)
    dow = pd.Series(idx.dayofweek, index=idx)

    closed_rate = (missing | zero).groupby(dow).mean()
    regular = [int(d) for d, r in closed_rate.items() if r >= closed_threshold]
    regular_mask = dow.isin(regular)

    irregular_closed = zero & ~regular_mask
    to_impute = missing & ~regular_mask

    y = s.copy()
    y[regular_mask & missing] = 0.0
    imputation = _seasonal_impute(s, to_impute, ~regular_mask & ~zero)
    y[to_impute] = imputation[to_impute]

    y_clean = y.copy()
    clean_fill = _seasonal_impute(y.where(~irregular_closed), irregular_closed,
                                  ~regular_mask & ~irregular_closed)
    y_clean[irregular_closed] = clean_fill[irregular_closed]

    y = y.asfreq("D").rename("Income")
    y_clean = y_clean.asfreq("D").rename("Income")
    meta = {
        "start": idx[0].date().isoformat(),
        "end": idx[-1].date().isoformat(),
        "n_days": int(len(idx)),
        "n_weeks": round(len(idx) / 7, 1),
        "n_years": round(len(idx) / 365.25, 2),
        "n_missing_days": int(missing.sum()),
        "n_imputed": int(to_impute.sum()),
        "n_zero_days": int(zero.sum()),
        "n_irregular_closures": int(irregular_closed.sum()),
        "n_negative": int((s < 0).sum()),
        "regular_closed_days": [WEEKDAYS[d] for d in regular],
        "closed_rate_by_dow": {WEEKDAYS[int(d)]: round(float(r), 3) for d, r in closed_rate.items()},
    }
    return Prepared(
        y=y,
        y_clean=y_clean,
        closed=irregular_closed.astype(int).rename("closed"),
        imputed=to_impute.rename("imputed"),
        regular_closed_dows=regular,
        meta=meta,
    )

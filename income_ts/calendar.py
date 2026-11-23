"""Calendar effects: public holidays (computed, no external package needed).

Daily analogue of the course's trading-day / Easter regressors (CalendarEffects.r):
a store's income on a public holiday is not explained by the weekly pattern, so
holidays enter the regression models as a known-in-advance dummy, which, unlike
an unexplained closure, can also be set for future dates.
"""

import datetime as dt

import pandas as pd

# (month, day, name)
_SPAIN_FIXED = [
    (1, 1, "New Year"), (1, 6, "Epiphany"), (5, 1, "Labour Day"),
    (8, 15, "Assumption"), (10, 12, "National Day"), (11, 1, "All Saints"),
    (12, 6, "Constitution Day"), (12, 8, "Immaculate Conception"), (12, 25, "Christmas"),
]
_CATALONIA_EXTRA = [(6, 24, "Sant Joan"), (9, 11, "Diada"), (12, 26, "Sant Esteve")]
_BARCELONA_EXTRA = [(9, 24, "La Merce")]


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def holidays(years, region: str = "catalonia") -> dict:
    """Return {date: name} for the given years. region: spain|catalonia|barcelona|none."""
    if region == "none":
        return {}
    fixed = list(_SPAIN_FIXED)
    if region in ("catalonia", "barcelona"):
        fixed += _CATALONIA_EXTRA
    if region == "barcelona":
        fixed += _BARCELONA_EXTRA
    out = {}
    for y in years:
        for m, d, name in fixed:
            out[dt.date(y, m, d)] = name
        e = easter_sunday(y)
        out[e - dt.timedelta(days=2)] = "Good Friday"
        if region in ("catalonia", "barcelona"):
            out[e + dt.timedelta(days=1)] = "Easter Monday"
    return out


def holiday_series(index: pd.DatetimeIndex, region: str = "catalonia") -> pd.Series:
    hol = holidays(range(index.min().year, index.max().year + 1), region)
    days = pd.to_datetime(sorted(hol))
    return pd.Series(index.isin(days).astype(int), index=index, name="holiday")


def holiday_names(index: pd.DatetimeIndex, region: str = "catalonia") -> pd.Series:
    hol = holidays(range(index.min().year, index.max().year + 1), region)
    return pd.Series([hol.get(d.date(), "") for d in index], index=index)

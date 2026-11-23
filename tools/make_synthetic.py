#!/usr/bin/env python3
"""Generate a synthetic store-income CSV (same schema as income_data.csv) to test the
pipeline without real data:  python3 tools/make_synthetic.py out.csv --years 3.2"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from income_ts.calendar import holiday_series  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--years", type=float, default=3.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="2022-09-01")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    idx = pd.date_range(a.start, periods=int(a.years * 365.25), freq="D")
    n = len(idx)
    t = np.arange(n)
    level = 900 + 0.12 * t
    level[int(n * 0.6):] += 120                              # a level shift
    annual = 1 + 0.15 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 80) / 365.25)
    weekly = np.array([0.85, 0.9, 0.95, 1.0, 1.25, 1.45, 0.0])[idx.dayofweek.to_numpy()]  # closed Sundays
    ar = np.zeros(n)
    eps = rng.normal(0, 0.08, n)
    for i in range(1, n):
        ar[i] = 0.5 * ar[i - 1] + eps[i]
    y = level * annual * weekly * (1 + ar)
    hol = holiday_series(idx, "catalonia").to_numpy().astype(bool)
    y[hol] = 0
    vac = (idx.month == 8) & (idx.day >= 10) & (idx.day <= 20)   # summer vacation
    y[vac] = 0
    for k in rng.choice(n, 4, replace=False):                     # additive outliers
        if y[k] > 0:
            y[k] *= 2.2
    y = np.round(np.clip(y, 0, None), 2)
    df = pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Income": y})
    drop = rng.choice(n, 6, replace=False)                        # missing rows
    df = df.drop(index=drop)
    df = df.sample(frac=1, random_state=a.seed)                   # unsorted, like real exports
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")


if __name__ == "__main__":
    main()

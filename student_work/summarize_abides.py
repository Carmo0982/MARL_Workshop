"""Summarize ABIDES evaluation CSVs (from abides_eval.py) into one table.

    .venv/bin/python -m student_work.summarize_abides outputs/abides_eval/v1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path, help="CSV files or folders containing CSVs")
    p.add_argument("--out", type=Path, default=None, help="optional markdown output file")
    a = p.parse_args()
    files = []
    for x in a.paths:
        files += sorted(x.glob("*.csv")) if x.is_dir() else [x]
    rows = [r for f in files for r in csv.DictReader(open(f))]
    if not rows:
        raise SystemExit("No rows found")
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))

    def cell(pol, sc, key="pnl"):
        v = [float(r[key]) for r in rows if r["policy"] == pol and r["scenario"] == sc]
        return np.mean(v) if v else float("nan")

    lines = ["| policy | " + " | ".join(scenarios) + " | mean PnL | mean excl. stress | worst scen. | mean maxDD | maxDD excl. stress"
             " | mean max abs(inv) | mean fills |",
             "|---" * (len(scenarios) + 8) + "|"]
    for pol in policies:
        per = [cell(pol, sc) for sc in scenarios]
        mine = [r for r in rows if r["policy"] == pol]
        dd = np.mean([float(r["max_drawdown"]) for r in mine])
        fills = np.mean([float(r["fills"]) for r in mine])
        calm_rows = [r for r in mine if r["scenario"] != "stress_test"]
        dd_calm = np.mean([float(r["max_drawdown"]) for r in calm_rows]) if calm_rows else float("nan")
        max_inv = np.mean([float(r["max_abs_inventory"]) for r in mine])
        # stress_test PnL is dominated by mark-to-market jumps when the book empties,
        # so we also report the mean over the other scenarios.
        calm = [v for v, sc in zip(per, scenarios) if sc != "stress_test"]
        lines.append(f"| {pol} | " + " | ".join(f"{v:,.0f}" for v in per)
                     + f" | {np.nanmean(per):,.0f} | **{np.nanmean(calm) if calm else float('nan'):,.0f}**"
                     + f" | {np.nanmin(per):,.0f} | {dd:,.0f} | **{dd_calm:,.0f}** | {max_inv:,.0f} | {fills:,.0f} |")
    n_seeds = len(set(r["seed"] for r in rows))
    text = f"ABIDES-NG mean PnL ($) per scenario ({n_seeds} seed(s), mode={rows[0]['mode']})\n\n" + "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()

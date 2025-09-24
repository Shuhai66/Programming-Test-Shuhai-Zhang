# -*- coding: utf-8 -*-
"""
Task 6 (Cohort Comparison, US-only): 1995–1999 vs 2003–2007
Metrics per startup:
  - innov_quantity: # of patents within 0–5 years after first patent date (exclude the first patent)
  - innov_quality : # of forward citations to the first patent within 0–5 years
Tests:
  - Welch's t-test (unequal variances)
  - Mann–Whitney U (nonparametric)
Robust checks added:
  - Median & IQR
  - 5% trimmed mean
  - Mean of log(1 + x)
Saving:
  - Try Desktop/Programming Test Task6; fallback to Home or script folder; skip silently if not writable.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from utils import (
    load_csv, first_patents, compute_innov_quantity, compute_innov_quality_from_cites,
    assemble_per_startup, summarize_cohort_comparison, Timer
)

# -----------------------
# Config
# -----------------------
BASE_DIR = ""
PATENT_PATH   = os.path.join(BASE_DIR, "../data/g_patent.csv")
ASSIGNEE_PATH = os.path.join(BASE_DIR, "../data/g_assignee_disambiguated.csv")
LOCATION_PATH = os.path.join(BASE_DIR, "../data/g_location_disambiguated.csv")
CITE_PATHS = [
    os.path.join(BASE_DIR, "../data/g_us_patent_citation_1.csv"),
    os.path.join(BASE_DIR, "../data/g_us_patent_citation_2.csv"),
]

# Columns
PATENT_COLS = ["patent_id", "patent_date"]
ASSIG_COLS  = ["patent_id", "assignee_id", "location_id"]
LOC_COLS    = ["location_id", "disambig_state", "disambig_country"]

WINDOW_YEARS = 5
COHORT_A = (1995, 1999)  # inclusive
COHORT_B = (2003, 2007)  # inclusive
CITE_CHUNK = 1_000_000

# Whether to exclude the first patent itself from innov_quantity (common in literature)
EXCLUDE_FIRST_IN_QTY = True

def _permission_safe_folder(foldername: str):
    """Return a writable folder path (Desktop -> Home -> Script), or (None, False) if none works."""
    from datetime import datetime
    candidates = [
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.expanduser("~"),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for base in candidates:
        try:
            if not os.path.isdir(base):
                continue
            target = os.path.join(base, foldername)
            os.makedirs(target, exist_ok=True)
            # write probe
            probe = os.path.join(target, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return target, True
        except Exception:
            continue
    return None, False


def _robust_summary(series: pd.Series) -> dict:
    """Compute median, IQR, 5% trimmed mean, and mean log1p."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return dict(median=np.nan, iqr=np.nan, trimmed_mean=np.nan, mean_log1p=np.nan, n=0)
    q1, q3 = np.percentile(s, [25, 75])
    trimmed = stats.trim_mean(s.values, 0.05)  # 5% each tail
    mean_log1p = float(np.log1p(s).mean())
    return dict(median=float(np.median(s)), iqr=float(q3 - q1),
                trimmed_mean=float(trimmed), mean_log1p=mean_log1p, n=int(len(s)))


def main():
    # 1) Load patents, assignees (with location_id), locations
    with Timer("Load g_patent, g_assignee_disambiguated, g_location"):
        pat = load_csv(PATENT_PATH, usecols=PATENT_COLS)
        asg = load_csv(ASSIGNEE_PATH, usecols=ASSIG_COLS)
        loc = load_csv(LOCATION_PATH, usecols=LOC_COLS)
        pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")

    # 2) Identify first patents (founding)
    with Timer("Identify first patents per assignee (startups)"):
        fp = first_patents(patent_df=pat, assignee_df=asg)

    # 3) US-only on the EXACT assignee row of the first patent
    with Timer("US-filter on the assignee row of the FIRST patent"):
        fp_loc = fp.merge(
            asg[["assignee_id", "patent_id", "location_id"]],
            left_on=["assignee_id", "first_patent_id"],
            right_on=["assignee_id", "patent_id"],
            how="left"
        ).drop(columns=["patent_id"])
        fp_loc = fp_loc.merge(
            loc.rename(columns={"disambig_state": "state", "disambig_country": "country"}),
            on="location_id", how="left"
        )
        is_us = fp_loc["country"].astype(str).str.upper().isin(["US", "USA", "UNITED STATES"]) & fp_loc["state"].notna()
        fp = fp_loc[is_us].copy()

    # 4) Safe 5y window (avoid right truncation). Both cohorts are safe given typical data, but keep the guard.
    with Timer("Apply safe 5y window (avoid right truncation)"):
        max_year = int(pat["patent_date"].dt.year.max())
        safe_end = min(2015, max_year - WINDOW_YEARS)
        # Ensure both cohorts are within safe_end (they should be)
        a_min, a_max = COHORT_A
        b_min, b_max = COHORT_B
        if a_max > safe_end or b_max > safe_end:
            # tighten if needed
            a_max = min(a_max, safe_end)
            b_max = min(b_max, safe_end)
        print(f"Cohorts used (safe): A={a_min}-{a_max}, B={b_min}-{b_max}")
        fp = fp[(fp["first_patent_year"] >= min(a_min, b_min)) &
                (fp["first_patent_year"] <= max(a_max, b_max))].copy()

    # 5) Compute innov_quantity (0–5y patents per startup)
    with Timer("Compute innov_quantity (0–5y patents)"):
        qty = compute_innov_quantity(
            pat_df=pat,
            asg_df=asg[["patent_id", "assignee_id"]],
            fp_df=fp,
            window_years=WINDOW_YEARS
        )
        qty["innov_quantity"] = pd.to_numeric(qty["innov_quantity"], errors="coerce").fillna(0.0)
        if EXCLUDE_FIRST_IN_QTY:
            qty["innov_quantity"] = (qty["innov_quantity"] - 1.0).clip(lower=0.0)

    # 6) Compute innov_quality (0–5y citations to first patent)
    with Timer("Compute innov_quality (0–5y citations to FIRST patent)"):
        qual = compute_innov_quality_from_cites(
            fp_df=fp, pat_df=pat,
            cite_paths=CITE_PATHS,
            window_years=WINDOW_YEARS, chunksize=CITE_CHUNK
        )
        qual["innov_quality"] = pd.to_numeric(qual["innov_quality"], errors="coerce").fillna(0.0)

    # 7) Assemble per-startup dataset (US-only)
    with Timer("Assemble per-startup dataset (US-only)"):
        per = assemble_per_startup(fp_df=fp, qty_df=qty, qual_df=qual)
        per["innov_quantity"] = pd.to_numeric(per["innov_quantity"], errors="coerce").fillna(0.0)
        per["innov_quality"] = pd.to_numeric(per["innov_quality"], errors="coerce").fillna(0.0)

    # 8) Cohort summary & tests via your helper (Welch t + MWU)
    with Timer("Cohort summary & significance tests"):
        summary = summarize_cohort_comparison(
            per_startup=per,
            a_range=COHORT_A,
            b_range=COHORT_B
        )
        print("\n=== Task 6: Cohort Comparison (US-only; 1995–1999 vs 2003–2007) ===")
        print(summary.to_string(index=False))

    # 9) Robust checks (median/IQR, trimmed mean, mean log1p)
    a_mask = per["founding_year"].between(COHORT_A[0], COHORT_A[1])
    b_mask = per["founding_year"].between(COHORT_B[0], COHORT_B[1])

    rob_qty_a = _robust_summary(per.loc[a_mask, "innov_quantity"])
    rob_qty_b = _robust_summary(per.loc[b_mask, "innov_quantity"])
    rob_qlt_a = _robust_summary(per.loc[a_mask, "innov_quality"])
    rob_qlt_b = _robust_summary(per.loc[b_mask, "innov_quality"])

    robust_table = pd.DataFrame([
        ["innov_quantity (0–5y)", "A", rob_qty_a["median"], rob_qty_a["iqr"], rob_qty_a["trimmed_mean"], rob_qty_a["mean_log1p"], rob_qty_a["n"]],
        ["innov_quantity (0–5y)", "B", rob_qty_b["median"], rob_qty_b["iqr"], rob_qty_b["trimmed_mean"], rob_qty_b["mean_log1p"], rob_qty_b["n"]],
        ["innov_quality (0–5y)",  "A", rob_qlt_a["median"], rob_qlt_a["iqr"], rob_qlt_a["trimmed_mean"], rob_qlt_a["mean_log1p"], rob_qlt_a["n"]],
        ["innov_quality (0–5y)",  "B", rob_qlt_b["median"], rob_qlt_b["iqr"], rob_qlt_b["trimmed_mean"], rob_qlt_b["mean_log1p"], rob_qlt_b["n"]],
    ], columns=["metric", "cohort", "median", "IQR", "trimmed_mean_5pct", "mean_log1p", "n"])

    print("\n=== Robust Checks (median/IQR, 5% trimmed mean, mean log(1+x)) ===")
    print(robust_table.to_string(index=False))

    # 10) Save outputs (CSV + console text) to a permission-safe folder
    with Timer("Save outputs"):
        out_dir, ok = _permission_safe_folder("Programming Test Task6")
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Always try to save CSVs in the current working directory too (these are small)
        per_out = f"task6_per_startup_metrics_us_only.csv"
        sum_out = f"task6_cohort_comparison_summary_us_only.csv"
        robust_out = f"task6_robust_checks_us_only.csv"
        per.to_csv(per_out, index=False)
        summary.to_csv(sum_out, index=False)
        robust_table.to_csv(robust_out, index=False)

        # Build console text
        console_lines = []
        console_lines.append("=== Task 6: Cohort Comparison (US-only; 1995–1999 vs 2003–2007) ===")
        console_lines.append(summary.to_string(index=False))
        console_lines.append("")
        console_lines.append("=== Robust Checks (median/IQR, 5% trimmed mean, mean log(1+x)) ===")
        console_lines.append(robust_table.to_string(index=False))
        console_text = "\n".join(console_lines) + "\n"

        if ok and out_dir is not None:
            txt_path = os.path.join(out_dir, f"task6_console_output_{ts}.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(console_text)
                print("\nSaved to Desktop (or fallback) folder:")
                print(f" - {txt_path}")
            except Exception:
                pass  # silently skip
        print("\nSaved:")
        print(f"  - {per_out}")
        print(f"  - {sum_out}")
        print(f"  - {robust_out}")

if __name__ == "__main__":
    main()

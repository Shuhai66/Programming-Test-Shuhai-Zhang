# task5.py (minimal changes; aligned with Tasks 1–4)
# -----------------------------------------------------------
# Goal: Average innovation quantity & quality by founding year
# Alignment with Tasks 1–4:
#   - Startup = assignee; founding year = grant year of FIRST patent
#   - Keep ONLY US startups using the assignee row ON the FIRST patent
#   - Use 0–5y windows (quantity = patents; quality = forward citations)
#   - Avoid right truncation: require a full 5-year window
#   - Treat missing metrics as 0 (not NaN) before averaging
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils import (
    Timer, load_csv, first_patents,
    compute_innov_quantity, compute_innov_quality_from_cites,
    assemble_per_startup
)

# --------- Config ---------
BASE_DIR = ""
PATENT_PATH   = os.path.join(BASE_DIR, "../data/g_patent.csv")
ASSIGNEE_PATH = os.path.join(BASE_DIR, "../data/g_assignee_disambiguated.csv")
LOC_PATH      = os.path.join(BASE_DIR, "../data/g_location_disambiguated.csv")
CITE1_PATH    = os.path.join(BASE_DIR, "../data/g_us_patent_citation_1.csv")
CITE2_PATH    = os.path.join(BASE_DIR, "../data/g_us_patent_citation_2.csv")

ANALYSIS_START, ANALYSIS_END = 1985, 2015
COUNT_WINDOW_YEARS = 5    # quantity window
CITE_WINDOW_YEARS  = 5    # quality window
CHUNK_SIZE = 1_000_000    # citation streaming size

# Whether "innovation quantity" includes the first patent itself.
# Set to False (common in literature: count follow-on patents after the first).
INCLUDE_FIRST_IN_QTY = False

# --------- Step 1: Load minimal tables ---------
with Timer("Load g_patent, g_assignee_disambiguated, g_location"):
    pat = load_csv(PATENT_PATH, usecols=["patent_id", "patent_date"])
    # include location_id so we can US-filter on the assignee row of the FIRST patent
    asg = load_csv(ASSIGNEE_PATH, usecols=["patent_id", "assignee_id", "location_id"])
    loc = load_csv(LOC_PATH, usecols=["location_id", "disambig_state", "disambig_country"])
    pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")
print("pat shape:", pat.shape, "| asg shape:", asg.shape, "| loc shape:", loc.shape)

# --------- Step 2: First patent per assignee (founding year) ---------
with Timer("Identify first patent per assignee"):
    fp = first_patents(patent_df=pat, assignee_df=asg)  # ['assignee_id','first_patent_id','first_patent_date','first_patent_year']

# --------- Step 3: US-only startups (same rule as Tasks 1–3) ---------
# Join the EXACT assignee row on its first patent -> location_id -> location table
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

# --------- Step 4: Avoid right truncation (require full 5y window) ---------
# If max grant year in the data is Ymax, cohorts after Ymax - 5 have an incomplete window.
with Timer("Apply safe cohort window (full 5y coverage)"):
    max_grant_year = int(pat["patent_date"].dt.year.max())
    safe_end_year = min(ANALYSIS_END, max_grant_year - CITE_WINDOW_YEARS)
    fp = fp[(fp["first_patent_year"] >= ANALYSIS_START) & (fp["first_patent_year"] <= safe_end_year)].copy()
    print(f"Founding years used: {ANALYSIS_START}–{safe_end_year} (full 5y windows)")

# --------- Step 5: Innovation quantity (0–5y) ---------
with Timer("Compute innovation QUANTITY (0–5y)"):
    qty = compute_innov_quantity(
        pat_df=pat, asg_df=asg[["patent_id", "assignee_id"]], fp_df=fp,
        window_years=COUNT_WINDOW_YEARS
    )
    # Ensure numeric and fill missing with zeros
    qty["innov_quantity"] = pd.to_numeric(qty["innov_quantity"], errors="coerce").fillna(0.0)

    # Optional: exclude the first patent itself from the count
    if not INCLUDE_FIRST_IN_QTY:
        qty["innov_quantity"] = (qty["innov_quantity"] - 1.0).clip(lower=0.0)

# --------- Step 6: Innovation quality (forward citations to first patent within 5y) ---------
with Timer("Compute innovation QUALITY via streaming citations"):
    qual = compute_innov_quality_from_cites(
        fp_df=fp, pat_df=pat,
        cite_paths=[CITE1_PATH, CITE2_PATH],
        window_years=CITE_WINDOW_YEARS, chunksize=CHUNK_SIZE
    )
    # Ensure numeric and fill missing with zeros
    qual["innov_quality"] = pd.to_numeric(qual["innov_quality"], errors="coerce").fillna(0.0)

# --------- Step 7: Assemble per-startup and aggregate by founding year ---------
with Timer("Assemble per-startup table and aggregate means by year"):
    per_startup = assemble_per_startup(fp_df=fp, qty_df=qty, qual_df=qual)
    # Guard against any residual NaN
    per_startup["innov_quantity"] = pd.to_numeric(per_startup["innov_quantity"], errors="coerce").fillna(0.0)
    per_startup["innov_quality"] = pd.to_numeric(per_startup["innov_quality"], errors="coerce").fillna(0.0)

    summary = (per_startup
               .groupby("founding_year", as_index=False)
               .agg(avg_innov_quantity=("innov_quantity", "mean"),
                    avg_innov_quality=("innov_quality", "mean")))

    print(summary.head())

# --------- Step 8: Save console + figure (permission-safe) and show 10s ---------
from datetime import datetime
import os

# Build a console-style summary string for the FULL table
console_header = (
    "=== Average Startup Innovation Quantity & Quality by Founding Year (1985–2015, US only) ===\n"
    f"(Computed on cohorts {ANALYSIS_START}–{ANALYSIS_END} "
    f"with a {CITE_WINDOW_YEARS}-year window after the first patent)\n"
)
console_table = summary.to_string(index=False)  # <-- FULL table, not head()
console_text = console_header + console_table + "\n"

# Print FULL table to console
print(console_header)
print(console_table)

# Decide a writable target folder: Desktop -> Home -> Script folder
timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
foldername = "Programming Test Task5"
save_enabled = False
target_dir = None

candidates = [
    os.path.join(os.path.expanduser("~"), "Desktop"),   # Desktop (macOS/Windows)
    os.path.expanduser("~"),                            # Home
    os.path.dirname(os.path.abspath(__file__)),         # Script folder
]

for base in candidates:
    try:
        if not os.path.isdir(base):
            continue
        candidate_dir = os.path.join(base, foldername)
        os.makedirs(candidate_dir, exist_ok=True)
        # Probe write permission
        probe = os.path.join(candidate_dir, ".write_probe")
        with open(probe, "w", encoding="utf-8") as _f:
            _f.write("ok")
        os.remove(probe)
        target_dir = candidate_dir
        save_enabled = True
        break
    except Exception:
        # silently try next candidate
        continue

# File paths (only if a writable folder was found)
if save_enabled:
    txt_path = os.path.join(target_dir, f"task5_console_output_{timestamp}.txt")
    fig_path = os.path.join(target_dir, f"task5_avg_qty_quality_{timestamp}.png")
    try:
        # Save FULL console text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console_text)
    except Exception:
        txt_path = None  # skip on error
else:
    txt_path, fig_path = None, None

# ---- Plot the FULL range (1985–2015) to match assignment wording ----
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(summary["founding_year"], summary["avg_innov_quantity"],
         marker="o", label="Avg. Quantity (0–5y)")
ax2.plot(summary["founding_year"], summary["avg_innov_quality"],
         marker="s", linestyle="--", label="Avg. Quality (0–5y)")

ax1.set_xlabel("Founding Year")
ax1.set_ylabel("Average Innovation Quantity (patents within 5y)")
ax2.set_ylabel("Average Innovation Quality (forward citations within 5y)")
ax1.set_title("Average Startup Innovation Quantity & Quality by Founding Year (1985–2015, US only)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure if possible
if save_enabled and fig_path is not None:
    try:
        plt.savefig(fig_path, dpi=180, bbox_inches="tight")
        print(f"\n[Saved] Files:")
        if txt_path: print(f" - TXT: {txt_path}")
        print(f" - FIG: {fig_path}")
    except Exception:
        pass  # silently skip on error

# Show for 10 seconds then close
plt.show(block=False)
plt.pause(10)
plt.close()

# ===========================
# Task 3: Top US State by Startup Count
# (location_id is in g_assignee_disambiguated;
#  state/country columns are disambig_state / disambig_country)
# ===========================

import pandas as pd
# Assumes your general-purpose functions are already imported:
# load_csv, first_patents


import matplotlib.pyplot as plt
from tqdm import tqdm
# import shared utils
from utils import load_csv, first_patents
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ---- Paths ----
PATH_PATENT = "../data/g_patent.csv"
PATH_ASSIG  = "../data/g_assignee_disambiguated.csv"
PATH_LOC    = "../data/g_location_disambiguated.csv"

START_YEAR = 1985
END_YEAR   = 2015

# ---------------------------
# Load minimal necessary columns
# ---------------------------
# patents: only id and date
pat = load_csv(PATH_PATENT, usecols=["patent_id", "patent_date"])

pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")

# assignees: id, assignee, and location_id (note: some location_id may be missing)
asg = load_csv(PATH_ASSIG,  usecols=["patent_id", "assignee_id", "location_id"])

# locations: use PatentsView-style disambiguated fields
loc = load_csv(PATH_LOC, usecols=["location_id", "disambig_state", "disambig_country"])

# Basic column checks (fail fast)
for df, need, name in [
    (pat, ["patent_id", "patent_date"], "g_patent.csv"),
    (asg, ["patent_id", "assignee_id", "location_id"], "g_assignee_disambiguated.csv"),
    (loc, ["location_id", "disambig_state", "disambig_country"], "g_location_disambiguated.csv"),
]:
    missing = set(need) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

# ---------------------------
# Build first patent (startup) table
# ---------------------------
# Identify the first patent of each assignee and its founding year (year of first grant)
fp = first_patents(patent_df=pat, assignee_df=asg)

# Restrict to the analysis window 1985–2015
fp = fp[(fp["first_patent_year"] >= START_YEAR) & (fp["first_patent_year"] <= END_YEAR)].copy()

# ---------------------------
# Attach location (state) to the first patent
# ---------------------------
# Join to the *exact* assignee row on its first patent to fetch that assignee's location_id.
# This avoids mixing other co-assignees' locations on the same patent.
asg_first = asg[["assignee_id", "patent_id", "location_id"]].copy()

first_pat_loc = fp.merge(
    asg_first,
    left_on=["assignee_id", "first_patent_id"],
    right_on=["assignee_id", "patent_id"],
    how="left"
)

# Count how many startups have missing location_id (cannot be attributed to a state)
missing_loc = first_pat_loc["location_id"].isna().sum()
if missing_loc > 0:
    print(f"[Info] Startups without location_id on their first patent: {missing_loc}")

# Drop rows without a resolvable location_id
first_pat_loc = first_pat_loc.dropna(subset=["location_id"]).copy()

# Map location_id -> state/country (use disambiguated columns)
first_pat_loc = first_pat_loc.merge(
    loc,
    on="location_id",
    how="left"
)

# Keep U.S. only (allow NA just in case, though your file shows 'US')
is_us = first_pat_loc["disambig_country"].astype(str).str.upper().isin(["US", "USA", "UNITED STATES"])
first_pat_loc = first_pat_loc[is_us | first_pat_loc["disambig_country"].isna()]

# We must have a state code (e.g., 'PA', 'NY')
first_pat_loc = first_pat_loc.dropna(subset=["disambig_state"]).copy()

# ---------------------------
# Count startups by state
# ---------------------------
# Each row corresponds to a startup (one assignee) via the state of its first patent.
# ---------------------------
# Save console summary + figure (robust cross-platform)
# ---------------------------

# Pretty console string (top 15 shown in figure; text file saves full table)
# Build state_counts = startups per state (descending)
state_counts = (
    first_pat_loc
    .groupby("disambig_state", as_index=False)
    .size()
    .rename(columns={"disambig_state": "state", "size": "num_startups"})
    .sort_values("num_startups", ascending=False)
    .reset_index(drop=True)
)

console_header = f"=== Top US States by Startup Count (Founding years {START_YEAR}-{END_YEAR}) ==="
console_table  = state_counts.to_string(index=False)
console_str    = f"{console_header}\n{console_table}\n"

# Print to console as usual
print(console_header)
print(state_counts.head(10))
if not state_counts.empty:
    top_state = state_counts.iloc[0]
    print(f"\n*** State with the most startups: {top_state['state']} ({int(top_state['num_startups'])} startups) ***")
else:
    print("\nNo state results available. Check location linkage and state fields.")

# --- Decide a writable target folder: Desktop -> Home -> Script folder ---
timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
foldername = "Programming Test Task3"
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
        # probe write permission
        probe = os.path.join(candidate_dir, ".write_probe")
        with open(probe, "w", encoding="utf-8") as _f:
            _f.write("ok")
        os.remove(probe)
        target_dir = candidate_dir
        save_enabled = True
        break
    except Exception:
        # try next candidate silently
        continue

# Build file paths (only if a writable folder was found)
if save_enabled:
    txt_path = os.path.join(target_dir, f"task3_console_output_{timestamp}.txt")
    fig_path = os.path.join(target_dir, f"task3_top_states_{timestamp}.png")
    try:
        # Save console text (full table)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console_str)
    except Exception:
        pass  # silently skip if permission issues

# --- Plot top 15 states as a bar chart ---
topN = 15
plot_df = state_counts.head(topN)

plt.figure(figsize=(11, 6))
plt.bar(plot_df["state"], plot_df["num_startups"])
plt.title(f"Top {topN} US States by Startup Count ({START_YEAR}–{END_YEAR})")
plt.xlabel("State")
plt.ylabel("Number of Startups")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save figure if possible
if save_enabled:
    try:
        plt.savefig(fig_path, dpi=180, bbox_inches="tight")
        print(f"\nSaved files:\n- TXT: {txt_path}\n- FIG: {fig_path}")
    except Exception:
        pass  # silently skip if permission issues

# Show for 10 seconds then close
plt.show(block=False)
plt.pause(10)
plt.close()

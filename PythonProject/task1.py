# Task 1 — US startups per year (assignee-based, first-patent year)
# Requirements:
#   - A startup is founded in the grant year of its FIRST patent.
#   - Count UNIQUE assignees by that year.
#   - Keep ONLY US startups (by state of the FIRST patent).
#   - Visualize the annual counts (1985–2015).

import pandas as pd
import matplotlib.pyplot as plt
from utils import load_csv, first_patents
import os
from datetime import datetime

# ---------------- Config ----------------
PATENT_PATH   = "../data/g_patent.csv"
ASSIGNEE_PATH = "../data/g_assignee_disambiguated.csv"
LOC_PATH      = "../data/g_location_disambiguated.csv"
YEAR_MIN, YEAR_MAX = 1985, 2015

# Whitelist of US state/territory codes we keep as "US" (include DC; add "PR" if desired)
US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
    "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
    "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
    "WI","WY","DC"
}

# ------------- 1) Load minimal columns -------------
pat = load_csv(PATENT_PATH,   usecols=["patent_id", "patent_date"])
asg = load_csv(ASSIGNEE_PATH, usecols=["patent_id", "assignee_id", "location_id"])
loc = load_csv(LOC_PATH,      usecols=["location_id", "disambig_state", "disambig_country"])

# Parse dates once
pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")

# ------------- 2) First patent per assignee -------------
# Returns: ['assignee_id','first_patent_id','first_patent_year','first_patent_date']
firsts = first_patents(patent_df=pat, assignee_df=asg)

# ------------- 3) Map FIRST patent -> US state -------------
# Build mapping patent_id -> (state, country) via assignee's location_id
pat2state = (
    asg[["patent_id", "location_id"]].dropna().drop_duplicates()
       .merge(loc.rename(columns={"disambig_state":"state", "disambig_country":"country"}),
              on="location_id", how="left")
       [["patent_id", "state", "country"]]
)

# Attach location of the FIRST patent
firsts = firsts.merge(pat2state, left_on="first_patent_id", right_on="patent_id", how="left") \
               .drop(columns=["patent_id"])

# Robust US filter:
#   Keep if state is a valid US code OR (country == "US" and state is non-missing)
is_us = firsts["state"].isin(US_STATES) | (
    (firsts["country"] == "US") & firsts["state"].notna()
)
firsts_us = firsts.loc[is_us].copy()

# ------------- 4) Restrict window and count startups -------------
firsts_us = firsts_us.query("@YEAR_MIN <= first_patent_year <= @YEAR_MAX")
counts = (firsts_us
          .groupby("first_patent_year", as_index=False)
          .size()
          .rename(columns={"first_patent_year": "founding_year", "size": "num_startups"})
          .sort_values("founding_year")
          .reset_index(drop=True))


# ------------- 5) Print & plot (robust cross-platform saving) -------------



# Console header + table
console_header = f"=== US startups (unique assignees) founded per year ({YEAR_MIN}–{YEAR_MAX}) ==="
console_table = counts.to_string(index=False)
console_str = f"{console_header}\n{console_table}\n"

# Print to console
print(console_header)
print(counts)

# Make the plot (before deciding where to save)
plt.figure(figsize=(10, 6))
plt.plot(counts["founding_year"], counts["num_startups"], marker="o")
plt.title("US Startups Founded per Year (Assignees' First-Patent Year)")
plt.xlabel("Year")
plt.ylabel("Number of Startups")
plt.grid(True)
plt.tight_layout()

# --- Decide a writable target folder: Desktop -> Home -> Script folder ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
folder_name = "Programming Test Task1"

save_enabled = False
target_dir = None

candidates = [
    os.path.join(os.path.expanduser("~"), "Desktop"),  # Desktop (macOS/Windows)
    os.path.expanduser("~"),                           # Home
    os.path.dirname(os.path.abspath(__file__)),        # Script folder
]

for base in candidates:
    try:
        if not os.path.isdir(base):
            continue
        candidate_dir = os.path.join(base, folder_name)
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

# Build file paths (only if we found a writable folder)
if save_enabled:
    csv_path = os.path.join(target_dir, f"task1_startup_counts_us_by_assignee_{timestamp}.csv")
    txt_path = os.path.join(target_dir, f"task1_console_output_{timestamp}.txt")
    fig_path = os.path.join(target_dir, f"task1_startup_counts_us_by_assignee_{timestamp}.png")

    try:
        # Save TXT (console output)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console_str)

        # Save CSV (data)
        counts.to_csv(csv_path, index=False)

        # Save FIG (plot)
        plt.savefig(fig_path, dpi=180, bbox_inches="tight")

        print(f"\nSaved files:\n- CSV: {csv_path}\n- TXT: {txt_path}\n- FIG: {fig_path}")
    except Exception:
        # Any saving problem -> skip silently
        pass
# If no writable folder found, just skip saving silently

# Show for 10 seconds then close
plt.show(block=False)
plt.pause(10)
plt.close()


# ------------- 6) (Optional) save for later tasks -------------
# counts.to_csv("task1_startup_counts_us_by_assignee.csv", index=False)

# ------------- 7) Minimal sanity checks (non-invasive) -------------
assert counts["founding_year"].between(YEAR_MIN, YEAR_MAX).all(), "Found years outside the window."
assert (counts["num_startups"] >= 0).all(), "Counts must be non-negative."
assert counts["num_startups"].sum() > 0, "No startups found—check joins and filters."

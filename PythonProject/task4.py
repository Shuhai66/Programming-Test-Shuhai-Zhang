# task4.py
# -----------------------------------------------------------
# Task 4: CPC classes with the largest number of US startups
# Window: startups founded between 1985 and 2015 (by FIRST patent's grant year)
# Counting rule: use CPC of the FIRST patent only
# US filter: the assignee row on the FIRST patent must map to a US location
# Overcount guard: prefer "primary CPC only" if such a flag exists; otherwise
#                  deduplicate by (assignee_id, cpc_class)
# Outputs:
#   - task4_startups_by_cpc_class.csv      (full table)
#   - task4_top10_cpc_class.csv            (top 10)
#   - (optional) console summary saved under "Programming Test Task4"
# -----------------------------------------------------------

import os
from datetime import datetime
import pandas as pd
from utils import load_csv, first_patents  # assumed available in your repo

# ---------------- Config ----------------
PATENT_PATH   = "../data/g_patent.csv"
ASSIGNEE_PATH = "../data/g_assignee_disambiguated.csv"
CPC_PATH      = "../data/g_cpc_current.csv"
LOC_PATH      = "../data/g_location_disambiguated.csv"

YEAR_MIN, YEAR_MAX = 1985, 2015

# US state/territory whitelist (adjust if you want to include PR, etc.)
US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
    "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
    "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
    "WI","WY","DC"
}

# -----------------------------------------------------------
# 1) Load minimal columns
# -----------------------------------------------------------
pat = load_csv(PATENT_PATH,   usecols=["patent_id", "patent_date"])
asg = load_csv(ASSIGNEE_PATH, usecols=["patent_id", "assignee_id", "location_id"])
loc = load_csv(LOC_PATH,      usecols=["location_id", "disambig_state", "disambig_country"])

# g_cpc_current can be wide; try to read the columns we need robustly
# We will try to detect an indicator for "primary" classification if present.
cpc = load_csv(CPC_PATH)

# Column detection
possible_pat_id_cols = [c for c in cpc.columns if c.lower() in {"patent_id", "patentid"}]
possible_class_cols  = [c for c in cpc.columns if c.lower() in {"cpc_class", "cpcclass", "group_id", "groupid"}]
if not possible_pat_id_cols or not possible_class_cols:
    raise ValueError("g_cpc_current.csv must contain patent_id and a CPC class-like column (e.g., cpc_class).")

pat_id_col     = possible_pat_id_cols[0]
cpc_class_col  = possible_class_cols[0]

# Try to detect a "primary" flag/sequence column
primary_cols_candidates = [c for c in cpc.columns if c.lower() in {
    "sequence", "cpc_sequence", "position", "is_primary", "primary", "is_main"
}]

keep_cols = [pat_id_col, cpc_class_col] + primary_cols_candidates
cpc = cpc[keep_cols].copy()

# -----------------------------------------------------------
# 2) Parse dates; compute first patents (startups)
# -----------------------------------------------------------
pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")

# first_patents(...) should return:
#   ['assignee_id','first_patent_id','first_patent_date','first_patent_year']
fp = first_patents(patent_df=pat, assignee_df=asg)

# Restrict to founding year window
fp = fp[(fp["first_patent_year"] >= YEAR_MIN) & (fp["first_patent_year"] <= YEAR_MAX)].copy()

# -----------------------------------------------------------
# 3) Attach US location using the assignee row ON the first patent
#    (assignee_id + first_patent_id → location_id → loc)
# -----------------------------------------------------------
asg_first = asg[["assignee_id", "patent_id", "location_id"]].copy()

fp_loc = fp.merge(
    asg_first,
    left_on=["assignee_id", "first_patent_id"],
    right_on=["assignee_id", "patent_id"],
    how="left"
).drop(columns=["patent_id"])

missing_loc = fp_loc["location_id"].isna().sum()
if missing_loc:
    print(f"[Info] Startups without location_id on their first patent: {missing_loc}")

# Map to country/state
fp_loc = fp_loc.merge(
    loc.rename(columns={"disambig_state":"state", "disambig_country":"country"}),
    on="location_id", how="left"
)

# US filter: prefer explicit country == "US" AND state in whitelist
is_us = (fp_loc["country"] == "US") & fp_loc["state"].isin(US_STATES)
fp_us = fp_loc[is_us].copy()

dropped_non_us = len(fp_loc) - len(fp_us)
print(f"[Info] Non-US startups dropped: {dropped_non_us}")

# Unique US startups in window (by assignee_id)
us_startups_in_window = fp_us["assignee_id"].nunique()

# -----------------------------------------------------------
# 4) Join CPC of the FIRST patent only
# -----------------------------------------------------------
# Keep only CPC rows for those first_patent_id values
cpc_first = cpc[cpc[pat_id_col].isin(fp_us["first_patent_id"].unique())].copy()

# Drop rows with missing CPC class
cpc_first = cpc_first.dropna(subset=[cpc_class_col]).copy()

# If a primary indicator exists, try to prefer it; otherwise deduplicate.
used_primary_only = False
if primary_cols_candidates:
    prim_col = primary_cols_candidates[0]
    col = cpc_first[prim_col]
    # Case A: boolean / 0-1 primary
    if col.dropna().isin([0, 1, True, False]).all():
        cpc_first = cpc_first[col.astype(int) == 1]
        used_primary_only = True
    else:
        # Case B: sequence/position → keep the minimum per patent_id (lowest is primary)
        cpc_first[prim_col] = pd.to_numeric(cpc_first[prim_col], errors="coerce")
        cpc_first = (
            cpc_first.sort_values([pat_id_col, prim_col], na_position="last")
                     .groupby(pat_id_col, as_index=False)
                     .first()
        )
        used_primary_only = True

# Link CPC back to US startups (first patents only)
fp_cpc = fp_us.merge(
    cpc_first[[pat_id_col, cpc_class_col]],
    left_on="first_patent_id",
    right_on=pat_id_col,
    how="left"
).drop(columns=[pat_id_col])

# -----------------------------------------------------------
# 5) Count startups by CPC class (guard against duplicates)
# -----------------------------------------------------------
# If not using primary-only, deduplicate (assignee_id, cpc_class) to avoid overcounting.
if not used_primary_only:
    fp_cpc = fp_cpc.drop_duplicates(subset=["assignee_id", cpc_class_col])

# Drop startups with no CPC class (can't attribute)
missing_cpc = fp_cpc[cpc_class_col].isna().sum()
if missing_cpc:
    print(f"[Info] Startups without CPC class on their first patent: {missing_cpc}")
fp_cpc = fp_cpc.dropna(subset=[cpc_class_col]).copy()

# Now counts
counts_all = (
    fp_cpc.groupby(cpc_class_col, as_index=False)
          .size()
          .rename(columns={"size": "num_startups", cpc_class_col: "cpc_class"})
          .sort_values("num_startups", ascending=False)
          .reset_index(drop=True)
)
top10 = counts_all.head(10).copy()

print("\n=== CPC classes with the largest number of US startups (1985–2015) ===")
print(top10.to_string(index=False))

# -----------------------------------------------------------
# 6) Save outputs
# -----------------------------------------------------------
counts_all.to_csv("task4_startups_by_cpc_class.csv", index=False)
top10.to_csv("task4_top10_cpc_class.csv", index=False)

# -----------------------------------------------------------
# 7) Save console summary to "Programming Test Task4" (permission-safe)
# -----------------------------------------------------------
# Compute summary stats AFTER we have fp_cpc post-dropna
us_startups_with_cpc = fp_cpc["assignee_id"].nunique()

console_lines = []
console_lines.append("=== CPC classes with the largest number of US startups (1985–2015) ===")
console_lines.append(top10.to_string(index=False))
console_lines.append("")
console_lines.append(f"[Sanity] US startups in window (input): {int(us_startups_in_window)}")
console_lines.append(f"[Sanity] US startups with mappable CPC: {int(us_startups_with_cpc)}")
if used_primary_only:
    console_lines.append("[Info] CPC counting used primary-only mode (one class per startup).")
else:
    console_lines.append("[Info] CPC counting used deduplicated multi-label mode (startup may appear in multiple classes, but counted at most once per class).")

console_text = "\n".join(console_lines) + "\n"

foldername = "Programming Test Task4"
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
        continue  # silently try next candidate

if save_enabled:
    txt_path = os.path.join(target_dir, f"task4_console_output_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console_text)
        print(f"\n[Saved] Console summary → {txt_path}")
    except Exception:
        pass  # silently skip if any IO issues

print("\nSaved files:")
print(" - task4_startups_by_cpc_class.csv")
print(" - task4_top10_cpc_class.csv")

# Optional sanity prints (same numbers as saved above)
total_startups_input = len(fp_us)
total_startups_with_cpc = fp_cpc["assignee_id"].nunique()
print(f"\n[Sanity] US startups in window (input): {total_startups_input:,}")
print(f"[Sanity] US startups with mappable CPC: {total_startups_with_cpc:,}")
if used_primary_only:
    print("[Info] CPC counting used primary-only mode (one class per startup).")
else:
    print("[Info] CPC counting used deduplicated multi-label mode (startup may appear in multiple classes, but counted at most once per class).")

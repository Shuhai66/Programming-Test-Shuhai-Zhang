import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------- Config ----------------
BASE_DIR = ""  # "" if CSVs are in ../data relative to this script
PATENT_PATH   = os.path.join(BASE_DIR, "../data/g_patent.csv")
ASSIGNEE_PATH = os.path.join(BASE_DIR, "../data/g_assignee_disambiguated.csv")
CPC_PATH      = os.path.join(BASE_DIR, "../data/g_cpc_current.csv")
LOC_PATH      = os.path.join(BASE_DIR, "../data/g_location_disambiguated.csv")
CITE1_PATH    = os.path.join(BASE_DIR, "../data/g_us_patent_citation_1.csv")
CITE2_PATH    = os.path.join(BASE_DIR, "../data/g_us_patent_citation_2.csv")
PA_PATH       = os.path.join(BASE_DIR, "../data/pa.csv")

ANALYSIS_START, ANALYSIS_END_NAIVE = 1985, 2010   # we will tighten END with pa.csv max year
CITE_WINDOW_YEARS = 5
ACQ_WINDOW_YEARS  = 10
MAX_N = None
TECH_MIN, STATE_MIN = 50, 50  # collapse rare dummies into "Other"

# --------------- Utils ------------------
from utils import (
    load_csv,
    first_patents,
    compute_innov_quantity,
    compute_innov_quality_from_cites,
    assemble_per_startup,
)

# ---------------- 0) Load core tables ----------------
pat = load_csv(PATENT_PATH, usecols=["patent_id","patent_date"])
pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")

asg = load_csv(ASSIGNEE_PATH, usecols=["patent_id","assignee_id","location_id"])
loc = load_csv(LOC_PATH, usecols=["location_id","disambig_state","disambig_country"])

# CPC: keep only columns we need; detect a possible primary column if present
cpc_raw = load_csv(CPC_PATH)
# Robust column detection
def _pick(colnames, candidates):
    cl = [c for c in colnames if c.lower() in candidates]
    return cl[0] if cl else None

pid_col = _pick(cpc_raw.columns, {"patent_id","patentid"})
cls_col = _pick(cpc_raw.columns, {"cpc_class","cpcclass","group_id","groupid"})
prim_col = _pick(cpc_raw.columns, {"sequence","cpc_sequence","position","is_primary","primary","is_main"})
if pid_col is None or cls_col is None:
    raise ValueError("g_cpc_current must contain patent_id and a CPC class-like column.")

cpc = cpc_raw[[pid_col, cls_col] + ([prim_col] if prim_col else [])].copy()
cpc.rename(columns={pid_col:"patent_id", cls_col:"cpc_class"}, inplace=True)

# ---------------- 1) First patents (startups) ----------------
fp_all = first_patents(patent_df=pat, assignee_df=asg)  # ['assignee_id','first_patent_id','first_patent_date','first_patent_year']

# ---------------- 2) US-only via the EXACT assignee row on the FIRST patent ----------------
# Map (assignee_id, first_patent_id) -> location_id -> country/state
fp_loc = fp_all.merge(
    asg[["assignee_id","patent_id","location_id"]],
    left_on=["assignee_id","first_patent_id"],
    right_on=["assignee_id","patent_id"],
    how="left"
).drop(columns=["patent_id"])

fp_loc = fp_loc.merge(
    loc.rename(columns={"disambig_state":"state","disambig_country":"country"}),
    on="location_id", how="left"
)

is_us = fp_loc["country"].astype(str).str.upper().isin(["US","USA","UNITED STATES"]) & fp_loc["state"].notna()
fp_us = fp_loc[is_us].copy()

# ---------------- 3) Safe END for 10y acquisition window using pa.csv max year ----------------
# Read pa.csv in a streaming-friendly way to get max exec_dt year
pa_cols = ["patent_id","exec_dt","convey_type"]
pa_max_year = None
for chunk in pd.read_csv(PA_PATH, usecols=pa_cols, sep=None, engine="python", chunksize=1_000_000):
    y = pd.to_datetime(chunk["exec_dt"], errors="coerce").dt.year
    mx = y.max()
    if pd.notna(mx):
        pa_max_year = mx if pa_max_year is None else max(pa_max_year, mx)
if pa_max_year is None:
    # fallback: assume pa covers to 2020+
    pa_max_year = 2020

safe_end = min(ANALYSIS_END_NAIVE, pa_max_year - ACQ_WINDOW_YEARS)

fp = fp_us[(fp_us["first_patent_year"] >= ANALYSIS_START) &
           (fp_us["first_patent_year"] <= safe_end)].copy()

# ---------------- 4) Innovation metrics (aligned with Task 5/6) ----------------
# Quantity in 0–5y, EXCLUDING the first patent itself (common in our earlier tasks)
qty  = compute_innov_quantity(
    pat_df=pat, asg_df=asg[["patent_id","assignee_id"]], fp_df=fp, window_years=CITE_WINDOW_YEARS
)
qty["innov_quantity"] = pd.to_numeric(qty["innov_quantity"], errors="coerce").fillna(0.0)
# Exclude the first patent from the count if your function includes it
qty["innov_quantity"] = (qty["innov_quantity"] - 1.0).clip(lower=0.0)

# Quality = forward citations to first patent within 5y
qual = compute_innov_quality_from_cites(
    fp_df=fp, pat_df=pat,
    cite_paths=[CITE1_PATH, CITE2_PATH],
    window_years=CITE_WINDOW_YEARS, chunksize=1_000_000
)
qual["innov_quality"] = pd.to_numeric(qual["innov_quality"], errors="coerce").fillna(0.0)

# ---------------- 5) Tech field (single label from the FIRST patent) ----------------
# Prefer primary CPC if we have a marker; otherwise, pick one class deterministically.
cpc_first = cpc[cpc["patent_id"].isin(fp["first_patent_id"].unique())].dropna(subset=["cpc_class"]).copy()

if prim_col:
    # Try to interpret primary: boolean or smallest sequence as primary
    col = cpc_raw[prim_col]
    # Rebuild cpc_first to include prim_col properly (types may differ in subset)
    cpc_first = cpc_raw[[pid_col, cls_col, prim_col]].rename(
        columns={pid_col:"patent_id", cls_col:"cpc_class"}
    )
    cpc_first = cpc_first[cpc_first["patent_id"].isin(fp["first_patent_id"])].dropna(subset=["cpc_class"]).copy()

    if pd.api.types.is_bool_dtype(cpc_first[prim_col]) or cpc_first[prim_col].dropna().isin([0,1,True,False]).all():
        cpc_first[prim_col] = cpc_first[prim_col].astype(int)
        cpc_first = cpc_first[cpc_first[prim_col] == 1]
    else:
        cpc_first[prim_col] = pd.to_numeric(cpc_first[prim_col], errors="coerce")
        cpc_first = (cpc_first.sort_values(["patent_id", prim_col], na_position="last")
                               .groupby("patent_id", as_index=False)
                               .first())
# After this, keep ONE class per first_patent_id (if still multiple, dedupe deterministically)
cpc_first = (cpc_first.sort_values(["patent_id","cpc_class"])
                        .drop_duplicates(subset=["patent_id"], keep="first"))[["patent_id","cpc_class"]]

tech = fp[["assignee_id","first_patent_id"]].merge(
    cpc_first, left_on="first_patent_id", right_on="patent_id", how="left"
)[["assignee_id","cpc_class"]].rename(columns={"cpc_class":"tech_field"})

# ---------------- 6) State from the EXACT first-patent assignee row (already in fp) ----------------
state = fp[["assignee_id","state"]].copy()

# ---------------- 7) Acquisition within 10y (assignment/merger) ----------------
pa = load_csv(PA_PATH, usecols=["patent_id","exec_dt","convey_type"]).rename(
    columns={"exec_dt":"event_date","convey_type":"event_type"}
)
pa["event_date"] = pd.to_datetime(pa["event_date"], errors="coerce")
pa["event_type"] = pa["event_type"].astype(str).str.lower()
pa.loc[pa["event_type"].str.contains("assign", na=False), "event_type"] = "assignment"
pa.loc[pa["event_type"].str.contains("merg",   na=False), "event_type"] = "merger"
pa = pa[pa["event_type"].isin({"assignment","merger"})].copy()

acq = fp[["assignee_id","first_patent_id","first_patent_date"]].merge(
    pa, left_on="first_patent_id", right_on="patent_id", how="left"
)
acq = acq.dropna(subset=["first_patent_date"]).copy()
t0 = pd.to_datetime(acq["first_patent_date"], errors="coerce")
t1 = t0 + pd.DateOffset(years=ACQ_WINDOW_YEARS)
flag = acq["event_type"].isin(["assignment","merger"]) & (acq["event_date"] >= t0) & (acq["event_date"] < t1)
acq_flag = acq.loc[flag, ["assignee_id"]].drop_duplicates().assign(acquired=1)
acq_flag = fp[["assignee_id"]].merge(acq_flag, on="assignee_id", how="left").fillna({"acquired":0})

# ---------------- 8) Assemble regression frame ----------------
per = assemble_per_startup(fp_df=fp, qty_df=qty, qual_df=qual)
df = (per.merge(tech, on="assignee_id", how="left")
         .merge(state, on="assignee_id", how="left")
         .merge(acq_flag, on="assignee_id", how="left"))

# Clean & drop NAs on key fields
df["acquired"]       = pd.to_numeric(df["acquired"], errors="coerce").fillna(0).astype(int)
df["innov_quantity"] = pd.to_numeric(df["innov_quantity"], errors="coerce").fillna(0.0)
df["innov_quality"]  = pd.to_numeric(df["innov_quality"], errors="coerce").fillna(0.0)
df = df.dropna(subset=["tech_field","state","founding_year"]).copy()

# Collapse rare tech/state to "Other" AFTER US-only & clean merges
tech_keep  = set(df["tech_field"].value_counts()[lambda s: s>=TECH_MIN].index)
state_keep = set(df["state"].value_counts()[lambda s: s>=STATE_MIN].index)
df["tech_slim"]  = np.where(df["tech_field"].isin(tech_keep), df["tech_field"], "Other")
df["state_slim"] = np.where(df["state"].isin(state_keep),     df["state"],     "Other")

if MAX_N is not None and len(df) > MAX_N:
    df = df.sample(n=MAX_N, random_state=42)

# ---------------- 9) OLS Linear Probability Model (robust SE) ----------------
df["year_c"] = df["founding_year"] - df["founding_year"].mean()

X = pd.get_dummies(
    df[["innov_quantity","innov_quality","year_c","tech_slim","state_slim"]],
    columns=["tech_slim","state_slim"], drop_first=True
)
y = df["acquired"].astype(float)

X = sm.add_constant(X, has_constant="add")
X = X.apply(pd.to_numeric, errors="coerce").astype(float)
mask = X.notna().all(axis=1) & y.notna()
X, y = X.loc[mask], y.loc[mask]
# Drop zero-variance columns to avoid singularities
X = X.loc[:, (X.columns == "const") | (X.nunique() > 1)]

res = sm.OLS(y, X).fit(cov_type="HC1")
print(res.summary().as_text())

# Compact coefficient table for report
co, se = res.params, res.bse

def get(k):
    return (co.get(k, np.nan), se.get(k, np.nan))

rows = []
for k in ["const", "innov_quantity", "innov_quality", "year_c"]:
    b, s = get(k)
    rows.append([k, b, s, b * 100])  # percentage points

out = pd.DataFrame(rows, columns=["term", "coef", "se", "coef(pp)"])
print("\nKey coefficients (Δ acquisition probability; 'coef(pp)' = percentage points):")
print(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

# ===== Save to "Programming Test Task8" (Desktop -> Home -> Script) =====
from datetime import datetime
import os

def _permission_safe_folder(foldername: str):
    candidates = [
        os.path.join(os.path.expanduser("~"), "Desktop"),  # Desktop
        os.path.expanduser("~"),  # Home directory
        os.path.dirname(os.path.abspath(__file__)),  # Current script directory
    ]
    for base in candidates:
        try:
            if not os.path.isdir(base):
                continue
            target = os.path.join(base, foldername)
            os.makedirs(target, exist_ok=True)
            # test write permission
            probe = os.path.join(target, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return target
        except Exception:
            continue
    return None

_target = _permission_safe_folder("Programming Test Task8")
ts = datetime.now().strftime("%Y%m%d-%H%M%S")

if _target is not None:
    # 1) Compact coefficient table CSV
    coef_csv = os.path.join(_target, f"task8_key_coefficients_{ts}.csv")
    try:
        out.to_csv(coef_csv, index=False)
    except Exception:
        coef_csv = None

    # 2) Full regression summary + compact table TXT
    txt_path = os.path.join(_target, f"task8_model_summary_{ts}.txt")
    try:
        console_lines = []
        console_lines.append("=== Task 8: Linear Probability Model (OLS, HC1 robust) ===")
        console_lines.append(res.summary().as_text())
        console_lines.append("")
        console_lines.append("=== Key coefficients (Δ acquisition probability; coef(pp)=percentage points) ===")
        console_lines.append(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        console_lines.append("")
        console_lines.append("=== Columns in final design matrix (for reproducibility) ===")
        console_lines.append(", ".join(list(X.columns)))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(console_lines) + "\n")
    except Exception:
        txt_path = None

    # 3) Save design matrix column names separately (optional)
    cols_path = os.path.join(_target, f"task8_design_matrix_cols_{ts}.txt")
    try:
        with open(cols_path, "w", encoding="utf-8") as f:
            for c in X.columns:
                f.write(str(c) + "\n")
    except Exception:
        cols_path = None

    print("\n[Saved] Programming Test Task8 outputs:")
    if coef_csv: print(" -", coef_csv)
    if txt_path: print(" -", txt_path)
    if cols_path: print(" -", cols_path)
else:
    print("\n[Info] Could not find a writable folder (Desktop/Home/Script). Skipped saving files.")

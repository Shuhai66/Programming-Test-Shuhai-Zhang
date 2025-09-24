# ============================================
# Task 7 – Share of Acquired Startups by Founding Year (US-only, 1985–2010 safe)
# Aligned with Tasks 1–6
# --------------------------------------------
# Definitions:
#   - Startup = assignee; founding year = grant year of FIRST patent.
#   - US-only = use the EXACT assignee row on the FIRST patent to map location_id → loc,
#               keep country == "US" and non-missing state.
#   - Acquisition = at least one qualifying PA event within 10 years after the first patent date.
#       * Mode A (default, comprehensive): consider ANY patent of the startup within [t0, t0+10y].
#       * Mode B (conservative): consider ONLY the first patent itself.
#   - Safe right-censoring: set cohort end to min(requested_end, max_exec_year_in_pa - 10).
#
# Outputs:
#   - task7_acquisition_share_by_year.csv
#   - task7_acquisition_share_by_year.png
#   - Console summary saved to "Programming Test Task7" (Desktop → Home → script), if permitted.
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    load_csv, first_patents, Timer
)

# ---------------- Config ----------------
PATENT_PATH   = "../data/g_patent.csv"
ASSIGNEE_PATH = "../data/g_assignee_disambiguated.csv"
LOCATION_PATH = "../data/g_location_disambiguated.csv"
PA_PATH       = "../data/pa.csv"   # USPTO Patent Assignments (events)

START_YEAR = 1985
END_YEAR   = 2010            # nominal end-year before safety adjustment
WINDOW_YEARS = 10            # acquisition window length
PA_CHUNKSIZE = 2_000_000     # tune if needed

# Acquisition mode:
#   True  = consider ANY patent of the startup within 0–10y after the first patent (more complete)
#   False = consider ONLY the first patent's PA events (conservative)
ACQ_USE_ALL_STARTUP_PATENTS = True

# --- convey_type normalization / filtering rules ---
EXCLUDE_PATTERNS = (
    "namechg", "name chg", "change of name", "security", "security interest",
    "release", "correct", "correction", "license", "mortgage", "lien",
    "collateral", "re-record", "rrcord", "terminat", "reconvey", "surrender"
)
INCLUDE_TOKENS = ("assign", "merg")  # assignment / merger tokens

def _is_included_event(raw: str) -> bool:
    """Return True if convey_type string likely signals acquisition; False otherwise."""
    if pd.isna(raw):
        return False
    s = str(raw).strip().lower()
    # hard excludes first
    for bad in EXCLUDE_PATTERNS:
        if bad in s:
            return False
    # then includes
    return any(tok in s for tok in INCLUDE_TOKENS)

def _permission_safe_folder(foldername: str):
    """Return (path, True) if a writable folder can be created (Desktop → Home → Script), else (None, False)."""
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
            # probe write permission
            probe = os.path.join(target, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return target, True
        except Exception:
            continue
    return None, False


def compute_task7_acquisition_share():
    # 1) Load minimal tables
    with Timer("Load patents, assignees, locations"):
        pat = load_csv(PATENT_PATH,   usecols=["patent_id", "patent_date"])
        asg = load_csv(ASSIGNEE_PATH, usecols=["patent_id", "assignee_id", "location_id"])
        loc = load_csv(LOCATION_PATH, usecols=["location_id", "disambig_state", "disambig_country"])

        pat["patent_date"] = pd.to_datetime(pat["patent_date"], errors="coerce")
        # quick column checks
        for df, need, name in [
            (pat, {"patent_id", "patent_date"}, "g_patent.csv"),
            (asg, {"patent_id", "assignee_id", "location_id"}, "g_assignee_disambiguated.csv"),
            (loc, {"location_id", "disambig_state", "disambig_country"}, "g_location_disambiguated.csv"),
        ]:
            missing = need - set(df.columns)
            if missing:
                raise ValueError(f"[{name}] missing columns: {sorted(missing)}")

    # 2) First patents per assignee
    with Timer("Identify first patents per assignee"):
        fp = first_patents(patent_df=pat, assignee_df=asg)  # expects: first_patent_id, first_patent_date, first_patent_year

    # 3) US-only filter using the EXACT assignee row on the FIRST patent
    with Timer("US-only on the FIRST-patent assignee row"):
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
        fp_us = fp_loc[is_us].copy()
        if fp_us.empty:
            raise ValueError("After US-only filter, no startups remain. Check inputs.")
        # Initial window (1985–2010 nominal)
        fp_us = fp_us[(fp_us["first_patent_year"] >= START_YEAR) & (fp_us["first_patent_year"] <= END_YEAR)].copy()

    # 4) Scan pa.csv to get max exec_dt year and set safe cohort end
    with Timer("Scan pa.csv for max exec_dt (safe window)"):
        max_exec_year = None
        usecols = ["patent_id", "exec_dt"]  # minimal to find max date
        for ch in pd.read_csv(PA_PATH, usecols=usecols, chunksize=PA_CHUNKSIZE):
            d = pd.to_datetime(ch["exec_dt"], errors="coerce")
            if len(d):
                y = d.dt.year.max()
                if pd.notna(y):
                    y = int(y)
                    max_exec_year = y if max_exec_year is None else max(max_exec_year, y)
        if max_exec_year is None:
            raise ValueError("Could not parse any exec_dt from pa.csv; cannot determine safe window.")
        safe_end_year = min(END_YEAR, max_exec_year - WINDOW_YEARS)
        if safe_end_year < START_YEAR:
            raise ValueError(f"pa.csv seems too old for a {WINDOW_YEARS}y window (max exec year={max_exec_year}).")
        # apply safe end
        fp_us = fp_us[fp_us["first_patent_year"].between(START_YEAR, safe_end_year)].copy()

    # 5) Build startup→(patents in 0–10y) mapping if using comprehensive mode
    if ACQ_USE_ALL_STARTUP_PATENTS:
        with Timer("Build per-startup patent set within 0–10y window"):
            # Join asg→pat to get each (assignee_id, patent_id, patent_date)
            asg_pat = asg.merge(pat, on="patent_id", how="left")
            # Bring in each startup's first date
            map_df = asg_pat.merge(fp_us[["assignee_id", "first_patent_date"]], on="assignee_id", how="inner")
            # Keep patents in [t0, t0+10y]
            map_df["patent_date"] = pd.to_datetime(map_df["patent_date"], errors="coerce")
            in_win = (map_df["patent_date"] >= map_df["first_patent_date"]) & \
                     (map_df["patent_date"] <= (map_df["first_patent_date"] + pd.DateOffset(years=WINDOW_YEARS)))
            map_df = map_df.loc[in_win, ["assignee_id", "patent_id", "first_patent_date"]].dropna().drop_duplicates()
            # We'll need patent_id → startup (assignee_id) + first date
            pat_to_startup = map_df  # columns: assignee_id, patent_id, first_patent_date
            candidate_patent_ids = set(pat_to_startup["patent_id"].unique())
    else:
        # Conservative: only the first patent of each startup
        pat_to_startup = fp_us[["assignee_id", "first_patent_id", "first_patent_date"]].rename(
            columns={"first_patent_id": "patent_id"}
        )
        candidate_patent_ids = set(pat_to_startup["patent_id"].unique())

    # 6) Stream pa.csv and flag acquisitions
    with Timer("Flag acquisitions from pa.csv (streaming)"):
        acquired = pd.Series(False, index=fp_us["assignee_id"].unique(), name="acquired")
        usecols = ["patent_id", "exec_dt", "convey_type"]
        for ch in pd.read_csv(PA_PATH, usecols=usecols, chunksize=PA_CHUNKSIZE):
            # Keep only candidate patents to reduce work
            ch = ch[ch["patent_id"].isin(candidate_patent_ids)]
            if ch.empty:
                continue
            # Filter to meaningful acquisition-like events
            mask_inc = ch["convey_type"].apply(_is_included_event)
            ch = ch[mask_inc]
            if ch.empty:
                continue
            # Map events → startups
            m = ch.merge(pat_to_startup, on="patent_id", how="left")
            if m.empty:
                continue
            evt_date = pd.to_datetime(m["exec_dt"], errors="coerce")
            within_10y = (evt_date >= m["first_patent_date"]) & \
                         (evt_date <= (m["first_patent_date"] + pd.DateOffset(years=WINDOW_YEARS)))
            if within_10y.any():
                hit = m.loc[within_10y, "assignee_id"].dropna().unique()
                acquired.loc[hit] = True

    # 7) Aggregate by founding year
    with Timer("Aggregate acquisition share by founding year"):
        res = (fp_us
               .merge(acquired.rename("acquired"), left_on="assignee_id", right_index=True, how="left")
               .assign(acquired=lambda df: df["acquired"].fillna(False).astype(bool))
               .groupby("first_patent_year", as_index=False)
               .agg(startups=("assignee_id", "nunique"),
                    acquired=("acquired", "sum"))
               .rename(columns={"first_patent_year": "founding_year"})
               .sort_values("founding_year")
               .reset_index(drop=True))
        res["acquired_share"] = res["acquired"] / res["startups"]

    # 8) Save CSV + Figure + Console (permission-safe)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_csv = "task7_acquisition_share_by_year.csv"
    out_png = "task7_acquisition_share_by_year.png"
    res.to_csv(out_csv, index=False)

    # Pretty console print
    printable = res.copy()
    printable["acquired_share(%)"] = (printable["acquired_share"] * 100).round(2)
    printable = printable[["founding_year", "startups", "acquired", "acquired_share(%)"]]

    header = (f"=== Task 7: Share of Acquired US Startups by Founding Year "
              f"({START_YEAR}–{min(END_YEAR, safe_end_year)}) ===\n"
              f"Mode: {'ALL patents within 10y' if ACQ_USE_ALL_STARTUP_PATENTS else 'FIRST patent only'}\n"
              f"(Safe-end based on pa.csv max exec_dt year = {max_exec_year})\n")
    print("\n" + header)
    print(printable.to_string(index=False))

    # Save permission-safe text + figure
    target_dir, ok = _permission_safe_folder("Programming Test Task7")
    if ok and target_dir:
        try:
            txt_path = os.path.join(target_dir, f"task7_console_output_{ts}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(printable.to_string(index=False))
                f.write("\n")
        except Exception:
            txt_path = None
    else:
        txt_path = None

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(res["founding_year"], res["acquired_share"], marker="o")
    plt.title("Share of Acquired US Startups by Founding Year")
    plt.xlabel("Founding Year")
    plt.ylabel("Acquired Share")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save locally
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    # Save to permission-safe folder too
    if ok and target_dir:
        try:
            fig_path = os.path.join(target_dir, f"task7_acquisition_share_{ts}.png")
            plt.savefig(fig_path, dpi=180, bbox_inches="tight")
            print(f"\n[Saved] Files:")
            if txt_path: print(f" - TXT: {txt_path}")
            print(f" - FIG: {fig_path}")
        except Exception:
            pass

    # Show for 10 seconds then close
    plt.show(block=False)
    plt.pause(10)
    plt.close()

    # Always tell where the main CSV/PNG are
    print(f"\nSaved table: {out_csv}")
    print(f"Saved figure: {out_png}")

    return res


if __name__ == "__main__":
    compute_task7_acquisition_share()

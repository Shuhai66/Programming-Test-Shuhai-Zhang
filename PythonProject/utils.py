import pandas as pd

# ---------------------------
# General purpose functions
# ---------------------------



def load_csv(path, usecols=None, **kwargs):
        # engine='python' + sep=None 让 pandas 自动猜测分隔符（逗号/制表符等）
        return pd.read_csv(path, sep=None, engine='python', usecols=usecols, **kwargs)


def first_patent_year(patent_df):
    """
    Given g_patent, return DataFrame with assignee_id and its first patent year.
    """
    patent_df["patent_date"] = pd.to_datetime(patent_df["patent_date"], errors="coerce")
    patent_df["year"] = patent_df["patent_date"].dt.year
    firsts = patent_df.groupby("assignee_id")["year"].min().reset_index()
    firsts.rename(columns={"year": "founding_year"}, inplace=True)
    #print("DEBUG first_patent_year:", firsts.head()) 老代码
    return firsts


def count_startups_by_year(firsts):
    """
    Count number of startups by founding year.
    """
    counts = firsts.groupby("founding_year").size().reset_index(name="num_startups")
    #print("DEBUG first_patent_year:", counts.head()) 老代码
    return counts


def merge_location(patent_df, location_df):
    """
    Merge patent info with location data (state level).
    """
    merged = patent_df.merge(location_df, on="location_id", how="left")
    #print("DEBUG first_patent_year:", merged.head()) 老代码
    return merged


def merge_cpc(patent_df, cpc_df):
    """
    Merge patent info with CPC technology classification.
    """
    merged = patent_df.merge(cpc_df, on="patent_id", how="left")
    #print("DEBUG first_patent_year:", merged.head()) 老代码
    return merged


def citations_within_years(cite_df, patent_df, years=5):
    """
    Count forward citations within X years after first patent.
    - cite_df: citation table
    - patent_df: must contain patent_id, patent_date
    """
    patent_df["patent_date"] = pd.to_datetime(patent_df["patent_date"], errors="coerce")
    cite_df = cite_df.merge(patent_df[["patent_id", "patent_date"]],
                            left_on="cited_patent_id", right_on="patent_id",
                            how="left", suffixes=("", "_cited"))
    cite_df = cite_df.merge(patent_df[["patent_id", "patent_date"]],
                            left_on="citing_patent_id", right_on="patent_id",
                            how="left", suffixes=("", "_citing"))

    # citation time difference
    cite_df["time_diff"] = (cite_df["patent_date_citing"] - cite_df["patent_date_cited"]).dt.days / 365
    filtered = cite_df[cite_df["time_diff"].between(0, years)]
    counts = filtered.groupby("cited_patent_id").size().reset_index(name="forward_citations")
    #print("DEBUG first_patent_year:", counts.head()) 老代码
    return counts


def acquisition_within_years(pa_df, years=10):
    """
    Identify acquisitions: type = assignment/merger within X years of first patent.
    """
    pa_df["record_date"] = pd.to_datetime(pa_df["record_date"], errors="coerce")
    acq = pa_df[pa_df["type"].isin(["assignment", "merger"])].copy()
    # Example: later you merge this with founding_year to filter within 10 years
    #print("DEBUG first_patent_year:", acq.head()) 老代码
    return acq




def first_patents(patent_df, assignee_df):
    """
    Identify the first patent of each assignee (startup).

    Returns
    -------
    DataFrame with:
      ['assignee_id', 'first_patent_id', 'first_patent_year', 'first_patent_date']
    """
    # ensure datetime
    tmp_pat = patent_df[['patent_id', 'patent_date']].copy()
    tmp_pat['patent_date'] = pd.to_datetime(tmp_pat['patent_date'], errors='coerce')
    tmp_pat['year'] = tmp_pat['patent_date'].dt.year

    # link each patent to an assignee
    tmp = assignee_df[['patent_id', 'assignee_id']].dropna().copy()
    tmp = tmp.merge(tmp_pat, on='patent_id', how='left')

    # first patent per assignee by earliest date
    idx = (tmp.sort_values(['assignee_id', 'patent_date'])
           .groupby('assignee_id', as_index=False).head(1))

    out = idx[['assignee_id', 'patent_id', 'year', 'patent_date']].rename(
        columns={'patent_id': 'first_patent_id',
                 'year': 'first_patent_year',
                 'patent_date': 'first_patent_date'}
    )
    return out

def task4_top_cpc_class_by_startups(
    patent_path="g_patent.csv",
    assignee_path="g_assignee_disambiguated.csv",
    cpc_path="g_cpc_current.csv",
    year_min=1985,
    year_max=2015
):
    """Return counts and top10 CPC classes for startups founded in [year_min, year_max]."""
    # Load minimal columns; DO NOT pass low_memory because load_csv uses engine='python'
    pat = load_csv(patent_path, usecols=["patent_id", "patent_date"])
    asg = load_csv(assignee_path, usecols=["patent_id", "assignee_id"])
    cpc = load_csv(cpc_path, usecols=["patent_id", "cpc_class"])

    # Ensure cpc_class exists
    if "cpc_class" not in cpc.columns:
        raise ValueError(
            "g_cpc_current.csv must contain a column named 'cpc_class'. "
            "If your file uses a different name, tell me the exact column name."
        )

    # First patent per assignee (startup definition)
    first = first_patents(patent_df=pat, assignee_df=asg)

    # Keep startups founded within the assignment window
    first = first[(first["first_patent_year"] >= year_min) & (first["first_patent_year"] <= year_max)]

    # Attach CPC class of the first patent (counts a startup once per class if multiple classes exist)
    first_with_cpc = first.merge(
        cpc[["patent_id", "cpc_class"]],
        left_on="first_patent_id",
        right_on="patent_id",
        how="left"
    ).dropna(subset=["cpc_class"])

    # Count startups per CPC class
    counts = (
        first_with_cpc
        .groupby("cpc_class", as_index=False)
        .size()
        .rename(columns={"size": "num_startups"})
        .sort_values("num_startups", ascending=False)
        .reset_index(drop=True)
    )

    # Print Top 10 for quick inspection
    top10 = counts.head(10)
    print("\n[Task 4] Top 10 CPC classes by number of startups (1985–2015):")
    print(top10.to_string(index=False))

    return counts, top10

# ===========================
# Utils for Task 6 (cohort comparison)
# ===========================

import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Lightweight timer (safe to reuse across tasks)
# ---------------------------------------------------------------------
class Timer:
    """Simple context manager to time steps (pure utility, no side effects)."""
    def __init__(self, label):
        self.label = label
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        print(f"\n[START] {self.label} ...")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        print(f"[DONE ] {self.label} in {dt:,.1f}s")

# ---------------------------------------------------------------------
# Citation loader that normalizes schemas to ['citing_patent_id','cited_patent_id']
# Works with either (citing_patent_id, cited_patent_id) or (patent_id, citation_patent_id)
# ---------------------------------------------------------------------
def load_and_standardize_cites(path, chunksize=None):
    """
    Stream a citation CSV yielding standardized chunks with columns:
    ['citing_patent_id', 'cited_patent_id'].
    Designed to handle PatentsView variants.
    """
    # Preferred schema
    try:
        it = pd.read_csv(path, sep=None, engine="python",
                         usecols=["citing_patent_id", "cited_patent_id"],
                         chunksize=chunksize)
        for c in it:
            yield c[["citing_patent_id", "cited_patent_id"]]
        return
    except Exception:
        pass
    # Fallback schema
    it = pd.read_csv(path, sep=None, engine="python",
                     usecols=["patent_id", "citation_patent_id"],
                     chunksize=chunksize)
    for c in it:
        c = c.rename(columns={"patent_id": "citing_patent_id",
                              "citation_patent_id": "cited_patent_id"})
        yield c[["citing_patent_id", "cited_patent_id"]]

# ---------------------------------------------------------------------
# Innovation QUANTITY: number of patents by the same assignee within [0, window] years
# Inputs:
#   pat_df: ['patent_id','patent_date']
#   asg_df: ['patent_id','assignee_id']
#   fp_df : output of first_patents(...), must have ['assignee_id','first_patent_date']
# Output:
#   DataFrame ['assignee_id','innov_quantity']
# ---------------------------------------------------------------------
def compute_innov_quantity(pat_df, asg_df, fp_df, window_years=5):
    """Count patents of the same assignee within [0, window_years] from first_patent_date."""
    tmp_pat = pat_df[["patent_id", "patent_date"]].copy()
    tmp_pat["patent_date"] = pd.to_datetime(tmp_pat["patent_date"], errors="coerce")

    # Attach dates: each (assignee, patent) ± first date for that assignee
    pat_all = asg_df.merge(tmp_pat, on="patent_id", how="left")
    pat_all = pat_all.merge(fp_df[["assignee_id", "first_patent_date"]], on="assignee_id", how="inner")

    # Compute delta in years and count in-window patents
    pat_all["dt_years"] = (pat_all["patent_date"] - pat_all["first_patent_date"]).dt.days / 365.0
    mask = pat_all["dt_years"].between(0, window_years, inclusive="both")

    qty = (pat_all.loc[mask]
           .groupby("assignee_id")
           .size()
           .rename("innov_quantity")
           .reset_index())

    # Backfill zeros for startups without any in-window patents
    qty = fp_df[["assignee_id"]].merge(qty, on="assignee_id", how="left").fillna({"innov_quantity": 0})
    return qty

# ---------------------------------------------------------------------
# Innovation QUALITY: forward citations to the FIRST patent within [0, window] years
# Inputs:
#   fp_df: must have ['assignee_id','first_patent_id','first_patent_date']
#   pat_df: ['patent_id','patent_date']  (for citing dates)
#   cite_paths: list of file paths to citation CSVs
# Output:
#   DataFrame ['assignee_id','innov_quality']
# ---------------------------------------------------------------------
def compute_innov_quality_from_cites(fp_df, pat_df, cite_paths, window_years=5, chunksize=1_000_000):
    """Count citations to first_patent_id within [0, window_years] across streamed citation files."""
    # Build fast lookups
    pat_map = (pat_df.drop_duplicates("patent_id")
                     .set_index("patent_id")["patent_date"])
    pat_map = pd.to_datetime(pat_map, errors="coerce")

    first_date_map = (fp_df.drop_duplicates("first_patent_id")
                         .set_index("first_patent_id")["first_patent_date"])

    fc_counts = {}  # first_patent_id -> count within window

    for path in cite_paths:
        for chunk in load_and_standardize_cites(path, chunksize=chunksize):
            # Keep only citations where the cited patent is a first patent we track
            is_first = chunk["cited_patent_id"].isin(first_date_map.index)
            if not is_first.any():
                continue
            sub = chunk.loc[is_first].copy()

            # Map dates for citing and first (cited) patent
            sub["citing_patent_date"] = pd.to_datetime(sub["citing_patent_id"].map(pat_map), errors="coerce")
            sub["first_patent_date"]  = sub["cited_patent_id"].map(first_date_map)

            # Time window filter
            dt = (sub["citing_patent_date"] - sub["first_patent_date"]).dt.days / 365.0
            keep = dt.between(0, window_years, inclusive="both")
            if keep.any():
                counts = sub.loc[keep].groupby("cited_patent_id").size()
                for k, v in counts.items():
                    fc_counts[k] = fc_counts.get(k, 0) + int(v)

    qual = (pd.DataFrame({"first_patent_id": list(fc_counts.keys()),
                          "innov_quality":   list(fc_counts.values())})
            if fc_counts else
            pd.DataFrame(columns=["first_patent_id","innov_quality"]))

    qual = fp_df[["assignee_id", "first_patent_id"]].merge(qual, on="first_patent_id", how="left")
    qual["innov_quality"] = qual["innov_quality"].fillna(0)
    return qual

# ---------------------------------------------------------------------
# Assemble per-startup table used by Task 6
# Output columns: ['assignee_id','founding_year','innov_quantity','innov_quality']
# ---------------------------------------------------------------------
def assemble_per_startup(fp_df, qty_df, qual_df):
    """Merge first-patent info with quantity/quality metrics per startup."""
    per = (fp_df[["assignee_id", "first_patent_year"]]
           .merge(qty_df,  on="assignee_id", how="left")
           .merge(qual_df, on="assignee_id", how="left")
           .rename(columns={"first_patent_year": "founding_year"}))
    # Ensure numeric types
    per["innov_quantity"] = per["innov_quantity"].astype(float)
    per["innov_quality"]  = per["innov_quality"].astype(float)
    return per

# ---------------------------------------------------------------------
# Cohort helpers: split and describe
# ---------------------------------------------------------------------
def split_cohorts(per_startup, a_range=(1995, 1999), b_range=(2003, 2007)):
    """
    Split per_startup into two cohorts by founding_year (inclusive ranges).
    Returns (cohort_a, cohort_b).
    """
    a_lo, a_hi = a_range
    b_lo, b_hi = b_range
    coh_a = per_startup.query("founding_year >= @a_lo and founding_year <= @a_hi").copy()
    coh_b = per_startup.query("founding_year >= @b_lo and founding_year <= @b_hi").copy()
    return coh_a, coh_b

def describe_two_groups(a_df, b_df, col):
    """Return simple descriptive stats dict for a metric in two cohorts."""
    return {
        "A_mean": float(np.mean(a_df[col])),
        "A_sd":   float(np.std(a_df[col], ddof=1)),
        "A_n":    int(a_df[col].shape[0]),
        "B_mean": float(np.mean(b_df[col])),
        "B_sd":   float(np.std(b_df[col], ddof=1)),
        "B_n":    int(b_df[col].shape[0]),
    }

# ---------------------------------------------------------------------
# Significance tests (Welch t-test and Mann–Whitney U)
# We import scipy lazily inside functions to avoid hard runtime dependency at module import.
# ---------------------------------------------------------------------
def welch_t(a_df, b_df, col):
    """Welch’s t-test (unequal variances) for two independent samples."""
    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required for welch_t; please install scipy.") from e
    return stats.ttest_ind(a_df[col], b_df[col], equal_var=False, nan_policy="omit")

def mann_whitney_u(a_df, b_df, col):
    """Two-sided Mann–Whitney U test (nonparametric robustness)."""
    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required for mann_whitney_u; please install scipy.") from e
    return stats.mannwhitneyu(a_df[col], b_df[col], alternative="two-sided")

# ---------------------------------------------------------------------
# One-shot summary for Task 6 (returns a tidy DataFrame you can print or save)
# ---------------------------------------------------------------------
def summarize_cohort_comparison(per_startup, a_range=(1995, 1999), b_range=(2003, 2007)):
    """
    Produce a compact comparison table for Task 6 on both metrics:
    'innov_quantity' and 'innov_quality'.
    """
    coh_a, coh_b = split_cohorts(per_startup, a_range=a_range, b_range=b_range)

    if len(coh_a) == 0 or len(coh_b) == 0:
        raise ValueError("Empty cohort detected. Check founding_year ranges and per_startup content.")

    # Descriptives
    d_qty  = describe_two_groups(coh_a, coh_b, "innov_quantity")
    d_qual = describe_two_groups(coh_a, coh_b, "innov_quality")

    # Tests
    t_qty  = welch_t(coh_a, coh_b, "innov_quantity")
    t_qual = welch_t(coh_a, coh_b, "innov_quality")
    u_qty  = mann_whitney_u(coh_a, coh_b, "innov_quantity")
    u_qual = mann_whitney_u(coh_a, coh_b, "innov_quality")

    rows = []
    rows.append({
        "metric": "innov_quantity (0–5y patents)",
        "A_mean": d_qty["A_mean"],  "A_sd": d_qty["A_sd"],  "A_n": d_qty["A_n"],
        "B_mean": d_qty["B_mean"],  "B_sd": d_qty["B_sd"],  "B_n": d_qty["B_n"],
        "Welch_t": float(t_qty.statistic),  "Welch_p": float(t_qty.pvalue),
        "MWU_U":   float(u_qty.statistic),  "MWU_p":   float(u_qty.pvalue),
    })
    rows.append({
        "metric": "innov_quality (0–5y citations)",
        "A_mean": d_qual["A_mean"], "A_sd": d_qual["A_sd"], "A_n": d_qual["A_n"],
        "B_mean": d_qual["B_mean"], "B_sd": d_qual["B_sd"], "B_n": d_qual["B_n"],
        "Welch_t": float(t_qual.statistic), "Welch_p": float(t_qual.pvalue),
        "MWU_U":   float(u_qual.statistic), "MWU_p":   float(u_qual.pvalue),
    })
    return pd.DataFrame(rows)

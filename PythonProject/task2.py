import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_csv, first_patents
import os
from datetime import datetime

# ---------------------------
# 1) Load data (minimal cols)
# ---------------------------
pat = load_csv("../data/g_patent.csv", usecols=['patent_id', 'patent_date'])
ass = pd.read_csv("../data/g_assignee_disambiguated.csv", header=0, usecols=[0, 2, 7])
ass.columns = ['patent_id', 'assignee_id', 'location_id']

# ---------------------------
# 2) Filter for US startups only
# ---------------------------
US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
    "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
    "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
    "WI","WY","DC"
}

# Location filtering for US only
loc = load_csv("../data/g_location_disambiguated.csv", usecols=["location_id", "disambig_state", "disambig_country"])
us_loc = loc[loc["disambig_country"] == "US"]
ass_us = ass.merge(us_loc[["location_id"]], on="location_id", how="inner")

# ---------------------------
# 3) First patents
# ---------------------------
fp = first_patents(pat[['patent_id', 'patent_date']], ass_us[['patent_id', 'assignee_id']])

# ---------------------------
# 4) Build the US patents universe (denominator)
# ---------------------------
# Map patent_id -> location_id via assignee table, then filter to US locations
pat_loc = ass.merge(us_loc[["location_id"]], on="location_id", how="inner")[["patent_id"]]
pat_us_ids = pat_loc.drop_duplicates()["patent_id"]  # patents that have at least one US location

# Keep only US patents in the patent table (for denominator and year list)
pat_us = pat[pat["patent_id"].isin(pat_us_ids)].copy()

# Ensure 'year' exists in pat_us
pat_us['patent_date'] = pd.to_datetime(pat_us['patent_date'], errors='coerce')
pat_us['year'] = pat_us['patent_date'].dt.year.astype('Int64')  # 为每个专利记录生成年份

# ---------------------------
# 5) Count totals (US patents) and first patents (US startups) per year
# ---------------------------
# Map first-patent year back to each patent_id (only for US startups)
first_patent_years = fp[["first_patent_id", "first_patent_year"]].rename(
    columns={"first_patent_id": "patent_id"}
)

# For tagging "is first" we only need US patents (denominator universe)
is_first = pat_us[["patent_id", "year"]].merge(first_patent_years, on="patent_id", how="left")
# Using nullable Int64 to handle missing values (NaN or NaT)
is_first["is_first"] = (is_first["first_patent_year"] == is_first["year"]).astype("Int64")

# Compute per-year totals using US patents only
years = sorted(pat_us["year"].dropna().unique())
results = []
for y in tqdm(years, desc="Computing per-year stats (US only)"):
    # denominator: all US patents granted in year y
    total_us = (pat_us["year"] == y).sum()
    # numerator: US startups' first patents granted in year y
    first_us = is_first.loc[is_first["year"] == y, "is_first"].sum()
    results.append((y, int(total_us), int(first_us)))

res = pd.DataFrame(results, columns=["year", "all_patents_us", "first_patents_us"])
res = res[(res["year"] >= 1985) & (res["year"] <= 2015)].copy()
res["share_first"] = res["first_patents_us"] / res["all_patents_us"]

# ---------------------------
# 6) Print and plot + robust, cross-platform saving (optional)
# ---------------------------
import os
from datetime import datetime

# 1) Print result to console
print("\nShare of US startups' first patents among ALL US patents (1985–2015):")
print(res[(res["year"] >= 1985) & (res["year"] <= 2015)].to_string(index=False))

# 2) Make the plot
plt.figure()
plt.plot(res["year"], res["share_first"])
plt.title("Share of US Startups' First Patents Among ALL US Patents by Year")
plt.xlabel("Year")
plt.ylabel("Share (First / All, US only)")
plt.tight_layout()

# 3) Try to save figure + console table into a folder:
#    Prefer Desktop -> Home -> Script folder. If all fail, silently skip saving.
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
folder_name = "Programming Test Task2"
save_enabled = False
target_dir = None

# Candidate base directories in priority order
candidates = [
    os.path.join(os.path.expanduser("~"), "Desktop"),  # Desktop (macOS/Windows)
    os.path.expanduser("~"),                           # Home
    os.path.dirname(os.path.abspath(__file__)),        # Script folder
]

for base in candidates:
    try:
        if not os.path.isdir(base):
            continue
        target_dir = os.path.join(base, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        # touch a file to test write permission
        _probe = os.path.join(target_dir, ".write_probe")
        with open(_probe, "w", encoding="utf-8") as _f:
            _f.write("ok")
        os.remove(_probe)
        save_enabled = True
        break
    except Exception:
        # Try next candidate silently
        continue

# 4) Save if possible; otherwise skip silently
if save_enabled:
    try:
        txt_path = os.path.join(target_dir, f"task2_console_output_{timestamp}.txt")
        fig_path = os.path.join(target_dir, f"task2_figure_{timestamp}.png")

        # Save the table (1985–2015 window)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Share of US startups' first patents among ALL US patents (1985–2015):\n")
            f.write(res[(res["year"] >= 1985) & (res["year"] <= 2015)].to_string(index=False))

        # Save the figure (PNG)
        plt.savefig(fig_path)

        print(f"\nSaved files:\n- Console output: {txt_path}\n- Figure: {fig_path}")
    except Exception:
        # Silently ignore any saving error (permissions, disk, etc.)
        pass

# 5) Show plot for 10 seconds (or 30 if you prefer), then close and exit
plt.show(block=False)
plt.pause(10)  # change to 30 if you want more time
plt.close()

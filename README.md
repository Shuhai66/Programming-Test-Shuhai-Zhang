This repository reproduces analyses of U.S. startups and patents (1985–2015).
Python scripts load CSVs, compute metrics, and write summary tables/figures/plots.

Research Bias/
├─ data/                  # <== Place CSVs here (not included in repo)
│   └─ .gitkeep           # placeholder file so GitHub keeps this folder
├─ PythonProject/         # Code: main.py, task1.py … task8.py, utils.py
└─ Programming Report.pdf

Data Files (not uploaded)：
Due to file size, CSVs are not stored in GitHub. Please place the following files into the data/ folder:

g_assignee_disambiguated.csv
g_cpc_current.csv
g_location_disambiguated.csv
g_patent.csv
g_us_patent_citation_1.csv
g_us_patent_citation_2.csv
pa.csv

If your data are in another location, update the BASE_DIR variable at the top of each script.
BASE_DIR = "../data"      # default for repo layout

Requirements:
pandas
numpy
matplotlib
tqdm
statsmodels

pip install -r requirements.txt

How to Run:
From the repo root:
cd "Research Bias/PythonProject"
python task1.py
python task2.py
...
python task8.py

Each script runs independently and assumes CSVs are in ../data. Outputs (CSV/plots with timestamps) are saved in PythonProject/outputs/.






from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
TAB = ROOT / "tableau"
SQL = ROOT / "sql"
DB = DATA / "ecom.db"
SEED = 42
# UCI Online Retail II: 2009-12-01 to 2011-12-09 (Dec 2011 is a partial month)
DATA_END = "2011-12-09"
SNAPSHOT = "2011-12-10"      # RFM snapshot (day after last transaction)
COHORT_END = "2011-11-30"    # last full month; Dec 2011 excluded from cohort retention
LTV_CUTOFF = "2011-06-09"    # features use history up to here
LTV_END = "2011-12-09"       # target: revenue in the following 6 months (183 days)

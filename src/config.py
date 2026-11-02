"""Paths and settings shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"            # put the Lending Club CSV here
PROCESSED = ROOT / "data" / "processed" / "loans.csv"
TABLES = ROOT / "outputs" / "tables"
CHARTS = ROOT / "outputs" / "charts"
SQL_DIR = ROOT / "sql"

# Loans issued in these years have all finished (paid off or charged off),
# so every loan has a known final outcome.
YEARS = (2007, 2011)

# The most recent year is held out as an out-of-time test set:
# the model is trained on older loans and tested on newer ones, like in real life.
TEST_YEAR = 2011

# If the data doesn't let us estimate it, assume we lose 60% of a defaulted loan.
DEFAULT_LGD = 0.60

for d in (PROCESSED.parent, TABLES, CHARTS):
    d.mkdir(parents=True, exist_ok=True)

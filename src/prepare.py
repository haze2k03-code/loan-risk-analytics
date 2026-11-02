"""Step 1: load the raw Lending Club CSV, clean it, and save one tidy table.

Run:  python src/prepare.py

What it does
- keeps only finished loans (Fully Paid / Charged Off) issued in 2007-2011
- defines the target: is_default = 1 if the loan was charged off
- converts messy text columns ("36 months", "10.65%", "10+ years") into numbers
- builds a few features a lender would know at application time
- keeps post-loan columns only under a "post_" prefix, so they can be used for
  analysis (months to default, loss given default) but never for the model

Why the "post_" rule matters (data leakage):
columns like total_pymnt and recoveries are only known AFTER the loan ends.
A model that sees them looks near-perfect in testing but is useless for a new
applicant, because those numbers don't exist yet at application time.
"""
import glob
import sys

import numpy as np
import pandas as pd

from config import PROCESSED, RAW_DIR, YEARS

# Columns known when the customer applies (safe for the model)
APPLICATION_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "addr_state", "dti",
    "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "issue_d",
]
# Columns only known after the loan (analysis only, never model features)
POST_LOAN_COLS = ["loan_status", "total_pymnt", "recoveries", "last_pymnt_d"]


def find_raw_file() -> str:
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")) + glob.glob(str(RAW_DIR / "*.csv.gz")))
    if not files:
        sys.exit(f"No CSV found in {RAW_DIR}. Download the Lending Club data and put the CSV there (see README).")
    return files[0]


def read_raw(path: str) -> pd.DataFrame:
    # Some Lending Club exports have a one-line note above the header; detect and skip it.
    header = pd.read_csv(path, nrows=0).columns
    skip = 0 if "loan_amnt" in header else 1
    wanted = set(APPLICATION_COLS + POST_LOAN_COLS)
    df = pd.read_csv(path, skiprows=skip, low_memory=False,
                     usecols=lambda c: c in wanted)
    missing = [c for c in ["loan_amnt", "term", "issue_d", "loan_status", "annual_inc"] if c not in df.columns]
    if missing:
        sys.exit(f"The CSV is missing required columns: {missing}")
    return df


def to_number(series: pd.Series) -> pd.Series:
    """'10.65%' -> 10.65, ' 36 months' -> 36, already-numeric stays numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def parse_month(series: pd.Series) -> pd.Series:
    """Handles 'Dec-11', 'Dec-2011' and '2011-12-01'."""
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    out = out.fillna(pd.to_datetime(s, format="%b-%y", errors="coerce"))
    out = out.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
    return out


def emp_length_years(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    years = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    years[s.str.startswith("<")] = 0
    return years


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Outcome: strip the "Does not meet the credit policy. Status:" prefix some rows have
    df["loan_status"] = df["loan_status"].astype(str).str.replace(
        r"^Does not meet the credit policy\.\s*Status:", "", regex=True).str.strip()
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]
    df["is_default"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)

    # Dates
    df["issue_d"] = parse_month(df["issue_d"])
    df = df[df["issue_d"].dt.year.between(*YEARS)]
    df["issue_year"] = df["issue_d"].dt.year
    df["issue_month"] = df["issue_d"].dt.to_period("M").dt.to_timestamp()

    # Numbers stored as text
    for col in ["term", "int_rate", "revol_util", "loan_amnt", "installment", "annual_inc", "dti",
                "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc"]:
        if col in df.columns:
            df[col] = to_number(df[col])
    df = df.rename(columns={"term": "term_months"})
    if "emp_length" in df.columns:
        df["emp_length_years"] = emp_length_years(df["emp_length"])
        df = df.drop(columns="emp_length")

    # Credit history length at application. Two-digit years like "Jan-62" can parse as 2062, so fix those.
    if "earliest_cr_line" in df.columns:
        ecl = parse_month(df["earliest_cr_line"])
        ecl = ecl.where(ecl <= df["issue_d"], ecl - pd.DateOffset(years=100))
        df["credit_history_years"] = ((df["issue_d"] - ecl).dt.days / 365.25).round(1)
        df = df.drop(columns="earliest_cr_line")

    # Features a lender would compute at application
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)).round(3)
    df["income_band"] = pd.cut(
        df["annual_inc"], bins=[-np.inf, 30_000, 50_000, 75_000, 100_000, np.inf],
        labels=["<30k", "30-50k", "50-75k", "75-100k", "100k+"]).astype(str)
    df["loan_band"] = pd.cut(
        df["loan_amnt"], bins=[-np.inf, 5_000, 10_000, 15_000, 25_000, np.inf],
        labels=["<5k", "5-10k", "10-15k", "15-25k", "25k+"]).astype(str)

    # Post-loan columns, renamed so nobody mistakes them for features
    if "last_pymnt_d" in df.columns:
        last = parse_month(df["last_pymnt_d"])
        df["post_months_on_book"] = ((last.dt.year - df["issue_d"].dt.year) * 12
                                     + (last.dt.month - df["issue_d"].dt.month))
        df = df.drop(columns="last_pymnt_d")
    for col in ["total_pymnt", "recoveries"]:
        if col in df.columns:
            df = df.rename(columns={col: f"post_{col}"})
            df[f"post_{col}"] = to_number(df[f"post_{col}"])

    # Basic sanity filters: remove rows that can't be real applications
    df = df[(df["loan_amnt"] > 0) & (df["annual_inc"] > 0)]
    df = df.drop_duplicates()
    df.insert(0, "loan_id", range(1, len(df) + 1))
    return df


def main():
    path = find_raw_file()
    print(f"Reading {path}")
    raw = read_raw(path)
    print(f"  raw rows: {len(raw):,}")
    df = clean(raw)
    df.to_csv(PROCESSED, index=False, date_format="%Y-%m-%d")
    print(f"  clean rows: {len(df):,}  |  default rate: {df['is_default'].mean():.1%}")
    print(f"Saved {PROCESSED}")


if __name__ == "__main__":
    main()

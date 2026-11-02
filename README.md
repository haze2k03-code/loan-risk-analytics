# Personal Loan Risk & Collections Analytics

**Who defaults, how much risk should a lender take, and where should collections focus first?**

An end-to-end analysis of ~40K finished personal loans (Lending Club, 2007–2011): SQL portfolio analysis, vintage curves, an out-of-time default model, an approval-cutoff profit trade-off, and a collections priority list, with a Streamlit dashboard on top.

> Findings: see [`outputs/summary.md`](outputs/summary.md) and the charts in [`outputs/charts/`](outputs/charts/).

## What's inside

| Step | What it answers | Tools |
|---|---|---|
| 1. Data prep | Clean the raw data; keep only info known at application time | pandas |
| 2. Portfolio analysis | Which segments default most? Are newer vintages riskier? | SQL (DuckDB): CTEs, window functions |
| 3. Default model | Can we rank applicants by risk using borrower data alone? | scikit-learn: logistic regression, gradient boosting |
| 4. Approval cutoff | How many applicants to approve to maximise profit? | pandas |
| 5. Collections priority | Who should collections call first? | expected loss = PD × amount × LGD |
| 6. Dashboard | Explore everything, move the cutoff slider live | Streamlit |

## Key design choices (the parts worth explaining in an interview)

- **No data leakage.** Columns only known *after* a loan ends (total payments, recoveries, last payment date) are kept under a `post_` prefix and never used as model features. Using them gives a near-perfect but useless model.
- **Out-of-time validation.** Trained on 2007–2010 loans and tested on 2011 loans, the way a model is used in real life: built on the past, applied to new applicants.
- **Lending Club's grade is a benchmark, not a feature.** The model uses only borrower data (income, DTI, credit history, purpose…) and is compared against LC's own grade.
- **Decisions, not just accuracy.** The model feeds an approval cutoff (approval rate vs default rate vs profit) and a collections ranking by expected loss.
- **Simple, stated economics.** Profit = interest from repaid loans − LGD × amount for defaults, with LGD estimated from the data. It ignores funding and operating costs, which is fine for comparing cutoffs but not for exact profit.

## Run it

Needs Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get the data.** Download a Lending Club loan dataset from Kaggle (search "Lending Club loan data") and put the CSV in `data/raw/`. Either works:
- the 2007–2011 file (about 40K rows, often named `loan.csv` or `LoanStats3a.csv`), or
- the full `accepted_2007_to_2018Q4.csv` (large; the script keeps only loans issued 2007–2011, since all of those have finished).

```bash
python run_all.py          # prep -> SQL -> model -> charts -> summary
streamlit run app.py       # opens the dashboard in your browser
```

Each step can also be run on its own: `python src/prepare.py`, `src/run_sql.py`, `src/model.py`, `src/charts.py`, `src/summary.py`.

## Deploy the dashboard (free)

The app only reads `outputs/`, so the raw data never needs to go online.

1. Run `python run_all.py` locally, then commit the `outputs/` folder (the raw data is git-ignored).
2. Push to GitHub.
3. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo with main file `app.py`.

## Project structure

```
data/raw/            your downloaded CSV (not committed)
data/processed/      cleaned loans.csv (not committed)
sql/                 01-05 analysis queries, run by DuckDB
src/
  config.py          paths and settings (years, test year, default LGD)
  prepare.py         cleaning, features, leakage handling
  run_sql.py         runs every sql/*.sql and saves results
  model.py           models, deciles, risk drivers, cutoff curve, collections ranking
  cutoff.py          profit and cutoff logic (shared with the app)
  charts.py          PNG charts for the README and memo
  summary.py         key findings in plain English -> outputs/summary.md
outputs/tables/      every result as CSV/JSON
outputs/charts/      charts
app.py               Streamlit dashboard
run_all.py           runs the whole pipeline
```

## Limitations

- US data from 2007–2011; Indian salaried borrowers behave differently, so treat this as a method, not a direct benchmark.
- No bureau scores, bank statements or repayment-behaviour data, which a real lender would add.
- Months to default is approximated by the last payment date.

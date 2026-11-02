"""Step 2: run every query in sql/ with DuckDB and save each result as a CSV.

Run:  python src/run_sql.py

DuckDB reads the cleaned CSV directly, so there's no database to set up.
Each file in sql/ is one SELECT; its result is saved to outputs/tables/<file name>.csv
"""
import sys

import duckdb

from config import PROCESSED, SQL_DIR, TABLES


def main():
    if not PROCESSED.exists():
        sys.exit("Run src/prepare.py first.")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW loans AS SELECT * FROM read_csv_auto('{PROCESSED.as_posix()}', header=true)")

    for path in sorted(SQL_DIR.glob("*.sql")):
        df = con.execute(path.read_text()).df()
        out = TABLES / f"{path.stem}.csv"
        df.round(4).to_csv(out, index=False)
        print(f"\n== {path.name}  ({len(df)} rows) -> {out.name}")
        print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()

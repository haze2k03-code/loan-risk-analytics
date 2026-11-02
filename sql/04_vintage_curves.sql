-- Vintage curves: cumulative default rate by months on book, one line per issue year.
-- Months on book at default is approximated by the month of the last payment
-- (a charged-off loan with no payments at all counts as month 0).
WITH cohort AS (
  SELECT issue_year, COUNT(*) AS loans
  FROM loans
  GROUP BY issue_year
),
defaults_by_month AS (
  SELECT issue_year,
         COALESCE(post_months_on_book, 0) AS months_on_book,
         COUNT(*) AS defaults
  FROM loans
  WHERE is_default = 1
  GROUP BY 1, 2
)
SELECT
  d.issue_year,
  d.months_on_book,
  SUM(d.defaults) OVER (PARTITION BY d.issue_year ORDER BY d.months_on_book) * 1.0
    / c.loans AS cumulative_default_rate
FROM defaults_by_month d
JOIN cohort c ON c.issue_year = d.issue_year
ORDER BY d.issue_year, d.months_on_book;

-- Vintage view: default rate by the quarter the loans were issued.
-- Shows whether newer cohorts are getting riskier or safer.
SELECT
  issue_year,
  CAST(EXTRACT(QUARTER FROM CAST(issue_month AS DATE)) AS INTEGER) AS issue_quarter,
  COUNT(*)              AS loans,
  SUM(loan_amnt)        AS disbursed,
  AVG(is_default * 1.0) AS default_rate,
  AVG(int_rate)         AS avg_int_rate
FROM loans
GROUP BY 1, 2
ORDER BY 1, 2;

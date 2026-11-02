-- Headline numbers for the whole portfolio.
SELECT
  COUNT(*)                              AS loans,
  SUM(loan_amnt)                        AS total_disbursed,
  AVG(loan_amnt)                        AS avg_loan,
  AVG(int_rate)                         AS avg_int_rate,
  AVG(annual_inc)                       AS avg_income,
  SUM(is_default)                       AS defaults,
  AVG(is_default * 1.0)                 AS default_rate,
  SUM(CASE WHEN is_default = 1 THEN loan_amnt ELSE 0 END) * 1.0
    / SUM(loan_amnt)                    AS default_rate_by_amount
FROM loans;

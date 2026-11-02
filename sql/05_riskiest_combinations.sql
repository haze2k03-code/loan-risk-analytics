-- Which purpose x income combinations are riskiest? Ranked with a window function.
-- Only combinations with 200+ loans, so the ranking isn't driven by tiny groups.
WITH combo AS (
  SELECT purpose, income_band, COUNT(*) AS loans, AVG(is_default * 1.0) AS default_rate
  FROM loans
  GROUP BY purpose, income_band
  HAVING COUNT(*) >= 200
)
SELECT
  RANK() OVER (ORDER BY default_rate DESC) AS risk_rank,
  purpose,
  income_band,
  loans,
  default_rate
FROM combo
ORDER BY risk_rank
LIMIT 15;

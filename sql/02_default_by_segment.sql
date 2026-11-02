-- Default rate for every segment of every dimension, in one table.
-- lift = segment default rate / portfolio default rate (1.5 = 50% riskier than average).
-- share_of_defaults = what % of all defaults come from this segment.
WITH stacked AS (
  SELECT 'grade'               AS dimension, grade               AS segment, is_default FROM loans
  UNION ALL SELECT 'income_band',         income_band,                       is_default FROM loans
  UNION ALL SELECT 'loan_band',           loan_band,                         is_default FROM loans
  UNION ALL SELECT 'term',                CAST(term_months AS VARCHAR) || ' months', is_default FROM loans
  UNION ALL SELECT 'purpose',             purpose,                           is_default FROM loans
  UNION ALL SELECT 'home_ownership',      home_ownership,                    is_default FROM loans
  UNION ALL SELECT 'verification_status', verification_status,               is_default FROM loans
  UNION ALL SELECT 'emp_length',
    CASE WHEN emp_length_years IS NULL THEN 'unknown'
         WHEN emp_length_years < 2  THEN '0-1 yrs'
         WHEN emp_length_years < 5  THEN '2-4 yrs'
         WHEN emp_length_years < 10 THEN '5-9 yrs'
         ELSE '10+ yrs' END, is_default FROM loans
),
seg AS (
  SELECT dimension, segment, COUNT(*) AS loans, SUM(is_default) AS defaults,
         AVG(is_default * 1.0) AS default_rate
  FROM stacked
  WHERE segment IS NOT NULL
  GROUP BY dimension, segment
)
SELECT
  dimension,
  segment,
  loans,
  defaults,
  default_rate,
  default_rate / (SELECT AVG(is_default * 1.0) FROM loans)        AS lift,
  defaults * 1.0 / SUM(defaults) OVER (PARTITION BY dimension)    AS share_of_defaults
FROM seg
WHERE loans >= 100          -- ignore tiny segments: their rates are mostly noise
ORDER BY dimension, default_rate DESC;

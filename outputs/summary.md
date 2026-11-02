# Findings summary (auto-generated, check and rewrite in your own words)

## Portfolio
- 38,577 finished loans, $426M disbursed, overall default rate **14.6%**.
- Riskiest purpose: **small_business** at 27.1% (1.9x average). Safest: **major_purchase** at 10.3%.
- Income matters: borrowers earning <30k default at 18.4% vs 10.8% for 100k+.
- Riskiest combination (200+ loans): **small_business + income 50-75k** at 31.1%.

## Model
- Validation: out-of-time: train < 2011, test = 2011 (18,061 train / 20,516 test loans).
- Best model: **Logistic regression**, AUC 0.711 (Gini 0.421, KS 0.315).
- Benchmark, Lending Club's own grade: AUC 0.672. (Our model uses borrower data only, no grade or interest rate.)
- The safest 10% default at **3.7%**, the riskiest 10% at **37.8%**, 10.2x higher.
- Strongest drivers: purpose_small_business (raises risk); purpose_credit_card (lowers risk); purpose_wedding (lowers risk).

## Decisions
- Approving everyone in the test year: default rate 15.9%, profit -$3.88M (based on actual repayments, minus a 5% a year cost of funds).
- Best cutoff: approve the safest **70%**: default rate falls from 15.9% to **10.0%** and profit rises to **$2.64M** (+$6.52M).
- Collections: the top 10% of loans by expected loss carry **39%** of all expected loss, so calling them first covers the most risk per call.

## Recommendations (edit these)
- **Risk:** tighten approval for small_business loans and low-income applicants, or price them higher.
- **Marketing:** target low-risk segments such as major_purchase borrowers.
- **Collections:** prioritise outreach by expected loss (PD x loan amount), not by loan amount alone.
- **Caveats:** US data from 2007-2011; profit ignores funding and operating costs; a real model would also use bureau and bank-statement data.

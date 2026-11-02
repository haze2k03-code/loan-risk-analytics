"""Approval-cutoff economics, shared by the model step and the Streamlit app.

Idea: rank applicants from lowest to highest predicted default risk (PD) and
approve only the safest X%. Approving fewer people means fewer defaults but also
less interest earned. The cutoff with the highest total profit is the sweet spot.

Profit per loan is a simple, explainable approximation:
- repaid loan:    interest earned = installment * term - loan amount
- defaulted loan: loss = LGD * loan amount   (LGD = share of the loan we never get back)
It ignores funding costs, prepayments and operating costs, which a real
lender would add. Say so in the interview: it's the right shape, not exact money.
"""
import numpy as np
import pandas as pd


def monthly_installment(amount, annual_rate_pct, months):
    r = np.asarray(annual_rate_pct, dtype=float) / 100 / 12
    n = np.asarray(months, dtype=float)
    a = np.asarray(amount, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 0, a * r / (1 - (1 + r) ** -n), a / n)
    return pay


def add_profit(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    """Adds interest_if_repaid, loss_if_default and realised profit for each loan."""
    out = df.copy()
    inst = out["installment"] if "installment" in out else monthly_installment(
        out["loan_amnt"], out["int_rate"], out["term_months"])
    inst = pd.Series(inst, index=out.index).fillna(
        pd.Series(monthly_installment(out["loan_amnt"], out["int_rate"], out["term_months"]), index=out.index))
    out["interest_if_repaid"] = (inst * out["term_months"] - out["loan_amnt"]).clip(lower=0)
    out["loss_if_default"] = lgd * out["loan_amnt"]
    out["profit"] = np.where(out["is_default"] == 1, -out["loss_if_default"], out["interest_if_repaid"])
    return out


def cutoff_curve(scored: pd.DataFrame, lgd: float, steps=range(5, 101, 5)) -> pd.DataFrame:
    """For each approval rate, what happens if we approve only the lowest-PD loans?"""
    df = add_profit(scored, lgd).sort_values("pd").reset_index(drop=True)
    n = len(df)
    rows = []
    for pct in steps:
        k = max(1, int(round(n * pct / 100)))
        approved = df.iloc[:k]
        rows.append({
            "approval_rate": pct / 100,
            "pd_cutoff": approved["pd"].max(),
            "approved_loans": k,
            "default_rate": approved["is_default"].mean(),
            "disbursed": approved["loan_amnt"].sum(),
            "profit": approved["profit"].sum(),
            "profit_per_loan": approved["profit"].mean(),
        })
    return pd.DataFrame(rows)


def estimate_lgd(loans: pd.DataFrame, default_lgd: float) -> float:
    """Share of a defaulted loan never recovered, estimated from post-loan payments if available."""
    if "post_total_pymnt" not in loans:
        return default_lgd
    bad = loans[loans["is_default"] == 1].dropna(subset=["post_total_pymnt"])
    if len(bad) < 50:
        return default_lgd
    lost = (1 - bad["post_total_pymnt"] / bad["loan_amnt"]).clip(0, 1)
    return float(round(lost.mean(), 3))

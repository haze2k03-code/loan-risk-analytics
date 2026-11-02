"""Streamlit dashboard for the loan risk project.

Run:  streamlit run app.py      (after python run_all.py)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import CHARTS, TABLES  # noqa: E402
from cutoff import add_profit  # noqa: E402

st.set_page_config(page_title="Loan Risk Analytics", layout="wide")

if not (TABLES / "model_metrics.json").exists():
    st.error("No results yet. Run `python run_all.py` first, then refresh this page.")
    st.stop()


@st.cache_data
def load():
    t = {p.stem: pd.read_csv(p) for p in TABLES.glob("*.csv")}
    metrics = json.loads((TABLES / "model_metrics.json").read_text())
    return t, metrics


t, m = load()
summary = t["01_portfolio_summary"].iloc[0]
seg = t["02_default_by_segment"]

st.title("Personal Loan Risk & Collections Analytics")
st.caption(
    "Lending Club loans issued 2007–2011 (all finished). Who defaults, how much risk to take, "
    "and where collections should focus first."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Loans analysed", f"{int(summary['loans']):,}")
k2.metric("Amount disbursed", f"${summary['total_disbursed'] / 1e6:,.0f}M")
k3.metric("Default rate", f"{summary['default_rate']:.1%}")
k4.metric("Average interest rate", f"{summary['avg_int_rate']:.1f}%")

tab_seg, tab_vintage, tab_model, tab_cutoff, tab_coll = st.tabs(
    ["Who defaults", "Vintages", "Risk model", "Approval cutoff", "Collections priority"])

# ---------------- Who defaults ----------------
with tab_seg:
    dims = {"Loan purpose": "purpose", "Income band": "income_band", "Lending Club grade": "grade",
            "Loan amount": "loan_band", "Term": "term", "Employment length": "emp_length",
            "Home ownership": "home_ownership", "Income verification": "verification_status"}
    label = st.selectbox("Break down default rate by", list(dims))
    d = seg[seg["dimension"] == dims[label]].sort_values("default_rate", ascending=False)
    st.bar_chart(d.set_index("segment")["default_rate"], horizontal=True, y_label="", x_label="Default rate")
    st.dataframe(
        d[["segment", "loans", "default_rate", "lift", "share_of_defaults"]],
        hide_index=True,
        column_config={
            "default_rate": st.column_config.NumberColumn("Default rate", format="percent"),
            "lift": st.column_config.NumberColumn("Lift vs average", format="%.2fx"),
            "share_of_defaults": st.column_config.NumberColumn("Share of all defaults", format="percent"),
        })
    st.markdown("**Riskiest purpose × income combinations** (200+ loans each)")
    st.dataframe(t["05_riskiest_combinations"], hide_index=True,
                 column_config={"default_rate": st.column_config.NumberColumn("Default rate", format="percent")})

# ---------------- Vintages ----------------
with tab_vintage:
    st.markdown("Cumulative default rate by months since issue, one line per issue year. "
                "A line that rises faster means that year's loans went bad sooner.")
    vc = t["04_vintage_curves"].pivot_table(index="months_on_book", columns="issue_year",
                                             values="cumulative_default_rate").ffill()
    vc.columns = [str(c) for c in vc.columns]
    st.line_chart(vc, x_label="Months since loan was issued", y_label="Cumulative default rate")
    st.markdown("**Default rate by issue quarter**")
    vq = t["03_vintage_default"].copy()
    vq["quarter"] = vq["issue_year"].astype(str) + " Q" + vq["issue_quarter"].astype(str)
    st.bar_chart(vq.set_index("quarter")["default_rate"], y_label="Default rate")

# ---------------- Risk model ----------------
with tab_model:
    best = next(r for r in m["models"] if r["model"] == m["best_model"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", m["best_model"])
    c2.metric("AUC (test year)", f"{best['auc']:.3f}")
    c3.metric("Gini", f"{best['gini']:.3f}")
    st.caption(f"Validation: {m['split']}. Trained on {m['train_loans']:,} loans, tested on {m['test_loans']:,}. "
               "Only application-time data is used. Lending Club's grade and interest rate are excluded "
               "(they're the benchmark to beat), and post-loan columns like total payments are never used.")
    st.dataframe(pd.DataFrame(m["models"]), hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Does the model rank risk?** Actual default rate by risk decile (1 = safest 10%)")
        st.bar_chart(t["deciles"].set_index("risk_decile")["actual_default_rate"],
                     x_label="Risk decile", y_label="Actual default rate")
    with right:
        st.markdown("**Top risk drivers** (logistic regression)")
        st.dataframe(t["risk_drivers"].head(12)[["feature", "direction", "strength"]],
                     hide_index=True)

# ---------------- Approval cutoff ----------------
with tab_cutoff:
    st.markdown("Approve only the safest share of applicants (ranked by the model) and see what happens "
                "to defaults and profit. Profit = interest from repaid loans minus losses on defaults.")
    scored = t["scored_test_loans"]
    a, b = st.columns(2)
    approve_pct = a.slider("Share of applicants approved", 10, 100, int(m["best_cutoff"]["approval_rate"] * 100), 5,
                           format="%d%%")
    lgd = b.slider("Loss given default (share of a bad loan never recovered)", 0.2, 1.0, float(m["lgd"]), 0.05)

    df = add_profit(scored, lgd).sort_values("pd").reset_index(drop=True)
    k = max(1, int(round(len(df) * approve_pct / 100)))
    approved, everyone = df.iloc[:k], df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loans approved", f"{k:,}", f"{k - len(df):,} vs approve all")
    c2.metric("Default rate", f"{approved['is_default'].mean():.1%}",
              f"{(approved['is_default'].mean() - everyone['is_default'].mean()) * 100:+.1f} pts", delta_color="inverse")
    c3.metric("Profit", f"${approved['profit'].sum() / 1e6:,.2f}M",
              f"${(approved['profit'].sum() - everyone['profit'].sum()) / 1e6:+,.2f}M")
    c4.metric("Max PD approved", f"{approved['pd'].max():.1%}")

    curve = []
    for pct in range(5, 101, 5):
        n = max(1, int(round(len(df) * pct / 100)))
        part = df.iloc[:n]
        curve.append({"approval_rate": pct, "profit_m": part["profit"].sum() / 1e6,
                      "default_rate": part["is_default"].mean()})
    curve = pd.DataFrame(curve).set_index("approval_rate")
    best_pct = int(curve["profit_m"].idxmax())
    l, r = st.columns(2)
    with l:
        st.markdown(f"**Profit by approval rate** (peak at {best_pct}% with this LGD)")
        st.line_chart(curve["profit_m"], x_label="% of applicants approved", y_label="Profit ($M)")
    with r:
        st.markdown("**Default rate of approved loans**")
        st.line_chart(curve["default_rate"], x_label="% of applicants approved", y_label="Default rate")
    st.caption("Simplified economics: ignores funding cost, prepayments and operating cost. "
               "Good for comparing cutoffs, not for exact profit.")

# ---------------- Collections priority ----------------
with tab_coll:
    share = m["top_10pct_expected_loss_share"]
    st.markdown(f"Rank loans by **expected loss = PD × loan amount × LGD**. The top 10% of loans carry "
                f"**{share:.0%}** of all expected loss, so a collections team calling them first covers "
                "the most risk per call.")
    top_n = st.slider("Show top N loans", 10, 200, 50, 10)
    cols = [c for c in ["loan_id", "purpose", "income_band", "loan_amnt", "int_rate", "pd", "expected_loss",
                        "risk_decile", "is_default"] if c in scored]
    st.dataframe(
        scored.sort_values("expected_loss", ascending=False).head(top_n)[cols],
        hide_index=True,
        column_config={
            "pd": st.column_config.NumberColumn("PD", format="percent"),
            "expected_loss": st.column_config.NumberColumn("Expected loss ($)", format="%.0f"),
            "is_default": st.column_config.NumberColumn("Actually defaulted", help="1 = yes (known in hindsight)"),
        })

    by_seg = (scored.groupby("purpose")
              .agg(loans=("pd", "size"), avg_pd=("pd", "mean"), expected_loss=("expected_loss", "sum"))
              .sort_values("expected_loss", ascending=False).reset_index())
    by_seg["share_of_expected_loss"] = by_seg["expected_loss"] / by_seg["expected_loss"].sum()
    st.markdown("**Expected loss by purpose**")
    st.dataframe(by_seg, hide_index=True,
                 column_config={"avg_pd": st.column_config.NumberColumn("Avg PD", format="percent"),
                                "expected_loss": st.column_config.NumberColumn("Expected loss ($)", format="%.0f"),
                                "share_of_expected_loss": st.column_config.NumberColumn("Share", format="percent")})

with st.expander("Charts used in the findings memo"):
    for png in sorted(CHARTS.glob("*.png")):
        st.image(str(png), caption=png.stem.replace("_", " "))

"""Step 5: write outputs/summary.md, the key numbers in plain English.

Run:  python src/summary.py
Use it as the starting point for your 1-page findings memo. The sentences are
generated from your actual results; read them, check them, and rewrite in your own words.
"""
import json

import pandas as pd

from config import ROOT, TABLES


def money(x):
    return f"{'-' if x < 0 else ''}${abs(x) / 1e6:,.2f}M"


def top(seg, dimension, n=1, ascending=False):
    d = seg[seg["dimension"] == dimension].sort_values("default_rate", ascending=ascending)
    return d.head(n)


def main():
    summary = pd.read_csv(TABLES / "01_portfolio_summary.csv").iloc[0]
    seg = pd.read_csv(TABLES / "02_default_by_segment.csv")
    combos = pd.read_csv(TABLES / "05_riskiest_combinations.csv")
    drivers = pd.read_csv(TABLES / "risk_drivers.csv")
    deciles = pd.read_csv(TABLES / "deciles.csv")
    m = json.loads((TABLES / "model_metrics.json").read_text())

    worst_purpose = top(seg, "purpose").iloc[0]
    best_purpose = top(seg, "purpose", ascending=True).iloc[0]
    income = seg[seg["dimension"] == "income_band"].set_index("segment")["default_rate"]
    best_model = next(r for r in m["models"] if r["model"] == m["best_model"])
    bench = next((r for r in m["models"] if r["model"].startswith("Benchmark")), None)
    cut, allin = m["best_cutoff"], m["approve_all"]
    d1, d10 = deciles.iloc[0], deciles.iloc[-1]

    lines = [
        "# Findings summary (auto-generated, check and rewrite in your own words)",
        "",
        "## Portfolio",
        f"- {int(summary['loans']):,} finished loans, ${summary['total_disbursed'] / 1e6:,.0f}M disbursed, "
        f"overall default rate **{summary['default_rate']:.1%}**.",
        f"- Riskiest purpose: **{worst_purpose['segment']}** at {worst_purpose['default_rate']:.1%} "
        f"({worst_purpose['lift']:.1f}x average). Safest: **{best_purpose['segment']}** at {best_purpose['default_rate']:.1%}.",
    ]
    if "<30k" in income and "100k+" in income:
        lines.append(f"- Income matters: borrowers earning <30k default at {income['<30k']:.1%} "
                     f"vs {income['100k+']:.1%} for 100k+.")
    c = combos.iloc[0]
    lines.append(f"- Riskiest combination (200+ loans): **{c['purpose']} + income {c['income_band']}** "
                 f"at {c['default_rate']:.1%}.")

    lines += [
        "",
        "## Model",
        f"- Validation: {m['split']} ({m['train_loans']:,} train / {m['test_loans']:,} test loans).",
        f"- Best model: **{m['best_model']}**, AUC {best_model['auc']:.3f} (Gini {best_model['gini']:.3f}, "
        f"KS {best_model['ks']:.3f}).",
    ]
    if bench:
        lines.append(f"- Benchmark, Lending Club's own grade: AUC {bench['auc']:.3f}. "
                     "(Our model uses borrower data only, no grade or interest rate.)")
    lines.append(f"- The safest 10% default at **{d1['actual_default_rate']:.1%}**, "
                 f"the riskiest 10% at **{d10['actual_default_rate']:.1%}**, "
                 f"{d10['actual_default_rate'] / max(d1['actual_default_rate'], 1e-9):.1f}x higher.")
    top3 = drivers.head(3)
    lines.append("- Strongest drivers: " + "; ".join(
        f"{r.feature} ({r.direction})" for r in top3.itertuples()) + ".")

    lines += [
        "",
        "## Decisions",
        f"- Approving everyone in the test year: default rate {allin['default_rate']:.1%}, "
        f"profit {money(allin['profit'])} (based on {m.get('profit_basis', 'estimated')} repayments, "
        f"minus a {m.get('cost_of_funds', 0):.0%} a year cost of funds).",
        (f"- Best cutoff: approve the safest **{cut['approval_rate']:.0%}**: default rate falls from "
         f"{allin['default_rate']:.1%} to **{cut['default_rate']:.1%}** and profit rises to "
         f"**{money(cut['profit'])}** (+{money(cut['profit'] - allin['profit'])})."
         if cut["approval_rate"] < 1 else
         "- Best cutoff: approving everyone. Under these assumptions every risk band is still profitable; "
         "try a higher cost of funds in the dashboard."),
        f"- Collections: the top 10% of loans by expected loss carry **{m['top_10pct_expected_loss_share']:.0%}** "
        "of all expected loss, so calling them first covers the most risk per call.",
        "",
        "## Recommendations (edit these)",
        f"- **Risk:** tighten approval for {worst_purpose['segment']} loans and low-income applicants, "
        "or price them higher.",
        f"- **Marketing:** target low-risk segments such as {best_purpose['segment']} borrowers.",
        "- **Collections:** prioritise outreach by expected loss (PD x loan amount), not by loan amount alone.",
        "- **Caveats:** US data from 2007-2011; profit ignores funding and operating costs; "
        "a real model would also use bureau and bank-statement data.",
    ]
    out = ROOT / "outputs" / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

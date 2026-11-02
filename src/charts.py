"""Step 4: save the key charts as PNGs (for the README and the findings memo).

Run:  python src/charts.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CHARTS, TABLES

BLUE = "#2a78d6"
INK = "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def segment_chart(seg: pd.DataFrame, dimension: str, title: str, order=None):
    d = seg[seg["dimension"] == dimension].copy()
    if d.empty:
        return
    if order:
        d["segment"] = pd.Categorical(d["segment"], order, ordered=True)
        d = d.sort_values("segment")
    else:
        d = d.sort_values("default_rate")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.2))
    ax.barh(d["segment"].astype(str), d["default_rate"], color=BLUE, height=0.6)
    for y, (v, n) in enumerate(zip(d["default_rate"], d["loans"])):
        ax.text(v, y, f"  {v:.1%}  ({n:,} loans)", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, d["default_rate"].max() * 1.45)
    pct_axis(ax, "x")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(CHARTS / f"default_by_{dimension}.png", dpi=160)
    plt.close(fig)


def main():
    seg = pd.read_csv(TABLES / "02_default_by_segment.csv")
    segment_chart(seg, "grade", "Default rate by Lending Club grade", order=list("ABCDEFG"))
    segment_chart(seg, "income_band", "Default rate by annual income",
                  order=["<30k", "30-50k", "50-75k", "75-100k", "100k+"])
    segment_chart(seg, "purpose", "Default rate by loan purpose")
    segment_chart(seg, "term", "Default rate by loan term")

    # Vintage curves: one line per issue year (5 lines, single blue ramp light -> dark)
    vc = pd.read_csv(TABLES / "04_vintage_curves.csv")
    years = sorted(vc["issue_year"].unique())
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"][-len(years):]
    fig, ax = plt.subplots(figsize=(7, 4))
    for year, color in zip(years, ramp):
        d = vc[vc["issue_year"] == year]
        ax.plot(d["months_on_book"], d["cumulative_default_rate"], color=color, linewidth=2, label=str(year))
    ax.legend(title="Issue year", frameon=False, loc="upper left", fontsize=9, title_fontsize=9)
    pct_axis(ax)
    ax.set_xlabel("Months since loan was issued")
    ax.set_title("Vintage curves: cumulative default rate by issue year", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS / "vintage_curves.png", dpi=160)
    plt.close(fig)

    # Risk deciles: predicted vs actual default rate
    dec = pd.read_csv(TABLES / "deciles.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(dec["risk_decile"], dec["actual_default_rate"], color=BLUE, width=0.65, label="Actual default rate")
    ax.plot(dec["risk_decile"], dec["avg_pd"], color="#0b0b0b", marker="o", markersize=5, linewidth=1.5,
            label="Predicted (model)")
    ax.set_xticks(dec["risk_decile"])
    ax.set_xlabel("Risk decile (1 = safest 10% of applicants, 10 = riskiest 10%)")
    pct_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Model ranks risk: default rate rises across deciles", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS / "risk_deciles.png", dpi=160)
    plt.close(fig)

    # Cutoff curve: two separate panels (never two y-axes on one chart)
    cut = pd.read_csv(TABLES / "cutoff_curve.csv")
    best = cut.loc[cut["profit"].idxmax()]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    a1.plot(cut["approval_rate"], cut["profit"] / 1e6, color=BLUE, linewidth=2)
    a1.axvline(best["approval_rate"], color="#7a7975", linestyle="--", linewidth=1)
    a1.text(best["approval_rate"], best["profit"] / 1e6, f"  best: approve {best['approval_rate']:.0%}",
            va="bottom", fontsize=9, color=INK)
    a1.set_ylabel("Profit ($M, test year)")
    a1.set_title("Approval cutoff trade-off", loc="left", fontweight="bold")
    a2.plot(cut["approval_rate"], cut["default_rate"], color=BLUE, linewidth=2)
    a2.axvline(best["approval_rate"], color="#7a7975", linestyle="--", linewidth=1)
    a2.set_ylabel("Default rate of approved")
    a2.set_xlabel("Share of applicants approved (safest first)")
    pct_axis(a2)
    pct_axis(a2, "x")
    fig.tight_layout()
    fig.savefig(CHARTS / "cutoff_curve.png", dpi=160)
    plt.close(fig)

    print(f"Charts saved to {CHARTS}")


if __name__ == "__main__":
    main()

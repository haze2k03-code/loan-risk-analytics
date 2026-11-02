"""Step 3: predict which loans default, and turn that into decisions.

Run:  python src/model.py

1. Out-of-time split: train on loans issued before 2011, test on 2011 loans.
   (A random split would let the model "see the future"; this mimics real use.)
2. Two models on application-time data only:
   - logistic regression: simple, explainable, standard in lending
   - gradient boosting: captures non-linear patterns, usually a bit stronger
3. Benchmark: how well does Lending Club's own grade rank risk on its own?
4. Outputs for the business:
   - decile table: does risk rise smoothly from the safest 10% to the riskiest 10%?
   - top risk drivers (from the logistic regression)
   - approval cutoff curve: approval rate vs default rate vs profit
   - collections priority: which loans carry the most expected loss
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import COST_OF_FUNDS, DEFAULT_LGD, PROCESSED, TABLES, TEST_YEAR
from cutoff import cutoff_curve, estimate_lgd

# Borrower and loan details known at application time.
# Lending Club's own grade and interest rate are left OUT of the model on purpose:
# they are LC's risk opinion, and we want our own view from borrower data.
# They're used only as a benchmark to beat.
NUMERIC = ["loan_amnt", "term_months", "log_annual_inc", "dti", "emp_length_years",
           "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "log_revol_bal",
           "revol_util", "total_acc", "credit_history_years", "loan_to_income"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]
GRADE_ORDER = {g: i for i, g in enumerate("ABCDEFG", start=1)}


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(name, y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"model": name, "auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_stat(y, p), 4)}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    if "revol_bal" in df:
        df["log_revol_bal"] = np.log1p(df["revol_bal"].clip(lower=0))
    return df


def preprocessor(numeric, categorical, scale: bool):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
        ]), categorical),
    ])


def main():
    loans = build_features(pd.read_csv(PROCESSED, parse_dates=["issue_d", "issue_month"]))
    numeric = [c for c in NUMERIC if c in loans]
    categorical = [c for c in CATEGORICAL if c in loans]

    train = loans[loans["issue_year"] < TEST_YEAR]
    test = loans[loans["issue_year"] == TEST_YEAR]
    split = f"out-of-time: train < {TEST_YEAR}, test = {TEST_YEAR}"
    if len(train) < 1000 or len(test) < 500:   # fallback if the file has an unusual year mix
        test = loans.sample(frac=0.25, random_state=42)
        train = loans.drop(test.index)
        split = "random 75/25"
    print(f"Split: {split}  |  train {len(train):,}  test {len(test):,}")

    X_tr, y_tr = train[numeric + categorical], train["is_default"]
    X_te, y_te = test[numeric + categorical], test["is_default"]

    logit = Pipeline([("prep", preprocessor(numeric, categorical, scale=True)),
                      ("model", LogisticRegression(max_iter=2000, C=0.5))])
    gbm = Pipeline([("prep", preprocessor(numeric, categorical, scale=False)),
                    ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                             max_leaf_nodes=15, min_samples_leaf=100,
                                                             l2_regularization=1.0, random_state=42))])
    logit.fit(X_tr, y_tr)
    gbm.fit(X_tr, y_tr)
    p_logit = logit.predict_proba(X_te)[:, 1]
    p_gbm = gbm.predict_proba(X_te)[:, 1]

    results = [evaluate("Logistic regression", y_te, p_logit),
               evaluate("Gradient boosting", y_te, p_gbm)]
    if "grade" in test:
        grade_score = test["grade"].map(GRADE_ORDER)
        ok = grade_score.notna()
        results.append(evaluate("Benchmark: LC grade only", y_te[ok], grade_score[ok]))
    comparison = pd.DataFrame(results)
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)
    print("\n" + comparison.to_string(index=False))

    # Use the better of our two models for the business outputs
    best_name, best_p = max([("Logistic regression", p_logit), ("Gradient boosting", p_gbm)],
                            key=lambda t: roc_auc_score(y_te, t[1]))
    keep = [c for c in ["loan_id", "issue_month", "grade", "purpose", "income_band", "loan_amnt",
                        "int_rate", "term_months", "installment", "annual_inc", "is_default",
                        "post_total_pymnt"] if c in test]
    scored = test[keep].copy()
    scored["pd"] = best_p
    # Decile 1 = safest 10% of applicants, decile 10 = riskiest 10%
    scored["risk_decile"] = pd.qcut(scored["pd"].rank(method="first"), 10, labels=range(1, 11)).astype(int)

    deciles = (scored.groupby("risk_decile")
               .agg(loans=("pd", "size"), avg_pd=("pd", "mean"), actual_default_rate=("is_default", "mean"),
                    min_pd=("pd", "min"), max_pd=("pd", "max"))
               .reset_index())
    deciles.round(4).to_csv(TABLES / "deciles.csv", index=False)
    print("\nRisk deciles (1 = safest):\n" + deciles.round(3).to_string(index=False))

    # Risk drivers from the logistic regression (coefficients on standardised inputs)
    names = logit.named_steps["prep"].get_feature_names_out()
    coefs = logit.named_steps["model"].coef_[0]
    drivers = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coefficient": coefs})
    # For categories, compare each level with the average of its group (e.g. small_business vs
    # the average purpose), otherwise every level of a group can look like it "lowers risk".
    for col in categorical:
        in_group = drivers["feature"].str.startswith(col + "_")
        drivers.loc[in_group, "coefficient"] -= drivers.loc[in_group, "coefficient"].mean()
    drivers = (drivers
               .assign(direction=lambda d: np.where(d["coefficient"] > 0, "raises risk", "lowers risk"),
                       strength=lambda d: d["coefficient"].abs())
               .sort_values("strength", ascending=False))
    drivers.round(4).to_csv(TABLES / "risk_drivers.csv", index=False)
    print("\nTop risk drivers:\n" + drivers.head(10).round(3).to_string(index=False))

    # Approval cutoff economics
    lgd = estimate_lgd(loans, DEFAULT_LGD)
    curve = cutoff_curve(scored, lgd, COST_OF_FUNDS)
    curve.round(4).to_csv(TABLES / "cutoff_curve.csv", index=False)
    best = curve.loc[curve["profit"].idxmax()]
    print(f"\nLGD used for expected loss: {lgd:.0%}  |  cost of funds: {COST_OF_FUNDS:.0%} a year")
    print("Cutoff curve:\n" + curve.round(3).to_string(index=False))

    # Collections priority: expected loss = PD x exposure x LGD
    scored["expected_loss"] = scored["pd"] * scored["loan_amnt"] * lgd
    scored = scored.sort_values("expected_loss", ascending=False)
    top_decile_share = scored.head(len(scored) // 10)["expected_loss"].sum() / scored["expected_loss"].sum()
    scored.round({"pd": 4, "expected_loss": 2}).to_csv(TABLES / "scored_test_loans.csv", index=False)

    metrics = {
        "split": split,
        "train_loans": int(len(train)),
        "test_loans": int(len(test)),
        "test_default_rate": round(float(y_te.mean()), 4),
        "best_model": best_name,
        "models": results,
        "lgd": lgd,
        "cost_of_funds": COST_OF_FUNDS,
        "profit_basis": "actual" if "post_total_pymnt" in scored else "estimated",
        "best_cutoff": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
                        for k, v in best.items()},
        "approve_all": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
                        for k, v in curve.iloc[-1].items()},
        "top_10pct_expected_loss_share": round(float(top_decile_share), 4),
    }
    with open(TABLES / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTop 10% of loans by expected loss carry {top_decile_share:.0%} of all expected loss.")


if __name__ == "__main__":
    main()

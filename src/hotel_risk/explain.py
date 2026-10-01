"""Which booking-time features drive the model? Permutation importance (+ optional SHAP).

Permutation importance: shuffle one feature on the held-out test period and measure how much
ROC-AUC drops. A big drop means the model relies on that feature. Uses scikit-learn only.

Run: python -m hotel_risk.explain data/raw/hotel_bookings.csv [--shap]
Writes reports/feature_importance.json and reports/feature_importance.png
(and reports/shap_summary.png when --shap is given and `shap` is installed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.inspection import permutation_importance

from .calibrate import split3
from .data import TARGET, booking_time_features, clean, load_raw
from .models import make_gbm


def run(path: str, use_shap: bool = False) -> dict:
    df = clean(load_raw(path))
    train, _, test = split3(df)
    X_train, X_test = booking_time_features(train), booking_time_features(test)
    model = make_gbm().fit(X_train, train[TARGET])

    res = permutation_importance(model, X_test, test[TARGET], scoring="roc_auc",
                                 n_repeats=5, random_state=0, n_jobs=-1)
    ranking = sorted(zip(X_test.columns, res.importances_mean, res.importances_std),
                     key=lambda t: -t[1])
    out = {"metric": "drop in test ROC-AUC when the feature is shuffled (mean, std over 5 repeats)",
           "features": [{"feature": f, "auc_drop": round(float(m), 4), "std": round(float(s), 4)}
                        for f, m, s in ranking]}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path("reports").mkdir(exist_ok=True)
    top = ranking[:12][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f for f, _, _ in top], [m for _, m, _ in top], xerr=[s for _, _, s in top], color="#3b6ea5")
    ax.set(xlabel="Drop in test ROC-AUC when shuffled", title="What the model relies on (top 12)")
    fig.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=150)

    if use_shap:
        try:
            import shap
            sample = X_test.sample(2000, random_state=0)
            pre, clf = model.named_steps["pre"], model.named_steps["clf"]
            Xt = pre.transform(sample)
            names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xt)
            plt.figure()
            shap.summary_plot(sv, Xt, feature_names=names, show=False)
            plt.tight_layout()
            plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
            out["shap"] = "reports/shap_summary.png"
        except ImportError:
            out["shap"] = "skipped: `pip install shap` to enable"
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = run(args[0] if args else "data/raw/hotel_bookings.csv", use_shap="--shap" in sys.argv)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/feature_importance.json").write_text(json.dumps(res, indent=2))
    for r in res["features"][:10]:
        print(f"{r['feature']:32s} {r['auc_drop']:.4f} +/- {r['std']:.4f}")

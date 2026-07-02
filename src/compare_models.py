"""Aggregate per-architecture eval_*.json into a comparison table, pick the best model,
and write an analysis of which classes get confused with which."""
import json
from pathlib import Path

import pandas as pd

from models import ARCHITECTURES

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Weighted score balancing quality and CPU speed: quality matters most for a
# sorting system (misclassified waste defeats the purpose), speed is a
# secondary tie-breaker so the model is still practical to run live.
QUALITY_WEIGHT = 0.85
SPEED_WEIGHT = 0.15


def load_results():
    results = []
    for arch in ARCHITECTURES:
        path = REPORTS_DIR / f"eval_{arch}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    return results


def top_confusions(result, top_k=5):
    """Return the top_k (true_class, predicted_class, count) off-diagonal confusions."""
    cm = result["confusion_matrix"]
    classes = result["classes"]
    pairs = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i][j] > 0:
                pairs.append((true_cls, pred_cls, cm[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def main():
    results = load_results()
    if not results:
        print("No eval_*.json files found in reports/. Run evaluate.py first.")
        return

    rows = []
    for r in results:
        rows.append({
            "arch": r["arch"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "latency_ms_per_image": r["latency_ms_per_image"],
            "model_size_mb": r["model_size_mb"],
            "n_params_M": r["n_params"] / 1e6,
        })
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # Normalize for scoring: higher macro_f1 better, lower latency better.
    f1_norm = (df["macro_f1"] - df["macro_f1"].min()) / (df["macro_f1"].max() - df["macro_f1"].min() + 1e-9)
    lat_norm = 1 - (df["latency_ms_per_image"] - df["latency_ms_per_image"].min()) / (
        df["latency_ms_per_image"].max() - df["latency_ms_per_image"].min() + 1e-9
    )
    df["score"] = QUALITY_WEIGHT * f1_norm + SPEED_WEIGHT * lat_norm
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(df.to_string(index=False))

    best = df.iloc[0]
    best_result = next(r for r in results if r["arch"] == best["arch"])

    print(f"\n=== Лучшая модель: {best['arch']} ===")
    print(f"accuracy={best['accuracy']:.4f} macro_f1={best['macro_f1']:.4f} "
          f"latency={best['latency_ms_per_image']:.2f}ms size={best['model_size_mb']:.1f}MB")

    print("\nТоп-путаницы (истинный класс -> предсказанный, кол-во) по лучшей модели:")
    confusions = top_confusions(best_result)
    for true_cls, pred_cls, count in confusions:
        print(f"  {true_cls:10s} -> {pred_cls:10s} : {count}")

    with open(REPORTS_DIR / "best_model.json", "w") as f:
        json.dump({
            "arch": best["arch"],
            "metrics": best.to_dict(),
            "top_confusions": confusions,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()

"""Reproducible evaluation of the collision-risk classifier.

Trains on the ESA CDM dataset with a fixed seed and prints the exact metrics
to quote in the README/writeup. Single source of truth for reported numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import load_training_data, _FEATURES  # noqa: E402


def main() -> None:
    data = load_training_data()
    X = data.select(_FEATURES).to_numpy()
    y = data["tier"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced",
        n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "n_events_total": int(data.height),
        "n_actionable_total": int(data["tier"].sum()),
        "n_test": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_actionable": round(precision_score(y_test, y_pred), 4),
        "recall_actionable": round(recall_score(y_test, y_pred), 4),
        "f1_actionable": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n=== METRICS FOR README (copy these) ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== Full classification report ===")
    print(classification_report(y_test, y_pred,
                                target_names=["NON-ACTIONABLE", "ACTIONABLE"]))
    print("=== Confusion matrix (row=true, col=pred) ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
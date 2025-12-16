import json
import os
import traceback

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from apps.visualization.ml.constants import ALL_FEATURE_KEYS, CHART_TYPES
from apps.visualization.services.profiler import DatasetProfiler

BASE_DIR = settings.BASE_DIR
MODEL_PATH = BASE_DIR / "apps/visualization/ml/models/chart_recommend_model.pkl"
ANALYSIS_DIR = BASE_DIR / "model_analysis_external"
TEST_DATA_DIR = BASE_DIR / "apps/visualization/ml/data/test"


class Command(BaseCommand):
    help = "Evaluate trained chart recommender on external test datasets (no data leakage)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--testdir",
            type=str,
            default=str(TEST_DATA_DIR),
            help="Directory of external CSV files for evaluation (e.g., data/test_external/)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=str(MODEL_PATH),
            help="Path to trained model .pkl file",
        )

    def handle(self, *args, **options):
        test_dir = options["testdir"]
        model_path = options["model"]

        if not os.path.exists(test_dir):
            self.stderr.write(f"Test directory not found: {test_dir}")
            return

        if not os.path.exists(model_path):
            self.stderr.write(f"Model not found: {model_path}")
            return

        ANALYSIS_DIR.mkdir(exist_ok=True)

        # Load model
        self.stdout.write(f"Loading model from {model_path}")
        saved = joblib.load(model_path)
        clf = saved["model"]
        expected_features = saved.get("feature_names", ALL_FEATURE_KEYS)
        expected_classes = saved.get("classes", CHART_TYPES)

        if expected_classes != CHART_TYPES:
            self.stdout.write(
                self.style.WARNING(
                    "⚠️  CHART_TYPES mismatch between training and evaluation!"
                )
            )

        profiler = DatasetProfiler(sample_size=200)

        all_X = []
        all_Y_true = []

        csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
        self.stdout.write(f"Processing {len(csv_files)} external CSV files...")

        for fname in csv_files:
            fpath = os.path.join(test_dir, fname)
            try:
                df = pd.read_csv(fpath)
                if len(df) < 5:
                    self.stdout.write(f"Skipping {fname}: too few rows (<5)")
                    continue

                # Get visualizable columns & profile
                viz_cols = profiler.get_visualizable_columns(df)
                df_viz = df[viz_cols["visualizable_columns"]].copy()
                if df_viz.empty:
                    self.stdout.write(f"Skipping {fname}: no visualizable columns")
                    continue

                profile_result = profiler.profile(df_viz)
                relationships = profile_result.get("pairwise_relationships", [])

                file_samples = []

                # Process pairs
                for rel in relationships:
                    col1, col2 = rel["col1"], rel["col2"]
                    true_charts = rel.get("suggested_charts", [])
                    if not true_charts:
                        continue

                    features = profiler.extract_ml_features(
                        df_viz, col1, col2, precomputed_profile=profile_result
                    )
                    if not features:
                        continue

                    x = np.array(
                        [features.get(k, 0) for k in ALL_FEATURE_KEYS], dtype=float
                    )
                    y_true = self._encode_labels(true_charts)
                    file_samples.append((x, y_true))

                # Process single numeric columns
                for col_profile in profile_result["columns"]:
                    if col_profile["dtype"] == "numeric":
                        col_name = col_profile["name"]
                        features = profiler.extract_ml_features(
                            df_viz,
                            col_name,
                            col2=None,
                            precomputed_profile=profile_result,
                        )
                        if not features:
                            continue

                        x = np.array(
                            [features.get(k, 0) for k in ALL_FEATURE_KEYS], dtype=float
                        )
                        y_true = self._encode_labels(["histogram", "boxplot", "violin"])
                        file_samples.append((x, y_true))

                if file_samples:
                    for x, y in file_samples:
                        all_X.append(x)
                        all_Y_true.append(y)
                else:
                    self.stdout.write(f"Skipping {fname}: no valid samples generated")

            except Exception as e:
                self.stderr.write(f"Error processing {fname}: {e}")
                traceback.print_exc()
                continue

        if not all_Y_true:
            self.stderr.write("No valid samples found in test set.")
            return

        X_test = np.array(all_X)
        Y_true = np.array(all_Y_true)

        self.stdout.write(
            self.style.SUCCESS(
                f"\n✅ Processed {len(Y_true)} samples from {len(csv_files)} external datasets"
            )
        )

        # Predict
        Y_pred = clf.predict(X_test)
        Y_pred_proba = None

        try:
            probas = clf.predict_proba(X_test)
            if isinstance(probas, list):
                # MultiOutputClassifier returns list of arrays, each (n_samples, 2)
                Y_pred_proba = np.array(
                    [p[:, 1] for p in probas]
                ).T  # shape: (n_samples, n_labels)
        except Exception as e:
            self.stdout.write(f"Probability estimation failed: {e}")

        # Save raw arrays for debugging
        np.save(ANALYSIS_DIR / "X_test.npy", X_test)
        np.save(ANALYSIS_DIR / "Y_true.npy", Y_true)
        np.save(ANALYSIS_DIR / "Y_pred.npy", Y_pred)
        if Y_pred_proba is not None:
            np.save(ANALYSIS_DIR / "Y_proba.npy", Y_pred_proba)

        # Run evaluation
        self._run_evaluation(Y_true, Y_pred, Y_pred_proba, ANALYSIS_DIR)

    def _encode_labels(self, suggestions):
        vector = np.zeros(len(CHART_TYPES), dtype=int)
        for chart in suggestions:
            if chart in CHART_TYPES:
                idx = CHART_TYPES.index(chart)
                vector[idx] = 1
        return vector

    def _run_evaluation(self, Y_true, Y_pred, Y_proba, output_dir):
        # Overall metrics
        exact_acc = accuracy_score(Y_true, Y_pred)
        at_least_one = np.mean(
            [np.any((yt == 1) & (yp == 1)) for yt, yp in zip(Y_true, Y_pred)]
        )

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            Y_true, Y_pred, average=None, zero_division=0
        )

        roc_aucs = []
        for i in range(len(CHART_TYPES)):
            if Y_proba is not None and len(np.unique(Y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(Y_true[:, i], Y_proba[:, i])
                    roc_aucs.append(auc)
                except:
                    roc_aucs.append(np.nan)
            else:
                roc_aucs.append(np.nan)

        # Save metrics
        metrics = {
            "overall": {
                "exact_match_accuracy": float(exact_acc),
                "at_least_one_accuracy": float(at_least_one),
            },
            "per_class": {},
        }
        for i, chart in enumerate(CHART_TYPES):
            metrics["per_class"][chart] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
                "roc_auc": float(roc_aucs[i]) if not np.isnan(roc_aucs[i]) else None,
            }

        with open(output_dir / "external_evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        n_classes = len(CHART_TYPES)

        # === 1. Per-Class Metrics Bar Chart ===
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(n_classes)
        width = 0.25
        ax.bar(x - width, precision, width, label="Precision", color="#1f77b4")
        ax.bar(x, recall, width, label="Recall", color="#ff7f0e")
        ax.bar(x + width, f1, width, label="F1", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(CHART_TYPES, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("External Test Set: Per-Class Metrics")
        plt.tight_layout()
        plt.savefig(output_dir / "external_per_class_metrics.png", dpi=150)
        plt.close()

        # === 2. Confusion Matrices ===
        cols, rows = 4, (n_classes + 3) // 4
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten() if n_classes > 1 else [axes]
        for i, chart in enumerate(CHART_TYPES):
            cm = confusion_matrix(Y_true[:, i], Y_pred[:, i], labels=[0, 1])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
                ax=axes[i],
            )
            axes[i].set_title(f"{chart}\n(n={support[i]})")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "external_confusion_matrices.png", dpi=150)
        plt.close()

        # === 3. ROC Curves ===
        if Y_proba is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            for i in range(n_classes):
                if not np.isnan(roc_aucs[i]):
                    fpr, tpr, _ = roc_curve(Y_true[:, i], Y_proba[:, i])
                    ax.plot(fpr, tpr, label=f"{CHART_TYPES[i]} (AUC={roc_aucs[i]:.2f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves (External Test Set)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                output_dir / "external_roc_curves.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        # Final print
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("EXTERNAL EVALUATION COMPLETE")
        self.stdout.write("=" * 60)
        self.stdout.write(f"Exact Match Accuracy:   {exact_acc:.2%}")
        self.stdout.write(f"At-Least-One Accuracy:  {at_least_one:.2%}")
        self.stdout.write(f"Results saved to:       {output_dir}")

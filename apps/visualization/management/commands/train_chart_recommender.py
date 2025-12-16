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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

from apps.visualization.ml.constants import ALL_FEATURE_KEYS, CHART_TYPES, MODEL_PATH
from apps.visualization.services.profiler import DatasetProfiler

BASE_DIR = settings.BASE_DIR
MODEL_DIR = BASE_DIR / "apps/visualization/ml/models"
DATA_DIR = BASE_DIR / "apps/visualization/ml/data/raw"


# class Command(BaseCommand):
#     help = "Train chart recommender using DatasetProfiler features and heuristic labels."
#
#     def add_arguments(self, parser):
#         parser.add_argument('--csvdir', type=str, help="Directory of CSV files")
#         parser.add_argument('--incremental', action='store_true', help="Update existing model")
#         parser.add_argument('--model-strategy', type=str, default='unified',
#                             choices=['unified', 'separate', 'flag'],
#                             help="Model training strategy: 'unified' (single model, default), "
#                                  "'separate' (two models), 'flag' (single model with is_univariate feature)")
#
#     def handle(self, *args, **options):
#         csvdir = options['csvdir'] if options['csvdir'] else DATA_DIR
#         incremental = options['incremental']
#         model_strategy = options['model_strategy']
#
#         if not os.path.exists(MODEL_DIR):
#             os.makedirs(MODEL_DIR)
#
#         csv_files = [f for f in os.listdir(csvdir) if f.endswith('.csv')]
#         self.stdout.write(f"Found {len(csv_files)} CSV files. Extracting features per column-pair...")
#         self.stdout.write(f"Model strategy: {model_strategy}")
#
#         profiler = DatasetProfiler(sample_size=200)
#
#         samples_by_file = {}
#         univariate_samples_by_file = {}  # Only used for 'separate' strategy
#
#         for idx, fname in enumerate(csv_files):
#             fpath = os.path.join(csvdir, fname)
#             try:
#                 df = pd.read_csv(fpath)
#                 if len(df) < 5:
#                     continue
#
#                 visualizable_info = profiler.get_visualizable_columns(df)
#                 visualizable_columns = visualizable_info["visualizable_columns"]
#
#                 if not visualizable_columns:
#                     continue
#
#                 df_viz = df[visualizable_columns].copy()
#                 profile_result = profiler.profile(df_viz)
#                 relationships = profile_result.get('pairwise_relationships', [])
#
#                 file_samples = []
#                 file_univariate_samples = []
#
#                 # === 1. PAIRWISE RELATIONSHIPS (bivariate charts) ===
#                 for rel in relationships:
#                     col1, col2 = rel['col1'], rel['col2']
#                     suggested_charts = rel.get('suggested_charts', [])
#                     if not suggested_charts:
#                         continue
#
#                     features = profiler.extract_ml_features(
#                         df_viz, col1, col2, precomputed_profile=profile_result
#                     )
#                     if not features:
#                         continue
#
#                     label_vector = self._encode_labels(suggested_charts)
#                     feature_vector = np.array(
#                         [features.get(k, 0) for k in ALL_FEATURE_KEYS],
#                         dtype=float
#                     )
#                     file_samples.append((feature_vector, label_vector))
#
#                 # === 2. SINGLE-COLUMN SAMPLES (univariate charts) ===
#                 for col_profile in profile_result['columns']:
#                     col_name = col_profile['name']
#                     dtype = col_profile['dtype']
#                     univariate_charts = []
#
#                     if dtype == 'numeric':
#                         # Use profiler's method correctly (pass list of column profiles)
#                         if profiler._has_non_trivial_numeric([col_profile]):
#                             univariate_charts.append('histogram')
#                             # Optionally add boxplot for single numeric column
#                             univariate_charts.append('boxplot')
#
#                     elif dtype == 'categorical':
#                         # Only suggest pie if suitable
#                         cardinality = col_profile.get("cardinality", {})
#                         if cardinality.get("is_suitable_for_pie", False):
#                             univariate_charts.append('pie')
#                             univariate_charts.append('donut')  # Alternative to pie
#                         elif cardinality.get("is_suitable_for_bar", False):
#                             univariate_charts.append('bar')
#
#                     # Skip if no univariate charts recommended
#                     if not univariate_charts:
#                         continue
#
#                     features = profiler.extract_ml_features(
#                         df_viz, col_name, col2=None, precomputed_profile=profile_result
#                     )
#                     if not features:
#                         continue
#
#                     label_vector = self._encode_labels(univariate_charts)
#
#                     # === STRATEGY SELECTION ===
#                     if model_strategy == 'separate':
#                         # Use only non-pairwise features for separate univariate model
#                         univariate_keys = [k for k in ALL_FEATURE_KEYS
#                                            if not k.startswith('pair_') and not k.startswith('col2_')]
#                         feature_vector = np.array(
#                             [features.get(k, 0) for k in univariate_keys],
#                             dtype=float
#                         )
#                         file_univariate_samples.append((feature_vector, label_vector))
#
#                     elif model_strategy == 'flag':
#                         # Add 'is_univariate' flag to distinguish univariate samples
#                         feature_vector = np.array(
#                             [features.get(k, 0) for k in ALL_FEATURE_KEYS] + [1.0],  # 1.0 = is_univariate
#                             dtype=float
#                         )
#                         file_samples.append((feature_vector, label_vector))
#
#                     else:  # 'unified' strategy (default)
#                         # Use all features (pairwise will be zeros - model learns this pattern)
#                         feature_vector = np.array(
#                             [features.get(k, 0) for k in ALL_FEATURE_KEYS],
#                             dtype=float
#                         )
#                         file_samples.append((feature_vector, label_vector))
#
#                 if file_samples:
#                     samples_by_file[fname] = file_samples
#
#                 if file_univariate_samples and model_strategy == 'separate':
#                     univariate_samples_by_file[fname] = file_univariate_samples
#
#                 if (idx + 1) % 5 == 0:
#                     total_samples = sum(len(s) for s in samples_by_file.values())
#                     if model_strategy == 'separate':
#                         total_univariate = sum(len(s) for s in univariate_samples_by_file.values())
#                         self.stdout.write(
#                             f"Processed {idx + 1}/{len(csv_files)} files. "
#                             f"Bivariate: {total_samples}, Univariate: {total_univariate}"
#                         )
#                     else:
#                         self.stdout.write(
#                             f"Processed {idx + 1}/{len(csv_files)} files. Total samples: {total_samples}"
#                         )
#
#             except Exception as e:
#                 self.stderr.write(f"Skipping {fname}: {e}")
#                 traceback.print_exc()
#                 continue
#
#         if not samples_by_file:
#             self.stderr.write("No valid training samples found.")
#             return
#
#         # === DATASET-LEVEL SPLIT (with stratification by file size) ===
#         filenames = list(samples_by_file.keys())
#
#         # Stratify by file size to avoid imbalance
#         file_sizes = np.array([len(samples_by_file[f]) for f in filenames])
#         size_buckets = ['small' if size < 20 else 'medium' if size < 50 else 'large'
#                         for size in file_sizes]
#
#         try:
#             train_files, test_files = train_test_split(
#                 filenames, test_size=0.2, random_state=42, stratify=size_buckets
#             )
#         except ValueError:
#             # Fallback if stratification fails (too few samples per bucket)
#             self.stdout.write("⚠️  Stratification failed, using random split")
#             train_files, test_files = train_test_split(
#                 filenames, test_size=0.2, random_state=42
#             )
#
#         # Build bivariate dataset
#         X_train = []
#         Y_train = []
#         for fname in train_files:
#             for x, y in samples_by_file[fname]:
#                 X_train.append(x)
#                 Y_train.append(y)
#
#         X_test = []
#         Y_test = []
#         for fname in test_files:
#             for x, y in samples_by_file[fname]:
#                 X_test.append(x)
#                 Y_test.append(y)
#
#         if not X_train or not X_test:
#             self.stderr.write("Not enough data to split into train/test sets.")
#             return
#
#         X_train = np.array(X_train)
#         X_test = np.array(X_test)
#         Y_train = np.array(Y_train)
#         Y_test = np.array(Y_test)
#
#         # Store feature names based on strategy
#         if model_strategy == 'flag':
#             feature_names_used = ALL_FEATURE_KEYS + ['is_univariate']
#         else:
#             feature_names_used = ALL_FEATURE_KEYS
#
#         # === DEBUG: Check for label distribution issues ===
#         label_sums = Y_train.sum(axis=0)
#         total_train = Y_train.shape[0]
#
#         self.stdout.write("\n" + "=" * 60)
#         self.stdout.write("LABEL DISTRIBUTION ANALYSIS")
#         self.stdout.write("=" * 60)
#
#         for i, chart in enumerate(CHART_TYPES):
#             pos_count = label_sums[i]
#             pos_ratio = pos_count / total_train if total_train > 0 else 0
#
#             if pos_ratio > 0.95:
#                 self.stdout.write(
#                     self.style.WARNING(
#                         f"⚠️  '{chart}': {pos_ratio:.1%} positive ({pos_count}/{total_train}) - May cause overfitting"
#                     )
#                 )
#             elif pos_ratio < 0.05 and pos_count > 0:
#                 self.stdout.write(
#                     self.style.WARNING(
#                         f"⚠️  '{chart}': {pos_ratio:.1%} positive ({pos_count}/{total_train}) - Very rare, may underperform"
#                     )
#                 )
#             elif pos_count == 0:
#                 self.stdout.write(
#                     self.style.ERROR(
#                         f"❌ '{chart}': No positive examples in training set"
#                     )
#                 )
#             else:
#                 self.stdout.write(f"✓ '{chart}': {pos_ratio:.1%} positive ({pos_count}/{total_train})")
#
#         self.stdout.write("\n" + self.style.SUCCESS(f"Final Dataset:"))
#         self.stdout.write(f"  Strategy: {model_strategy}")
#         self.stdout.write(f"  Train files: {len(train_files)} | Test files: {len(test_files)}")
#         self.stdout.write(f"  Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
#         self.stdout.write(f"  Features per sample: {X_train.shape[1]}")
#         self.stdout.write(f"  Target Labels: {len(CHART_TYPES)} classes")
#
#         # === TRAIN MODEL ===
#         # Use ClassifierChain for better label dependency modeling
#         use_chain = False  # Set to True to enable chain model
#
#         if use_chain:
#             from sklearn.multioutput import ClassifierChain
#             self.stdout.write("Using ClassifierChain for label dependencies...")
#             clf = ClassifierChain(
#                 RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
#                 order='random',
#                 random_state=42
#             )
#         else:
#             clf = MultiOutputClassifier(
#                 RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
#             )
#
#         if incremental and os.path.exists(MODEL_PATH):
#             self.stdout.write(
#                 "⚠️  Incremental mode: Loading existing model not fully supported by sklearn RandomForest. "
#                 "Retraining with combined data would require custom implementation."
#             )
#
#         self.stdout.write("Training model...")
#         clf.fit(X_train, Y_train)
#
#         joblib.dump({
#             'model': clf,
#             'feature_names': feature_names_used,
#             'classes': CHART_TYPES,
#             'model_type': 'chain' if use_chain else 'multioutput',
#             'strategy': model_strategy
#         }, MODEL_PATH)
#
#         self.stdout.write(self.style.SUCCESS(f"Model saved to {MODEL_PATH}"))
#
#         # === TRAIN SEPARATE UNIVARIATE MODEL (only for 'separate' strategy) ===
#         if model_strategy == 'separate' and univariate_samples_by_file:
#             self.stdout.write("\n" + "=" * 60)
#             self.stdout.write("TRAINING SEPARATE UNIVARIATE MODEL")
#             self.stdout.write("=" * 60)
#
#             # Build univariate dataset
#             X_train_uni = []
#             Y_train_uni = []
#             for fname in train_files:
#                 if fname in univariate_samples_by_file:
#                     for x, y in univariate_samples_by_file[fname]:
#                         X_train_uni.append(x)
#                         Y_train_uni.append(y)
#
#             X_test_uni = []
#             Y_test_uni = []
#             for fname in test_files:
#                 if fname in univariate_samples_by_file:
#                     for x, y in univariate_samples_by_file[fname]:
#                         X_test_uni.append(x)
#                         Y_test_uni.append(y)
#
#             if X_train_uni and X_test_uni:
#                 X_train_uni = np.array(X_train_uni)
#                 X_test_uni = np.array(X_test_uni)
#                 Y_train_uni = np.array(Y_train_uni)
#                 Y_test_uni = np.array(Y_test_uni)
#
#                 univariate_keys = [k for k in ALL_FEATURE_KEYS if
#                                    not k.startswith('pair_') and not k.startswith('col2_')]
#
#                 clf_uni = MultiOutputClassifier(
#                     RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
#                 )
#
#                 self.stdout.write(f"Training univariate model with {X_train_uni.shape[0]} samples...")
#                 clf_uni.fit(X_train_uni, Y_train_uni)
#
#                 univariate_model_path = MODEL_PATH.replace('.pkl', '_univariate.pkl')
#                 joblib.dump({
#                     'model': clf_uni,
#                     'feature_names': univariate_keys,
#                     'classes': CHART_TYPES,
#                     'model_type': 'univariate'
#                 }, univariate_model_path)
#
#                 self.stdout.write(self.style.SUCCESS(f"Univariate model saved to {univariate_model_path}"))
#             else:
#                 self.stdout.write("⚠️  Not enough univariate samples for separate model")
#
#         # === EVALUATE ===
#         self._evaluate(clf, X_train, X_test, Y_train, Y_test)
#
#     def _has_non_trivial_numeric(self, col_profile):
#         """
#         Check if a numeric column has meaningful variation for histogram.
#
#         This is a wrapper method that works with the profiler's implementation.
#         """
#         if col_profile.get("dtype") != "numeric":
#             return False
#
#         card = col_profile.get("cardinality", {})
#         unique_ratio = card.get("unique_ratio", 1.0)
#
#         # Skip near-constant columns (but not low-cardinality like ratings 1-5)
#         if unique_ratio < 0.005:
#             return False
#
#         # Skip ID-like columns
#         if unique_ratio > 0.95:
#             return False
#
#         dist = col_profile.get("distribution", {})
#
#         # Skip if dominated by zeros
#         zero_ratio = dist.get("zero_ratio", 0)
#         if zero_ratio > 0.9:
#             return False
#
#         # Require some spread
#         if dist.get("coefficient_of_variation", 0) < 0.001:
#             return False
#
#         return True
#
#     def _encode_labels(self, suggestions):
#         """Encode chart suggestions as binary vector."""
#         vector = np.zeros(len(CHART_TYPES), dtype=int)
#         for chart in suggestions:
#             if chart in CHART_TYPES:
#                 idx = CHART_TYPES.index(chart)
#                 vector[idx] = 1
#         return vector
#
#     def _evaluate(self, clf, X_train, X_test, Y_train, Y_test):
#         """Comprehensive model evaluation with visualizations."""
#         # Ensure output directory exists
#         ANALYSIS_DIR = BASE_DIR / "model_analysis"
#         ANALYSIS_DIR.mkdir(exist_ok=True)
#
#         Y_pred = clf.predict(X_test)
#         Y_pred_proba = None
#
#         # Get probabilities for ROC
#         try:
#             probas = clf.predict_proba(X_test)
#             if isinstance(probas, list):
#                 # MultiOutputClassifier returns list of (n_samples, 2) arrays
#                 Y_pred_proba = np.array([p[:, 1] if p.shape[1] == 2 else p[:, -1] for p in probas]).T
#             else:
#                 Y_pred_proba = probas
#         except Exception as e:
#             self.stdout.write(f"⚠️  Probability estimation failed: {e}")
#             Y_pred_proba = None
#
#         n_classes = Y_test.shape[1]
#
#         # --- Overall Metrics ---
#         exact_acc = accuracy_score(Y_test, Y_pred)
#         at_least_one = np.mean([
#             np.any((y_true == 1) & (y_pred == 1))
#             for y_true, y_pred in zip(Y_test, Y_pred)
#         ])
#
#         # Hamming loss (fraction of wrong labels)
#         hamming = hamming_loss(Y_test, Y_pred)
#
#         # Jaccard similarity (intersection over union)
#         jaccard = jaccard_score(Y_test, Y_pred, average='samples', zero_division=0)
#
#         # --- Per-class metrics ---
#         precision, recall, f1, support = precision_recall_fscore_support(
#             Y_test, Y_pred, average=None, zero_division=0
#         )
#
#         roc_aucs = []
#         for i in range(n_classes):
#             if Y_pred_proba is not None and len(np.unique(Y_test[:, i])) > 1:
#                 try:
#                     auc = roc_auc_score(Y_test[:, i], Y_pred_proba[:, i])
#                     roc_aucs.append(auc)
#                 except:
#                     roc_aucs.append(np.nan)
#             else:
#                 roc_aucs.append(np.nan)
#
#         # Save metrics to JSON
#         metrics = {
#             "overall": {
#                 "exact_match_accuracy": float(exact_acc),
#                 "at_least_one_accuracy": float(at_least_one),
#                 "hamming_loss": float(hamming),
#                 "jaccard_similarity": float(jaccard)
#             },
#             "per_class": {}
#         }
#         for i, chart in enumerate(CHART_TYPES):
#             metrics["per_class"][chart] = {
#                 "precision": float(precision[i]),
#                 "recall": float(recall[i]),
#                 "f1_score": float(f1[i]),
#                 "support": int(support[i]),
#                 "roc_auc": float(roc_aucs[i]) if not np.isnan(roc_aucs[i]) else None
#             }
#
#         with open(ANALYSIS_DIR / "evaluation_metrics.json", "w") as f:
#             json.dump(metrics, f, indent=2)
#
#         # Plot per-class metrics
#         fig, ax = plt.subplots(figsize=(14, 8))
#         x = np.arange(n_classes)
#         width = 0.25
#
#         bars1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
#         bars2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
#         bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')
#
#         ax.set_xlabel('Chart Type')
#         ax.set_ylabel('Score')
#         ax.set_title('Per-Class Classification Metrics')
#         ax.set_xticks(x)
#         ax.set_xticklabels(CHART_TYPES, rotation=45, ha='right')
#         ax.set_ylim(0, 1)
#         ax.legend()
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#
#         plt.tight_layout()
#         plt.savefig(ANALYSIS_DIR / "per_class_metrics.png", dpi=150)
#         plt.close()
#
#         # Confusion matrices
#         cols = 4
#         rows = (n_classes + cols - 1) // cols
#         fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
#         axes = axes.flatten() if n_classes > 1 else [axes]
#
#         for i, chart in enumerate(CHART_TYPES):
#             cm = confusion_matrix(Y_test[:, i], Y_pred[:, i], labels=[0, 1])
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#                         xticklabels=['Pred 0', 'Pred 1'],
#                         yticklabels=['True 0', 'True 1'],
#                         ax=axes[i])
#             axes[i].set_title(f'{chart}\nSupport: {support[i]}')
#
#         for j in range(i + 1, len(axes)):
#             axes[j].axis('off')
#
#         plt.tight_layout()
#         plt.savefig(ANALYSIS_DIR / "confusion_matrices.png", dpi=150)
#         plt.close()
#
#         # ROC curves
#         if Y_pred_proba is not None:
#             fig, ax = plt.subplots(figsize=(10, 8))
#             for i in range(n_classes):
#                 if not np.isnan(roc_aucs[i]):
#                     fpr, tpr, _ = roc_curve(Y_test[:, i], Y_pred_proba[:, i])
#                     ax.plot(fpr, tpr, label=f'{CHART_TYPES[i]} (AUC = {roc_aucs[i]:.2f})')
#             ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#             ax.set_xlabel('False Positive Rate')
#             ax.set_ylabel('True Positive Rate')
#             ax.set_title('ROC Curves per Chart Type')
#             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.tight_layout()
#             plt.savefig(ANALYSIS_DIR / "roc_curves.png", dpi=150, bbox_inches='tight')
#             plt.close()
#
#         # === MODEL COMPARISON ===
#         self.stdout.write("\n" + "=" * 60)
#         self.stdout.write("MODEL COMPARISON (Micro F1 Score)")
#         self.stdout.write("=" * 60)
#
#         model_comparison = {}
#         models = {
#             "Random Forest": clf,
#             "Logistic Regression": MultiOutputClassifier(
#                 OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
#             ),
#             "SVM (Linear)": MultiOutputClassifier(
#                 OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
#             )
#         }
#
#         for name, model in models.items():
#             if name == "Random Forest":
#                 pred = Y_pred
#             else:
#                 try:
#                     self.stdout.write(f"Training {name}...")
#                     model.fit(X_train, Y_train)
#                     pred = model.predict(X_test)
#                 except Exception as e:
#                     self.stdout.write(f"⚠️  Failed to train {name}: {e}")
#                     model_comparison[name] = None
#                     continue
#
#             micro_f1 = f1_score(Y_test, pred, average='micro', zero_division=0)
#             macro_f1 = f1_score(Y_test, pred, average='macro', zero_division=0)
#             model_comparison[name] = {
#                 'micro_f1': micro_f1,
#                 'macro_f1': macro_f1
#             }
#             self.stdout.write(f"{name:20} → Micro F1: {micro_f1:.3f} | Macro F1: {macro_f1:.3f}")
#
#         with open(ANALYSIS_DIR / "model_comparison.json", "w") as f:
#             json.dump(model_comparison, f, indent=2)
#
#         # Plot model comparison
#         valid_models = {k: v for k, v in model_comparison.items() if v is not None}
#         if valid_models:
#             fig, ax = plt.subplots(figsize=(10, 6))
#             names = list(valid_models.keys())
#             micro_scores = [v['micro_f1'] for v in valid_models.values()]
#             macro_scores = [v['macro_f1'] for v in valid_models.values()]
#
#             x = np.arange(len(names))
#             width = 0.35
#
#             bars1 = ax.bar(x - width / 2, micro_scores, width, label='Micro F1', color='#2ca02c')
#             bars2 = ax.bar(x + width / 2, macro_scores, width, label='Macro F1', color='#1f77b4')
#
#             ax.set_ylabel('F1 Score')
#             ax.set_title('Model Comparison')
#             ax.set_xticks(x)
#             ax.set_xticklabels(names)
#             ax.set_ylim(0, 1)
#             ax.legend()
#             ax.grid(axis='y', linestyle='--', alpha=0.7)
#
#             for bars in [bars1, bars2]:
#                 for bar in bars:
#                     height = bar.get_height()
#                     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
#                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
#
#             plt.tight_layout()
#             plt.savefig(ANALYSIS_DIR / "model_comparison.png", dpi=150)
#             plt.close()
#
#         # === FINAL SUMMARY ===
#         self.stdout.write("\n" + "=" * 60)
#         self.stdout.write("EVALUATION COMPLETE!")
#         self.stdout.write("=" * 60)
#         self.stdout.write(f"Exact Match Accuracy:      {exact_acc:.2%}")
#         self.stdout.write(f"At-Least-One Accuracy:     {at_least_one:.2%}")
#         self.stdout.write(f"Hamming Loss:              {hamming:.4f}")
#         self.stdout.write(f"Jaccard Similarity:        {jaccard:.2%}")
#         self.stdout.write(f"\nPer-class metrics saved to: {ANALYSIS_DIR / 'evaluation_metrics.json'}")
#         self.stdout.write(f"Plots saved to:             {ANALYSIS_DIR}")


class Command(BaseCommand):
    help = "Train chart recommender using real data only (no synthetic data)."

    def add_arguments(self, parser):
        parser.add_argument("--csvdir", type=str, help="Directory of CSV files")
        parser.add_argument(
            "--incremental", action="store_true", help="Update existing model"
        )

    def handle(self, *args, **options):
        csvdir = options["csvdir"] if options["csvdir"] else DATA_DIR
        incremental = options["incremental"]

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        csv_files = [f for f in os.listdir(csvdir) if f.endswith(".csv")]
        self.stdout.write(f"Found {len(csv_files)} CSV files.")

        profiler = DatasetProfiler(sample_size=200)
        samples_by_file = {}

        # === 1. COLLECT DATA (REAL EXAMPLES ONLY) ===
        for idx, fname in enumerate(csv_files):
            fpath = os.path.join(csvdir, fname)
            try:
                df = pd.read_csv(fpath)
                if len(df) < 5:
                    continue

                viz_info = profiler.get_visualizable_columns(df)
                viz_cols = viz_info["visualizable_columns"]
                if not viz_cols:
                    continue

                df_viz = df[viz_cols].copy()
                profile_result = profiler.profile(df_viz)
                relationships = profile_result.get("pairwise_relationships", [])

                file_samples = []

                # Pairwise samples
                for rel in relationships:
                    col1, col2 = rel["col1"], rel["col2"]
                    suggested_charts = rel.get("suggested_charts", [])
                    if not suggested_charts:
                        continue

                    features = profiler.extract_ml_features(
                        df_viz, col1, col2, precomputed_profile=profile_result
                    )
                    if not features:
                        continue

                    label_vector = self._encode_labels(suggested_charts)
                    feature_vector = np.array(
                        [features.get(k, 0) for k in ALL_FEATURE_KEYS], dtype=float
                    )
                    file_samples.append((feature_vector, label_vector))

                # Single-column samples (only univariate charts)
                for col_profile in profile_result["columns"]:
                    col_name = col_profile["name"]
                    univariate_charts = []

                    if col_profile["dtype"] == "numeric":
                        if self._has_non_trivial_numeric(col_profile):
                            univariate_charts.append("histogram")
                    elif col_profile["dtype"] == "categorical":
                        if col_profile.get("cardinality", {}).get(
                            "is_suitable_for_pie", False
                        ):
                            univariate_charts.append("pie")

                    if not univariate_charts:
                        continue

                    features = profiler.extract_ml_features(
                        df_viz, col_name, col2=None, precomputed_profile=profile_result
                    )
                    if not features:
                        continue

                    label_vector = self._encode_labels(univariate_charts)
                    feature_vector = np.array(
                        [features.get(k, 0) for k in ALL_FEATURE_KEYS], dtype=float
                    )
                    file_samples.append((feature_vector, label_vector))

                if file_samples:
                    samples_by_file[fname] = file_samples

                if (idx + 1) % 10 == 0:
                    total = sum(len(s) for s in samples_by_file.values())
                    self.stdout.write(
                        f"Processed {idx + 1}/{len(csv_files)} files. Samples: {total}"
                    )

            except Exception as e:
                self.stderr.write(f"Skipping {fname}: {e}")
                traceback.print_exc()
                continue

        if not samples_by_file:
            self.stderr.write("No valid samples found.")
            return

        # === 2. DATASET-LEVEL SPLIT ===
        filenames = list(samples_by_file.keys())
        train_files, test_files = train_test_split(
            filenames, test_size=0.2, random_state=42
        )

        X_train, Y_train = self._build_dataset(
            [samples_by_file[f] for f in train_files]
        )
        X_test, Y_test = self._build_dataset([samples_by_file[f] for f in test_files])

        self.stdout.write(self.style.SUCCESS(f"\nDataset split:"))
        self.stdout.write(
            f"  Train files: {len(train_files)} | Test files: {len(test_files)}"
        )
        self.stdout.write(
            f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}"
        )

        # === 3. DEBUG: CHECK FOR ALWAYS-POSITIVE LABELS ===
        self._debug_label_distribution(Y_train, "Training")

        # === 4. TRAIN MODEL ===
        clf = ClassifierChain(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            order="random",
            random_state=42,
        )

        self.stdout.write("Training model...")
        clf.fit(X_train, Y_train)

        # === 5. SAVE MODEL ===
        joblib.dump(
            {"model": clf, "feature_names": ALL_FEATURE_KEYS, "classes": CHART_TYPES},
            MODEL_PATH,
        )
        self.stdout.write(self.style.SUCCESS(f"Model saved to {MODEL_PATH}"))

        # === 6. EVALUATE ===
        self._evaluate(clf, X_test, Y_test)

    def _build_dataset(self, file_samples_list):
        """Flatten list of file samples into X, Y arrays."""
        X, Y = [], []
        for file_samples in file_samples_list:
            for x, y in file_samples:
                X.append(x)
                Y.append(y)
        return np.array(X), np.array(Y)

    def _has_non_trivial_numeric(self, col_profile):
        """Check if numeric column is suitable for histogram."""
        if col_profile.get("dtype") != "numeric":
            return False
        card = col_profile.get("cardinality", {})
        unique_ratio = card.get("unique_ratio", 1.0)
        if unique_ratio < 0.01 or unique_ratio > 0.95:
            return False
        dist = col_profile.get("distribution", {})
        if dist.get("zero_ratio", 0) > 0.9:
            return False
        if dist.get("coefficient_of_variation", 0) < 0.01:
            return False
        return True

    def _encode_labels(self, suggestions):
        vector = np.zeros(len(CHART_TYPES), dtype=int)
        for chart in suggestions:
            if chart in CHART_TYPES:
                idx = CHART_TYPES.index(chart)
                vector[idx] = 1
        return vector

    def _debug_label_distribution(self, Y, name):
        """Print label support stats."""
        label_sums = Y.sum(axis=0)
        total = Y.shape[0]
        always_positive = []
        for i, chart in enumerate(CHART_TYPES):
            support = label_sums[i]
            if support == total and total > 0:
                always_positive.append(chart)
            elif support == 0:
                self.stdout.write(f"ℹ️  {chart} has 0 support in {name}")
        if always_positive:
            self.stdout.write(
                self.style.WARNING(f"⚠️  ALWAYS POSITIVE in {name}: {always_positive}")
            )

    def _evaluate(self, clf, X_test, Y_test):
        ANALYSIS_DIR = BASE_DIR / "model_analysis"
        ANALYSIS_DIR.mkdir(exist_ok=True)

        Y_pred = clf.predict(X_test)
        Y_pred_proba = None

        try:
            probas = clf.predict_proba(X_test)
            if isinstance(probas, list):
                Y_pred_proba = np.array([p[:, 1] for p in probas]).T
        except Exception as e:
            self.stdout.write(f"Probability failed: {e}")

        # Metrics
        exact_acc = accuracy_score(Y_test, Y_pred)
        at_least_one = np.mean(
            [np.any((yt == 1) & (yp == 1)) for yt, yp in zip(Y_test, Y_pred)]
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            Y_test, Y_pred, average=None, zero_division=0
        )

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
            }

        with open(ANALYSIS_DIR / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Plots
        self._plot_metrics(precision, recall, f1, support, ANALYSIS_DIR)
        self._plot_confusion_matrices(Y_test, Y_pred, support, ANALYSIS_DIR)

        # Final print
        self.stdout.write(f"\n✅ Evaluation complete!")
        self.stdout.write(f"Exact Match Accuracy: {exact_acc:.2%}")
        self.stdout.write(f"At-Least-One Accuracy: {at_least_one:.2%}")
        self.stdout.write(f"Results saved to: {ANALYSIS_DIR}")

    def _plot_metrics(self, precision, recall, f1, support, output_dir):
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(CHART_TYPES))
        width = 0.25
        ax.bar(x - width, precision, width, label="Precision", color="#1f77b4")
        ax.bar(x, recall, width, label="Recall", color="#ff7f0e")
        ax.bar(x + width, f1, width, label="F1", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(CHART_TYPES, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Per-Class Metrics")
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_metrics.png", dpi=150)
        plt.close()

    def _plot_confusion_matrices(self, Y_test, Y_pred, support, output_dir):
        n = len(CHART_TYPES)
        cols, rows = 4, (n + 3) // 4
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten() if n > 1 else [axes]
        for i, chart in enumerate(CHART_TYPES):
            cm = confusion_matrix(Y_test[:, i], Y_pred[:, i], labels=[0, 1])
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
        plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
        plt.close()

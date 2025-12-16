from collections import Counter

import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.classes_ = np.unique(y)  # Store unique classes
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
        return self

    def _gini(self, y):
        counter = Counter(y)
        impurity = 1.0
        total = len(y)
        for count in counter.values():
            prob = count / total
            impurity -= prob**2
        return impurity

    def _split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        current_gini = self._gini(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(
                    X, y, feature_idx, threshold
                )

                if (
                    len(y_left) < self.min_samples_leaf
                    or len(y_right) < self.min_samples_leaf
                ):
                    continue

                n_left, n_right = len(y_left), len(y_right)
                gini = (n_left / n_samples) * self._gini(y_left) + (
                    n_right / n_samples
                ) * self._gini(y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_classes == 1
            or n_samples < self.min_samples_split
        ):
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return {"leaf": True, "class": most_common}

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return {"leaf": True, "class": most_common}

        X_left, X_right, y_left, y_right = self._split(X, y, feature_idx, threshold)

        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return {
            "leaf": False,
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _predict_sample(self, x, tree):
        if tree["leaf"]:
            return tree["class"]

        if x[tree["feature_idx"]] <= tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = None  # Store classes seen during training
        self.n_classes_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        if self.max_features == "sqrt":
            max_feat = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_feat = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_feat = self.max_features
        else:
            max_feat = n_features

        max_feat = max(1, max_feat)

        self.trees = []

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            feature_indices = np.random.choice(
                n_features, min(max_feat, n_features), replace=False
            )
            X_sample_features = X_sample[:, feature_indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample_features, y_sample)

            self.trees.append((tree, feature_indices))

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        predictions = []
        for tree, feature_indices in self.trees:
            X_features = X[:, feature_indices]
            pred = tree.predict(X_features)
            predictions.append(pred)

        predictions = np.array(predictions).T
        final_predictions = []
        for sample_preds in predictions:
            counter = Counter(sample_preds)
            most_common = counter.most_common(1)[0][0]
            final_predictions.append(most_common)

        return np.array(final_predictions)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]

        probas = np.zeros((n_samples, self.n_classes_))

        all_predictions = []
        for tree, feature_indices in self.trees:
            X_features = X[:, feature_indices]
            pred = tree.predict(X_features)
            all_predictions.append(pred)

        all_predictions = np.array(
            all_predictions
        ).T  # Shape: (n_samples, n_estimators)

        for i in range(n_samples):
            sample_preds = all_predictions[i]
            counter = Counter(sample_preds)

            for cls, count in counter.items():
                cls_idx = np.where(self.classes_ == cls)[0][0]
                probas[i, cls_idx] = count / len(sample_preds)

        return probas


class MultiOutputClassifier:
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.values

        Y = np.array(Y)

        self.estimators_ = []
        for i in range(Y.shape[1]):
            estimator = self._clone_estimator()
            estimator.fit(X, Y[:, i])
            self.estimators_.append(estimator)

        return self

    def _clone_estimator(self):
        if isinstance(self.estimator, RandomForestClassifier):
            return RandomForestClassifier(
                n_estimators=self.estimator.n_estimators,
                max_depth=self.estimator.max_depth,
                min_samples_split=self.estimator.min_samples_split,
                min_samples_leaf=self.estimator.min_samples_leaf,
                max_features=self.estimator.max_features,
                random_state=self.estimator.random_state,
            )
        else:
            raise ValueError("Unsupported estimator type")

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)

        return np.column_stack(predictions)

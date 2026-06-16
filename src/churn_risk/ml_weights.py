from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import AccountBehavior, FeatureWeights
from .preprocessing import accounts_to_dataframe, NUMERIC_FEATURE_COLS


FEATURE_COLUMN_MAPPING = {
    "login_frequency_30d": "login_count_last_30d",
    "login_trend": None,
    "recency": "last_login_days_ago",
    "transaction_frequency": "total_transactions_last_30d",
    "transaction_amount": "transaction_amount_last_30d",
    "engagement_depth": "avg_session_minutes",
    "support_tickets": "support_tickets_last_30d",
    "payment_issues": "payment_failures_last_30d",
    "email_engagement": "email_open_rate",
}


def _extract_feature_matrix(accounts: List[AccountBehavior]) -> pd.DataFrame:
    df = accounts_to_dataframe(accounts)

    features = pd.DataFrame()
    features["login_frequency_30d"] = df["login_count_last_30d"]

    weekly_avg_30d = df["login_count_last_30d"] / 4.0
    weekly_avg_30d = weekly_avg_30d.replace(0, 1e-9)
    features["login_trend"] = (df["login_count_last_7d"] / weekly_avg_30d).clip(0, 1)

    features["recency"] = df["last_login_days_ago"]
    features["transaction_frequency"] = df["total_transactions_last_30d"]
    features["transaction_amount"] = df["transaction_amount_last_30d"]
    features["engagement_depth"] = df["avg_session_minutes"]
    features["support_tickets"] = df["support_tickets_last_30d"] + df["refund_count_last_90d"]
    features["payment_issues"] = df["payment_failures_last_30d"]
    features["email_engagement"] = df["email_open_rate"]

    features = features.fillna(features.median(numeric_only=True))
    features = features.fillna(0)

    return features


def _generate_pseudo_labels(features: pd.DataFrame) -> np.ndarray:
    recency_score = (features["recency"] - features["recency"].min()) / (
        features["recency"].max() - features["recency"].min() + 1e-9
    )

    login_inv = 1 - (features["login_frequency_30d"] - features["login_frequency_30d"].min()) / (
        features["login_frequency_30d"].max() - features["login_frequency_30d"].min() + 1e-9
    )

    support_score = (features["support_tickets"] - features["support_tickets"].min()) / (
        features["support_tickets"].max() - features["support_tickets"].min() + 1e-9
    )

    payment_score = (features["payment_issues"] - features["payment_issues"].min()) / (
        features["payment_issues"].max() - features["payment_issues"].min() + 1e-9
    )

    churn_prob = (
        recency_score * 0.35
        + login_inv * 0.25
        + support_score * 0.20
        + payment_score * 0.20
    )

    labels = (churn_prob > churn_prob.median()).astype(int).values
    return labels


@dataclass
class MLTrainingResult:
    weights: FeatureWeights
    model_type: str
    feature_importance: Dict[str, float]
    accuracy: float
    roc_auc: float
    training_samples: int
    model: Optional[object] = None
    scaler: Optional[object] = None


class WeightLearner:
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []

    def train(
        self,
        accounts: List[AccountBehavior],
        labels: Optional[np.ndarray] = None,
    ) -> MLTrainingResult:
        if len(accounts) < 10:
            raise ValueError("需要至少 10 个样本进行训练")

        features = _extract_feature_matrix(accounts)
        self._feature_names = list(features.columns)

        if labels is None:
            labels = _generate_pseudo_labels(features)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(features.values)

        if self.model_type == "logistic":
            self._model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            )
        elif self.model_type == "random_forest":
            self._model = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        else:
            self._model = LogisticRegression(max_iter=1000, random_state=42)

        self._model.fit(X_scaled, labels)

        predictions = self._model.predict(X_scaled)
        accuracy = float(accuracy_score(labels, predictions))

        try:
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X_scaled)[:, 1]
                roc_auc = float(roc_auc_score(labels, proba))
            else:
                roc_auc = 0.0
        except Exception:
            roc_auc = 0.0

        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importance = np.abs(self._model.coef_[0])
        else:
            importance = np.ones(len(self._feature_names))

        importance = importance / importance.sum()
        feature_importance = dict(zip(self._feature_names, importance))

        weights = FeatureWeights(
            login_frequency_30d=float(feature_importance.get("login_frequency_30d", 0.15)),
            login_trend=float(feature_importance.get("login_trend", 0.10)),
            recency=float(feature_importance.get("recency", 0.20)),
            transaction_frequency=float(feature_importance.get("transaction_frequency", 0.15)),
            transaction_amount=float(feature_importance.get("transaction_amount", 0.10)),
            engagement_depth=float(feature_importance.get("engagement_depth", 0.08)),
            support_tickets=float(feature_importance.get("support_tickets", 0.07)),
            payment_issues=float(feature_importance.get("payment_issues", 0.10)),
            email_engagement=float(feature_importance.get("email_engagement", 0.05)),
        )

        total_weight = (
            weights.login_frequency_30d
            + weights.login_trend
            + weights.recency
            + weights.transaction_frequency
            + weights.transaction_amount
            + weights.engagement_depth
            + weights.support_tickets
            + weights.payment_issues
            + weights.email_engagement
        )
        if total_weight > 0:
            for attr in weights.__dataclass_fields__:
                current = getattr(weights, attr)
                setattr(weights, attr, round(current / total_weight, 4))

        return MLTrainingResult(
            weights=weights,
            model_type=self.model_type,
            feature_importance={k: round(v, 4) for k, v in feature_importance.items()},
            accuracy=round(accuracy, 4),
            roc_auc=round(roc_auc, 4),
            training_samples=len(accounts),
            model=self._model,
            scaler=self._scaler,
        )

    def predict_churn_probability(
        self,
        accounts: List[AccountBehavior],
    ) -> List[float]:
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        features = _extract_feature_matrix(accounts)
        X_scaled = self._scaler.transform(features.values)

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self._model.predict(X_scaled)
            probabilities = predictions.astype(float)

        return [round(float(p), 4) for p in probabilities]


class OnlineWeightAdjuster:
    def __init__(
        self,
        initial_weights: Optional[FeatureWeights] = None,
        learning_rate: float = 0.05,
    ):
        self.weights = initial_weights or FeatureWeights()
        self.learning_rate = learning_rate
        self._history: List[FeatureWeights] = [self.weights]
        self._feedback_count = 0

    def adjust_weights(
        self,
        account_scores: List[Tuple[str, float, float]],
    ) -> FeatureWeights:
        if not account_scores:
            return self.weights

        total_loss = 0.0
        for account_id, predicted_score, actual_label in account_scores:
            total_loss += abs(predicted_score / 100.0 - actual_label)

        avg_loss = total_loss / len(account_scores)
        adjustment_factor = 1.0 + self.learning_rate * avg_loss

        weight_dict = {
            "login_frequency_30d": self.weights.login_frequency_30d,
            "login_trend": self.weights.login_trend,
            "recency": self.weights.recency,
            "transaction_frequency": self.weights.transaction_frequency,
            "transaction_amount": self.weights.transaction_amount,
            "engagement_depth": self.weights.engagement_depth,
            "support_tickets": self.weights.support_tickets,
            "payment_issues": self.weights.payment_issues,
            "email_engagement": self.weights.email_engagement,
        }

        adjustments = np.random.dirichlet(np.ones(len(weight_dict))) * 0.1 * avg_loss
        for i, key in enumerate(weight_dict):
            weight_dict[key] = max(0.01, weight_dict[key] * adjustment_factor + adjustments[i])

        total = sum(weight_dict.values())
        for key in weight_dict:
            weight_dict[key] = round(weight_dict[key] / total, 4)

        self.weights = FeatureWeights(**weight_dict)
        self._history.append(self.weights)
        self._feedback_count += len(account_scores)

        return self.weights

    @property
    def feedback_count(self) -> int:
        return self._feedback_count

    @property
    def weight_history(self) -> List[FeatureWeights]:
        return self._history.copy()

    def get_weight_trend_report(self) -> str:
        if len(self._history) < 2:
            return "权重历史不足，无法生成趋势报告"

        lines = []
        lines.append("=" * 60)
        lines.append("在线权重调整趋势报告")
        lines.append("=" * 60)
        lines.append(f"调整次数: {len(self._history) - 1}")
        lines.append(f"反馈样本数: {self._feedback_count}")
        lines.append("")

        initial = self._history[0]
        current = self._history[-1]
        lines.append(f"{'特征':<25s} {'初始权重':>10s} {'当前权重':>10s} {'变化':>10s}")
        lines.append("-" * 60)

        for attr in initial.__dataclass_fields__:
            init_val = getattr(initial, attr)
            curr_val = getattr(current, attr)
            change = curr_val - init_val
            change_str = f"{change:+.4f}"
            lines.append(f"{attr:<25s} {init_val:10.4f} {curr_val:10.4f} {change_str:>10s}")

        lines.append("=" * 60)
        return "\n".join(lines)


def train_weights_auto(
    accounts: List[AccountBehavior],
    model_type: str = "logistic",
    labels: Optional[np.ndarray] = None,
) -> MLTrainingResult:
    learner = WeightLearner(model_type=model_type)
    return learner.train(accounts, labels=labels)


@dataclass
class DriftDetectionResult:
    is_drift: bool
    drift_score: float
    threshold: float
    feature_drifts: Dict[str, float]
    window_acc: float
    baseline_acc: float
    drift_type: str
    message: str


class SGDWeightLearner:
    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: str = "optimal",
        alpha: float = 0.0001,
        random_state: int = 42,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.random_state = random_state
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []
        self._drift_detector = ConceptDriftDetector()
        self._partial_fit_count = 0
        self._classes = np.array([0, 1])

    def partial_fit(
        self,
        accounts: List[AccountBehavior],
        labels: Optional[np.ndarray] = None,
    ) -> Optional[DriftDetectionResult]:
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        if len(accounts) < 1:
            return None

        features = _extract_feature_matrix(accounts)
        self._feature_names = list(features.columns)

        if labels is None:
            labels = _generate_pseudo_labels(features)

        if self._scaler is None:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(features.values)
        else:
            X_scaled = self._scaler.partial_fit(features.values)
            X_scaled = self._scaler.transform(features.values)

        if self._model is None:
            self._model = SGDClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                alpha=self.alpha,
                random_state=self.random_state,
                warm_start=False,
            )
            self._model.partial_fit(X_scaled, labels, classes=self._classes)
            acc = float((self._model.predict(X_scaled) == labels).mean())
            self._drift_detector.set_baseline(features, acc)
        else:
            self._model.partial_fit(X_scaled, labels)

        for i, account in enumerate(accounts):
            pred = float(self._model.predict_proba(X_scaled[i:i+1])[0][1]) if hasattr(
                self._model, "predict_proba") else float(self._model.predict(X_scaled[i:i+1])[0])
            actual = float(labels[i])
            feat_dict = {col: float(features.iloc[i][col]) for col in features.columns}
            self._drift_detector.record(feat_dict, pred, actual)

        self._partial_fit_count += 1

        if self._partial_fit_count % 5 == 0:
            return self._drift_detector.detect()
        return None

    def get_weights(self) -> FeatureWeights:
        if self._model is None:
            return FeatureWeights()

        coef = np.abs(self._model.coef_[0]) if hasattr(self._model, "coef_") else np.ones(len(self._feature_names))
        coef = coef / coef.sum()
        importance = dict(zip(self._feature_names, coef))

        return FeatureWeights(
            login_frequency_30d=float(importance.get("login_frequency_30d", 0.15)),
            login_trend=float(importance.get("login_trend", 0.10)),
            recency=float(importance.get("recency", 0.20)),
            transaction_frequency=float(importance.get("transaction_frequency", 0.15)),
            transaction_amount=float(importance.get("transaction_amount", 0.10)),
            engagement_depth=float(importance.get("engagement_depth", 0.08)),
            support_tickets=float(importance.get("support_tickets", 0.07)),
            payment_issues=float(importance.get("payment_issues", 0.10)),
            email_engagement=float(importance.get("email_engagement", 0.05)),
        )

    def check_drift(self) -> DriftDetectionResult:
        return self._drift_detector.detect()

    @property
    def total_updates(self) -> int:
        return self._partial_fit_count


@dataclass
class ADWINConfig:
    delta: float = 0.002
    min_window_size: int = 10
    max_window_size: int = 1000
    min_drift_magnitude: float = 0.05
    consecutive_detections: int = 3
    bonferroni_correction: bool = True
    seasonal_period: Optional[int] = None
    suppress_false_positives: bool = True


@dataclass
class ADWINDetectionResult:
    is_drift: bool
    cut_point: Optional[int]
    drift_magnitude: float
    window_size: int
    left_mean: float
    right_mean: float
    confidence: float
    consecutive_count: int
    false_positive_suppressed: bool
    suppression_reason: str


class ADWIN:
    def __init__(self, config: Optional[ADWINConfig] = None):
        self.config = config or ADWINConfig()
        self._window: List[float] = []
        self._window_sums: List[float] = []
        self._consecutive_detections: int = 0
        self._last_cut_points: List[int] = []
        self._drift_magnitude_history: List[float] = []
        self._total: int = 0

    @property
    def window_size(self) -> int:
        return len(self._window)

    @property
    def total_samples(self) -> int:
        return self._total

    def add_element(self, value: float) -> None:
        self._window.append(float(value))
        self._total += 1
        if len(self._window) > self.config.max_window_size:
            self._window = self._window[-self.config.max_window_size:]

    def _hoeffding_bound(self, n1: int, n2: int, delta: float) -> float:
        n = n1 + n2
        if n <= 1:
            return float('inf')
        m = 1 / ((1 / n1) + (1 / n2))
        return float(np.sqrt((1 / (2 * m)) * np.log(4 * np.log(n) / delta)))

    def _compute_variance(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.var(values, ddof=1))

    def _detect_seasonal_pattern(self, values: List[float]) -> bool:
        if not self.config.seasonal_period or len(values) < self.config.seasonal_period * 2:
            return False
        period = self.config.seasonal_period
        n_periods = len(values) // period
        if n_periods < 2:
            return False

        period_means = []
        for i in range(n_periods):
            start = i * period
            end = start + period
            period_means.append(float(np.mean(values[start:end])))

        if len(period_means) < 2:
            return False

        variance_between = float(np.var(period_means, ddof=1))
        overall_variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0

        return variance_between > overall_variance * 0.3 and variance_between > self.config.min_drift_magnitude

    def detect(self) -> ADWINDetectionResult:
        if len(self._window) < self.config.min_window_size * 2:
            return ADWINDetectionResult(
                is_drift=False,
                cut_point=None,
                drift_magnitude=0.0,
                window_size=len(self._window),
                left_mean=0.0,
                right_mean=0.0,
                confidence=0.0,
                consecutive_count=self._consecutive_detections,
                false_positive_suppressed=False,
                suppression_reason="",
            )

        best_cut = None
        best_diff = 0.0
        best_left_mean = 0.0
        best_right_mean = 0.0
        best_confidence = 0.0
        n = len(self._window)
        prefix_sums = [0.0]
        for v in self._window:
            prefix_sums.append(prefix_sums[-1] + v)

        for cut in range(self.config.min_window_size, n - self.config.min_window_size + 1):
            n1 = cut
            n2 = n - cut
            mean1 = prefix_sums[cut] / n1
            mean2 = (prefix_sums[n] - prefix_sums[cut]) / n2
            diff = abs(mean1 - mean2)
            epsilon = self._hoeffding_bound(n1, n2, self.config.delta)
            if diff > epsilon and diff > self.config.min_drift_magnitude:
                if diff > best_diff:
                    best_diff = diff
                    best_cut = cut
                    best_left_mean = mean1
                    best_right_mean = mean2
                    best_confidence = float(diff / (epsilon + diff))

        is_drift = best_cut is not None
        suppressed = False
        suppress_reason = ""

        if is_drift and self.config.suppress_false_positives:
            if self._detect_seasonal_pattern(self._window):
                suppressed = True
                suppress_reason = "seasonal_pattern_detected"
                is_drift = False
            else:
                self._consecutive_detections += 1
                if self._consecutive_detections < self.config.consecutive_detections:
                    suppressed = True
                    suppress_reason = (
                        f"consecutive_check_{self._consecutive_detections}/"
                        f"{self.config.consecutive_detections}"
                    )
                    is_drift = False
                else:
                    self._consecutive_detections = 0
        else:
            if not is_drift and self._consecutive_detections > 0:
                self._consecutive_detections = max(0, self._consecutive_detections - 1)

        if best_cut is not None and is_drift:
            self._window = self._window[best_cut:]
            self._drift_magnitude_history.append(best_diff)
            self._last_cut_points.append(best_cut)

        return ADWINDetectionResult(
            is_drift=is_drift,
            cut_point=best_cut,
            drift_magnitude=best_diff,
            window_size=len(self._window),
            left_mean=best_left_mean,
            right_mean=best_right_mean,
            confidence=best_confidence,
            consecutive_count=self._consecutive_detections,
            false_positive_suppressed=suppressed,
            suppression_reason=suppress_reason,
        )

    def reset(self) -> None:
        self._window = []
        self._window_sums = []
        self._consecutive_detections = 0
        self._last_cut_points = []
        self._drift_magnitude_history = []
        self._total = 0


class ConceptDriftDetector:
    def __init__(
        self,
        window_size: int = 50,
        drift_threshold: float = 0.25,
        feature_threshold: float = 0.3,
        adwin_config: Optional[ADWINConfig] = None,
        n_features: int = 9,
        alpha: float = 0.05,
        min_drift_samples: int = 20,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.feature_threshold = feature_threshold
        self.adwin_config = adwin_config or ADWINConfig()
        self.n_features = n_features
        self.alpha = alpha
        self.min_drift_samples = min_drift_samples
        self._baseline_features: Optional[pd.DataFrame] = None
        self._baseline_acc: Optional[float] = None
        self._recent_predictions: List[Tuple[float, float]] = []
        self._recent_features: List[Dict[str, float]] = []
        self._adwin_monitors: Dict[str, ADWIN] = {}
        self._accuracy_adwin = ADWIN(self.adwin_config)
        self._drift_history: List[DriftDetectionResult] = []
        self._bonferroni_threshold = self.alpha / max(1, n_features) if self.adwin_config.bonferroni_correction else self.alpha

    def set_baseline(self, features: pd.DataFrame, accuracy: float) -> None:
        self._baseline_features = features.copy()
        self._baseline_acc = accuracy
        self._adwin_monitors = {}
        for col in features.columns:
            self._adwin_monitors[col] = ADWIN(self.adwin_config)

    def record(self, features: Dict[str, float], predicted: float, actual: float) -> None:
        self._recent_predictions.append((predicted, actual))
        self._recent_features.append(features)
        if len(self._recent_predictions) > self.window_size * 3:
            self._recent_predictions = self._recent_predictions[-self.window_size:]
            self._recent_features = self._recent_features[-self.window_size:]

        error = 1.0 if (predicted >= 0.5) != (actual >= 0.5) else 0.0
        self._accuracy_adwin.add_element(error)

        for feat_name, feat_value in features.items():
            if feat_name not in self._adwin_monitors:
                self._adwin_monitors[feat_name] = ADWIN(self.adwin_config)
            self._adwin_monitors[feat_name].add_element(feat_value)

    def _calculate_accuracy(self, pairs: List[Tuple[float, float]]) -> float:
        if not pairs:
            return 0.0
        correct = 0
        for pred, actual in pairs:
            pred_label = 1 if pred >= 0.5 else 0
            actual_label = 1 if actual >= 0.5 else 0
            if pred_label == actual_label:
                correct += 1
        return correct / len(pairs)

    def _calculate_feature_drift(self, recent_df: pd.DataFrame) -> Dict[str, float]:
        if self._baseline_features is None or recent_df.empty:
            return {}

        feature_drifts = {}
        for col in recent_df.columns:
            if col not in self._baseline_features.columns:
                continue
            baseline_mean = float(self._baseline_features[col].mean())
            recent_mean = float(recent_df[col].mean())
            baseline_std = float(self._baseline_features[col].std()) + 1e-9
            drift = abs(recent_mean - baseline_mean) / baseline_std
            feature_drifts[col] = round(drift, 4)

        return feature_drifts

    def detect(self) -> DriftDetectionResult:
        if self._baseline_acc is None:
            return DriftDetectionResult(
                is_drift=False,
                drift_score=0.0,
                threshold=self.drift_threshold,
                feature_drifts={},
                window_acc=0.0,
                baseline_acc=self._baseline_acc or 0.0,
                drift_type="none",
                message="尚未设置基线数据，无法检测漂移",
            )

        recent_window = self._recent_predictions[-self.window_size:] if len(
            self._recent_predictions) >= self.window_size // 2 else self._recent_predictions

        if len(recent_window) < self.min_drift_samples:
            return DriftDetectionResult(
                is_drift=False,
                drift_score=0.0,
                threshold=self.drift_threshold,
                feature_drifts={},
                window_acc=self._calculate_accuracy(recent_window),
                baseline_acc=self._baseline_acc,
                drift_type="insufficient_data",
                message=f"窗口数据不足 (当前 {len(recent_window)} 条，需至少 {self.min_drift_samples} 条)",
            )

        window_acc = self._calculate_accuracy(recent_window)
        acc_drop = max(0.0, self._baseline_acc - window_acc)

        feature_drifts = {}
        avg_feature_drift = 0.0
        adwin_feature_drifts: List[str] = []
        if self._recent_features and self._baseline_features is not None:
            recent_feat_window = self._recent_features[-self.window_size:]
            recent_df = pd.DataFrame(recent_feat_window)
            feature_drifts = self._calculate_feature_drift(recent_df)

            for feat_name, adwin in self._adwin_monitors.items():
                min_samples = (
                    self.adwin_config.min_window_size * 2
                    if hasattr(self.adwin_config, 'min_window_size')
                    else 20
                )
                if adwin.total_samples >= min_samples:
                    adwin_result = adwin.detect()
                    if adwin_result.is_drift:
                        adwin_feature_drifts.append(feat_name)

            if feature_drifts:
                avg_feature_drift = float(np.mean(list(feature_drifts.values())))

        high_drift_features = [
            k for k, v in feature_drifts.items() if v > self.feature_threshold
        ]

        if self.adwin_config.bonferroni_correction:
            bonferroni_feat_threshold = self.feature_threshold * max(1, np.sqrt(self.n_features))
            high_drift_features = [
                k for k, v in feature_drifts.items()
                if v > bonferroni_feat_threshold
            ]

        drift_score = round(acc_drop * 0.6 + avg_feature_drift * 0.4, 4)

        adwin_acc_result = self._accuracy_adwin.detect()
        adwin_acc_drift = adwin_acc_result.is_drift and adwin_acc_result.drift_magnitude > self.adwin_config.min_drift_magnitude

        is_drift = (
            (drift_score > self.drift_threshold)
            or (len(high_drift_features) >= 3)
            or (adwin_acc_drift)
            or (len(adwin_feature_drifts) >= 3)
        )

        drift_messages = []
        drift_type_parts = []

        if adwin_acc_result.false_positive_suppressed:
            drift_messages.append(f"ADWIN 准确率假阳性抑制: {adwin_acc_result.suppression_reason}")

        if is_drift:
            if acc_drop > self.drift_threshold * 0.8 or adwin_acc_drift:
                drift_type_parts.append("accuracy_drop")
                drift_messages.append(f"准确率下降: 基线 {self._baseline_acc:.2f} -> 窗口 {window_acc:.2f}")
            if len(high_drift_features) >= 3:
                drift_type_parts.append("feature_distribution")
                drift_messages.append(f"特征漂移: {', '.join(high_drift_features[:5])}")
            if len(adwin_feature_drifts) >= 3:
                drift_type_parts.append("adwin_feature")
                drift_messages.append(f"ADWIN 特征漂移: {', '.join(adwin_feature_drifts[:5])}")

            drift_type = "_".join(drift_type_parts) if drift_type_parts else "mixed"
            drift_msg = "; ".join(drift_messages) if drift_messages else "检测到概念漂移"
        else:
            drift_type = "stable"
            drift_msg = "未检测到显著概念漂移"
            if adwin_acc_result.false_positive_suppressed:
                drift_msg += f" (抑制: {adwin_acc_result.suppression_reason})"

        result = DriftDetectionResult(
            is_drift=is_drift,
            drift_score=drift_score,
            threshold=self.drift_threshold,
            feature_drifts=feature_drifts,
            window_acc=round(window_acc, 4),
            baseline_acc=round(self._baseline_acc, 4),
            drift_type=drift_type,
            message=drift_msg,
        )

        self._drift_history.append(result)
        if len(self._drift_history) > 100:
            self._drift_history = self._drift_history[-100:]

        return result

    def get_adwin_status(self) -> Dict[str, Any]:
        status = {
            "accuracy_window": {
                "size": self._accuracy_adwin.window_size,
                "total": self._accuracy_adwin.total_samples,
                "consecutive_detections": self._accuracy_adwin._consecutive_detections,
            },
            "feature_monitors": {},
        }
        for feat_name, adwin in self._adwin_monitors.items():
            status["feature_monitors"][feat_name] = {
                "window_size": adwin.window_size,
                "total_samples": adwin.total_samples,
                "consecutive_detections": adwin._consecutive_detections,
            }
        return status


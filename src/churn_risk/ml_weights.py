from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

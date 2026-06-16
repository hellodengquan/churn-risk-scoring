from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import AccountBehavior, RiskScore
from .preprocessing import accounts_to_dataframe


FEATURE_LABELS = {
    "login_frequency_30d": "30天登录频次",
    "login_trend": "登录趋势变化",
    "recency": "最近登录间隔",
    "transaction_frequency": "交易频次",
    "transaction_amount": "交易金额",
    "engagement_depth": "参与深度(平均时长)",
    "support_tickets": "客服工单+退款数",
    "payment_issues": "支付失败次数",
    "email_engagement": "邮件打开率",
}


def _build_feature_frame(accounts: List[AccountBehavior]) -> pd.DataFrame:
    df = accounts_to_dataframe(accounts)
    features = pd.DataFrame()
    features["login_frequency_30d"] = df["login_count_last_30d"]
    weekly_avg = (df["login_count_last_30d"] / 4.0).replace(0, 1e-9)
    features["login_trend"] = (df["login_count_last_7d"] / weekly_avg).clip(0, 1)
    features["recency"] = df["last_login_days_ago"]
    features["transaction_frequency"] = df["total_transactions_last_30d"]
    features["transaction_amount"] = df["transaction_amount_last_30d"]
    features["engagement_depth"] = df["avg_session_minutes"]
    features["support_tickets"] = df["support_tickets_last_30d"] + df["refund_count_last_90d"]
    features["payment_issues"] = df["payment_failures_last_30d"]
    features["email_engagement"] = df["email_open_rate"]
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    return features


@dataclass
class SHAPExplanation:
    feature_name: str
    feature_label: str
    feature_value: float
    shap_value: float
    impact_direction: str


@dataclass
class SHAPAccountResult:
    account_id: str
    base_value: float
    predicted_score: float
    explanations: List[SHAPExplanation]
    top_positive_drivers: List[SHAPExplanation]
    top_negative_drivers: List[SHAPExplanation]


@dataclass
class SHAPGlobalResult:
    mean_abs_shap: Dict[str, float]
    feature_ranking: List[str]
    sample_count: int


class SHAPAnalyzer:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self._explainer = None
        self._feature_names: List[str] = []

    def _ensure_explainer(self, X: pd.DataFrame):
        if self._explainer is not None:
            return

        try:
            import shap
            self._feature_names = list(X.columns)

            if self.model is not None and self.scaler is not None:
                X_scaled = self.scaler.transform(X.values)
                X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                background = shap.sample(X_scaled_df, min(100, len(X_scaled_df)), random_state=42)
                self._explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler

                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X.values)

                pseudo_labels = self._generate_labels(X)
                self.model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
                self.model.fit(X_scaled, pseudo_labels)

                X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                background = shap.sample(X_scaled_df, min(50, len(X_scaled_df)), random_state=42)

                def predict_fn(x):
                    return self.model.predict_proba(x)

                self._explainer = shap.KernelExplainer(predict_fn, background)
                self._feature_names = list(X.columns)
        except Exception:
            self._explainer = None

    def _generate_labels(self, X: pd.DataFrame) -> np.ndarray:
        recency_norm = (X["recency"] - X["recency"].min()) / (X["recency"].max() - X["recency"].min() + 1e-9)
        login_inv = 1 - (X["login_frequency_30d"] - X["login_frequency_30d"].min()) / (
            X["login_frequency_30d"].max() - X["login_frequency_30d"].min() + 1e-9
        )
        churn_prob = recency_norm * 0.5 + login_inv * 0.3 + (X["payment_issues"] > 0).astype(float) * 0.2
        return (churn_prob > churn_prob.median()).astype(int).values

    def analyze_accounts(
        self,
        accounts: List[AccountBehavior],
        target_account_ids: Optional[List[str]] = None,
    ) -> List[SHAPAccountResult]:
        if not accounts:
            return []

        X = _build_feature_frame(accounts)
        self._ensure_explainer(X)

        if self.scaler is None:
            return self._fallback_analysis(accounts, X, target_ids=target_account_ids)

        X_scaled = self.scaler.transform(X.values)

        try:
            import shap
            shap_values = self._explainer.shap_values(
                X_scaled, nsamples=min(50, max(10, len(X_scaled) // 2))
            )

            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values

            base_value = float(self._explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[-1]) if len(base_value) > 1 else float(base_value[0])

            results = []
            for i, account in enumerate(accounts):
                if target_account_ids and account.account_id not in target_account_ids:
                    continue

                row_explanations = []
                for j, feat in enumerate(self._feature_names):
                    val = float(X.iloc[i][feat])
                    sv = float(shap_values_class1[i][j]) if len(shap_values_class1.shape) > 1 else float(shap_values_class1[i])
                    direction = "增加流失风险" if sv > 0 else "降低流失风险"

                    exp = SHAPExplanation(
                        feature_name=feat,
                        feature_label=FEATURE_LABELS.get(feat, feat),
                        feature_value=val,
                        shap_value=round(sv, 6),
                        impact_direction=direction,
                    )
                    row_explanations.append(exp)

                row_explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

                top_pos = [e for e in row_explanations if e.shap_value > 0][:3]
                top_neg = [e for e in row_explanations if e.shap_value < 0][:3]

                pred = float(shap_values_class1[i].sum()) if len(shap_values_class1.shape) > 1 else float(shap_values_class1[i])

                result = SHAPAccountResult(
                    account_id=account.account_id,
                    base_value=round(base_value, 6),
                    predicted_score=round(base_value + pred, 4),
                    explanations=row_explanations,
                    top_positive_drivers=top_pos,
                    top_negative_drivers=top_neg,
                )
                results.append(result)

            return results

        except Exception:
            return self._fallback_analysis(accounts, X, target_account_ids)

    def _fallback_analysis(
        self,
        accounts: List[AccountBehavior],
        X: pd.DataFrame,
        target_ids: Optional[List[str]] = None,
    ) -> List[SHAPAccountResult]:
        results = []
        feature_names = list(X.columns)

        X_norm = (X - X.min()) / (X.max() - X.min() + 1e-9)
        X_norm = X_norm.fillna(0)

        risk_weights = {
            "recency": 0.25,
            "payment_issues": 0.18,
            "support_tickets": 0.15,
            "login_frequency_30d": -0.18,
            "transaction_frequency": -0.12,
            "transaction_amount": -0.05,
            "engagement_depth": -0.04,
            "email_engagement": -0.02,
            "login_trend": -0.01,
        }

        filtered_indices = []
        filtered_accounts = []
        for i, account in enumerate(accounts):
            if target_ids is None or account.account_id in target_ids:
                filtered_indices.append(i)
                filtered_accounts.append(account)

        for idx, account in zip(filtered_indices, filtered_accounts):
            i = idx

            explanations = []
            total_contribution = 0.0

            for feat in feature_names:
                weight = risk_weights.get(feat, 0.0)
                norm_val = float(X_norm.iloc[i][feat])
                raw_val = float(X.iloc[i][feat])
                contribution = weight * norm_val
                total_contribution += contribution

                direction = "增加流失风险" if contribution > 0 else "降低流失风险"
                explanations.append(SHAPExplanation(
                    feature_name=feat,
                    feature_label=FEATURE_LABELS.get(feat, feat),
                    feature_value=raw_val,
                    shap_value=round(contribution, 6),
                    impact_direction=direction,
                ))

            explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

            top_pos = [e for e in explanations if e.shap_value > 0][:3]
            top_neg = [e for e in explanations if e.shap_value < 0][:3]

            results.append(SHAPAccountResult(
                account_id=account.account_id,
                base_value=0.5,
                predicted_score=round(max(0.0, min(1.0, 0.5 + total_contribution)), 4),
                explanations=explanations,
                top_positive_drivers=top_pos,
                top_negative_drivers=top_neg,
            ))

        return results

    def global_analysis(
        self,
        accounts: List[AccountBehavior],
    ) -> SHAPGlobalResult:
        X = _build_feature_frame(accounts)
        feature_names = list(X.columns)

        account_results = self.analyze_accounts(accounts)

        mean_abs_shap: Dict[str, float] = {}
        for feat in feature_names:
            abs_vals = []
            for r in account_results:
                for exp in r.explanations:
                    if exp.feature_name == feat:
                        abs_vals.append(abs(exp.shap_value))
            mean_abs_shap[feat] = round(float(np.mean(abs_vals)) if abs_vals else 0.0, 6)

        ranking = sorted(mean_abs_shap.keys(), key=lambda k: mean_abs_shap[k], reverse=True)

        return SHAPGlobalResult(
            mean_abs_shap=mean_abs_shap,
            feature_ranking=ranking,
            sample_count=len(accounts),
        )


def format_shap_account_report(result: SHAPAccountResult) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"SHAP 可解释性分析报告 - 账号: {result.account_id}")
    lines.append("=" * 70)
    lines.append(f"基准概率: {result.base_value:.4f}")
    lines.append(f"预测流失概率: {result.predicted_score:.4f}")
    lines.append("")

    lines.append("TOP 增加流失风险特征:")
    for i, exp in enumerate(result.top_positive_drivers, 1):
        lines.append(
            f"  {i}. {exp.feature_label}: 影响值={exp.shap_value:+.6f}, "
            f"当前值={exp.feature_value:.4f}"
        )
    lines.append("")

    lines.append("TOP 降低流失风险特征:")
    for i, exp in enumerate(result.top_negative_drivers, 1):
        lines.append(
            f"  {i}. {exp.feature_label}: 影响值={exp.shap_value:+.6f}, "
            f"当前值={exp.feature_value:.4f}"
        )
    lines.append("")

    lines.append("所有特征详细贡献:")
    lines.append(f"  {'特征':<20s} {'特征值':>12s} {'SHAP值':>12s} {'影响方向'}")
    lines.append("  " + "-" * 60)
    for exp in result.explanations:
        lines.append(
            f"  {exp.feature_label:<20s} {exp.feature_value:12.4f} "
            f"{exp.shap_value:+12.6f} {exp.impact_direction}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def format_shap_global_report(result: SHAPGlobalResult) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("SHAP 全局特征重要性报告")
    lines.append("=" * 60)
    lines.append(f"样本数量: {result.sample_count}")
    lines.append("")

    lines.append(f"{'排名':>4s}  {'特征':<20s}  {'平均|SHAP|':>12s}")
    lines.append("-" * 50)

    for i, feat in enumerate(result.feature_ranking, 1):
        label = FEATURE_LABELS.get(feat, feat)
        lines.append(f"{i:4d}  {label:<20s}  {result.mean_abs_shap[feat]:12.6f}")

    lines.append("=" * 60)
    return "\n".join(lines)


@dataclass
class AutoKMeansResult:
    optimal_k: int
    scores: Dict[int, Dict[str, float]]
    method_used: str
    elbow_point: Optional[int]
    silhouette_best: Optional[int]
    ch_best: Optional[int]
    gap_best: Optional[int]


@dataclass
class SHAPSamplingConfig:
    strategy: str = "auto"
    max_samples: int = 500
    background_samples: int = 100
    shap_evaluation_samples: int = 50
    quantile_bins: int = 5
    importance_weighted: bool = True
    risk_stratified: bool = True
    feature_diversity: bool = True
    min_samples_per_bin: int = 10
    random_state: int = 42
    auto_kmeans: bool = True
    k_min: int = 2
    k_max: int = 20
    k_step: int = 1
    k_selection_method: str = "ensemble"
    elbow_sensitivity: float = 0.85
    min_samples_per_cluster: int = 5
    use_gap_statistic: bool = False
    gap_bootstrap_samples: int = 50


@dataclass
class SHAPSamplingReport:
    total_accounts: int
    sampled_count: int
    strategy_used: str
    sampling_ratio: float
    risk_distribution_original: Dict[str, int]
    risk_distribution_sampled: Dict[str, int]
    feature_importance_weights: Dict[str, float]
    sample_selection_reason: str


class SHAPLargeScaleSampler:
    def __init__(self, config: Optional[SHAPSamplingConfig] = None):
        self.config = config or SHAPSamplingConfig()

    def _compute_risk_stratification(self, accounts: List[AccountBehavior]) -> List[str]:
        if not accounts:
            return []
        risk_buckets = []
        for acc in accounts:
            score = 0
            if acc.last_login_days_ago > 30:
                score += 2
            elif acc.last_login_days_ago > 14:
                score += 1
            if acc.login_count_last_30d < 3:
                score += 2
            elif acc.login_count_last_30d < 10:
                score += 1
            if acc.payment_failures_last_30d > 2:
                score += 1
            if score >= 4:
                risk_buckets.append("high")
            elif score >= 2:
                risk_buckets.append("medium")
            else:
                risk_buckets.append("low")
        return risk_buckets

    def _compute_feature_diversity_score(self, accounts: List[AccountBehavior]) -> np.ndarray:
        if not accounts:
            return np.array([])
        X = _build_feature_frame(accounts)
        X_norm = (X - X.min()) / (X.max() - X.min() + 1e-9)
        X_norm = X_norm.fillna(0)
        diversity = np.zeros(len(accounts))
        for col in X_norm.columns:
            vals = X_norm[col].values
            median_val = np.median(vals)
            diversity += np.abs(vals - median_val)
        if len(accounts) > 0:
            diversity = diversity / len(X_norm.columns)
        return diversity

    def _auto_select_strategy(self, total: int) -> str:
        if total <= self.config.max_samples:
            return "full"
        if total <= self.config.max_samples * 3:
            return "stratified"
        if total <= self.config.max_samples * 10:
            return "importance_weighted"
        return "importance_weighted"

    def sample(self, accounts: List[AccountBehavior]) -> Tuple[List[AccountBehavior], SHAPSamplingReport]:
        total = len(accounts)
        max_n = self.config.max_samples

        if total == 0:
            empty_report = SHAPSamplingReport(
                total_accounts=0,
                sampled_count=0,
                strategy_used="empty",
                sampling_ratio=0.0,
                risk_distribution_original={},
                risk_distribution_sampled={},
                feature_importance_weights={},
                sample_selection_reason="无账号数据",
            )
            return [], empty_report

        if total <= max_n:
            risk_dist = {}
            for r in self._compute_risk_stratification(accounts):
                risk_dist[r] = risk_dist.get(r, 0) + 1
            return accounts, SHAPSamplingReport(
                total_accounts=total,
                sampled_count=total,
                strategy_used="full",
                sampling_ratio=1.0,
                risk_distribution_original=risk_dist,
                risk_distribution_sampled=risk_dist,
                feature_importance_weights={},
                sample_selection_reason=f"样本数 {total} <= 上限 {max_n}，全量分析",
            )

        strategy = self.config.strategy
        if strategy == "auto":
            strategy = self._auto_select_strategy(total)

        np.random.seed(self.config.random_state)
        risk_buckets = self._compute_risk_stratification(accounts)
        original_risk_dist: Dict[str, int] = {}
        for r in risk_buckets:
            original_risk_dist[r] = original_risk_dist.get(r, 0) + 1

        selected_indices: List[int] = []
        reason_parts = []

        if strategy == "random":
            selected_indices = list(np.random.choice(
                total, size=max_n, replace=False))
            reason_parts.append(f"随机采样 {max_n}/{total}")

        elif strategy == "stratified" and self.config.risk_stratified:
            bucket_indices: Dict[str, List[int]] = {"high": [], "medium": [], "low": []}
            for i, r in enumerate(risk_buckets):
                bucket_indices.setdefault(r, []).append(i)

            per_bucket = max_n // 3
            extra = max_n % 3
            for bucket, idxs in bucket_indices.items():
                n = min(len(idxs), per_bucket + (1 if extra > 0 and bucket == "high" else 0))
                if n > 0:
                    sample = np.random.choice(idxs, size=min(n, len(idxs)), replace=False)
                    selected_indices.extend(sample.tolist())

            if len(selected_indices) < max_n:
                remaining = [i for i in range(total) if i not in set(selected_indices)]
                add_n = min(max_n - len(selected_indices), len(remaining))
                if add_n > 0:
                    add = np.random.choice(remaining, size=add_n, replace=False)
                    selected_indices.extend(add.tolist())
            reason_parts.append(f"按风险分层采样: high/medium/low ≈ {per_bucket}+{extra}/{per_bucket}/{per_bucket}")

        elif strategy == "quantile":
            X = _build_feature_frame(accounts)
            composite_score = X["recency"] * 0.3 - X["login_frequency_30d"] * 0.2 + X["payment_issues"] * 0.2
            bins = pd.qcut(composite_score.rank(method="first"), q=self.config.quantile_bins, labels=False)
            per_bin = max_n // self.config.quantile_bins
            for b in range(self.config.quantile_bins):
                bin_idx = np.where(bins == b)[0]
                if len(bin_idx) > 0:
                    n = min(per_bin, len(bin_idx))
                    sample = np.random.choice(bin_idx, size=n, replace=False)
                    selected_indices.extend(sample.tolist())
            reason_parts.append(f"按风险分位数采样: {self.config.quantile_bins} 组，每组约 {per_bin}")

        else:
            weights = np.ones(total)

            if self.config.risk_stratified:
                for i, r in enumerate(risk_buckets):
                    if r == "high":
                        weights[i] *= 3.0
                    elif r == "medium":
                        weights[i] *= 1.5

            if self.config.feature_diversity:
                div_scores = self._compute_feature_diversity_score(accounts)
                if len(div_scores) > 0:
                    div_norm = div_scores / (div_scores.max() + 1e-9)
                    weights *= (1.0 + div_norm * 2.0)

            weights = weights / weights.sum()
            selected_indices = list(np.random.choice(
                total, size=max_n, replace=False, p=weights))
            reason_parts.append(f"重要性加权采样: 风险×多样性")

        importance_weights = {}
        if self.config.importance_weighted:
            importance_weights = {
                "recency_weight": 3.0 if self.config.risk_stratified else 1.0,
                "payment_issues_weight": 2.0 if self.config.risk_stratified else 1.0,
                "diversity_boost": 2.0 if self.config.feature_diversity else 1.0,
            }

        selected_accounts = [accounts[i] for i in selected_indices]
        sampled_risk = {}
        for r in self._compute_risk_stratification(selected_accounts):
            sampled_risk[r] = sampled_risk.get(r, 0) + 1

        report = SHAPSamplingReport(
            total_accounts=total,
            sampled_count=len(selected_accounts),
            strategy_used=strategy,
            sampling_ratio=round(len(selected_accounts) / total, 4),
            risk_distribution_original=original_risk_dist,
            risk_distribution_sampled=sampled_risk,
            feature_importance_weights=importance_weights,
            sample_selection_reason="; ".join(reason_parts) if reason_parts else strategy,
        )

        return selected_accounts, report

    def find_optimal_k(
        self,
        X: pd.DataFrame,
    ) -> Optional[AutoKMeansResult]:
        if len(X) < self.config.k_min * 2:
            return None

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
        except ImportError:
            return None

        n_samples = len(X)
        max_possible_k = min(
            self.config.k_max,
            n_samples // self.config.min_samples_per_cluster,
            len(X) // 2,
        )
        if max_possible_k < self.config.k_min:
            return None

        k_range = list(range(self.config.k_min, max_possible_k + 1, self.config.k_step))
        if not k_range:
            return None

        inertias = []
        silhouette_scores = []
        ch_scores = []
        gap_scores = []
        reference_inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=3)
            labels = kmeans.fit_predict(X.values)
            inertias.append(kmeans.inertia_)

            if k >= 2 and len(set(labels)) >= 2:
                sil_score = silhouette_score(X.values, labels, random_state=self.config.random_state)
                silhouette_scores.append(sil_score)
                ch_score = calinski_harabasz_score(X.values, labels)
                ch_scores.append(ch_score)
            else:
                silhouette_scores.append(-1.0)
                ch_scores.append(0.0)

            if self.config.use_gap_statistic:
                ref_inertias = []
                for _ in range(self.config.gap_bootstrap_samples):
                    reference_dist = np.random.uniform(
                        low=X.min().values,
                        high=X.max().values,
                        size=X.shape
                    )
                    ref_kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=3)
                    ref_kmeans.fit(reference_dist)
                    ref_inertias.append(ref_kmeans.inertia_)
                mean_ref_inertia = float(np.mean(ref_inertias))
                reference_inertias.append(mean_ref_inertia)
                log_inertia = np.log(kmeans.inertia_ + 1e-10)
                log_ref = np.log(mean_ref_inertia + 1e-10)
                gap_scores.append(log_ref - log_inertia)
            else:
                reference_inertias.append(0.0)
                gap_scores.append(0.0)

        scores = {}
        for i, k in enumerate(k_range):
            scores[k] = {
                "inertia": float(inertias[i]),
                "silhouette": float(silhouette_scores[i]),
                "calinski_harabasz": float(ch_scores[i]),
                "gap": float(gap_scores[i]) if self.config.use_gap_statistic else 0.0,
                "reference_inertia": float(reference_inertias[i]),
            }

        elbow_point = None
        if len(inertias) >= 3:
            inertia_arr = np.array(inertias)
            first_derivative = np.diff(inertia_arr)
            second_derivative = np.diff(first_derivative)
            if len(second_derivative) >= 1:
                elbow_idx = int(np.argmax(second_derivative)) + 2
                elbow_point = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]

        silhouette_best = None
        valid_sil = [(i, s) for i, s in enumerate(silhouette_scores) if s > 0]
        if valid_sil:
            best_idx = max(valid_sil, key=lambda x: x[1])[0]
            silhouette_best = k_range[best_idx]

        ch_best = None
        if ch_scores:
            ch_arr = np.array(ch_scores)
            ch_diff = np.diff(ch_arr)
            if len(ch_diff) >= 1:
                ch_changes = ch_diff[:-1] / (ch_diff[1:] + 1e-9)
                ch_best_idx = int(np.argmax(ch_changes)) + 2 if len(ch_changes) >= 1 else np.argmax(ch_arr)
                ch_best = k_range[min(ch_best_idx, len(k_range) - 1)]

        gap_best = None
        if self.config.use_gap_statistic and gap_scores:
            gap_best_idx = int(np.argmax(gap_scores))
            gap_best = k_range[gap_best_idx]

        candidates = []
        if elbow_point is not None:
            candidates.append(elbow_point)
        if silhouette_best is not None:
            candidates.append(silhouette_best)
        if ch_best is not None:
            candidates.append(ch_best)
        if gap_best is not None:
            candidates.append(gap_best)

        method_used = self.config.k_selection_method
        if method_used == "elbow" and elbow_point is not None:
            optimal_k = elbow_point
        elif method_used == "silhouette" and silhouette_best is not None:
            optimal_k = silhouette_best
        elif method_used == "calinski_harabasz" and ch_best is not None:
            optimal_k = ch_best
        elif method_used == "gap" and gap_best is not None:
            optimal_k = gap_best
        elif method_used == "ensemble" and candidates:
            from collections import Counter
            counts = Counter(candidates)
            optimal_k = counts.most_common(1)[0][0]
        else:
            optimal_k = elbow_point or silhouette_best or ch_best or k_range[len(k_range) // 2]
            method_used = "fallback"

        return AutoKMeansResult(
            optimal_k=optimal_k,
            scores=scores,
            method_used=method_used,
            elbow_point=elbow_point,
            silhouette_best=silhouette_best,
            ch_best=ch_best,
            gap_best=gap_best,
        )

    def _adaptive_background_samples_internal(
        self,
        X: pd.DataFrame,
        target: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Optional[AutoKMeansResult]]:
        if random_state is None:
            random_state = self.config.random_state

        if len(X) <= target:
            return X, None

        auto_result = None
        n_clusters = min(target, len(X) // 5, 50)

        if self.config.auto_kmeans and len(X) >= self.config.k_min * 2:
            auto_result = self.find_optimal_k(X)
            if auto_result is not None:
                n_clusters = min(auto_result.optimal_k, target, len(X) // 2)
                n_clusters = max(self.config.k_min, n_clusters)

        try:
            from sklearn.cluster import KMeans
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
                labels = kmeans.fit_predict(X.values)
                selected = []
                for cluster in range(n_clusters):
                    cluster_idx = np.where(labels == cluster)[0]
                    if len(cluster_idx) > 0:
                        distances = np.sum((X.values[cluster_idx] - kmeans.cluster_centers_[cluster]) ** 2, axis=1)
                        closest_idx = cluster_idx[np.argmin(distances)]
                        selected.append(int(closest_idx))
                remaining = [i for i in range(len(X)) if i not in selected]
                fill = min(target - len(selected), len(remaining))
                if fill > 0:
                    np.random.seed(random_state)
                    extra = np.random.choice(remaining, size=fill, replace=False)
                    selected.extend(extra.tolist())
                return X.iloc[selected[:target]], auto_result
        except Exception:
            pass

        np.random.seed(random_state)
        idx = np.random.choice(len(X), size=min(target, len(X)), replace=False)
        return X.iloc[idx], auto_result

    def adaptive_background_samples_with_kmeans(
        self,
        X: pd.DataFrame,
        target: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Optional[AutoKMeansResult]]:
        return self._adaptive_background_samples_internal(X, target, random_state)

    @staticmethod
    def adaptive_background_samples(
        X: pd.DataFrame,
        target: int = 100,
        random_state: int = 42,
    ) -> pd.DataFrame:
        sampler = SHAPLargeScaleSampler(SHAPSamplingConfig(random_state=random_state))
        result, _ = sampler._adaptive_background_samples_internal(X, target=target, random_state=random_state)
        return result


def format_sampling_report(report: SHAPSamplingReport) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("SHAP 大规模采样报告")
    lines.append("=" * 60)
    lines.append(f"总账号数: {report.total_accounts}")
    lines.append(f"采样账号数: {report.sampled_count}")
    lines.append(f"采样比例: {report.sampling_ratio * 100:.2f}%")
    lines.append(f"采样策略: {report.strategy_used}")
    lines.append(f"选择依据: {report.sample_selection_reason}")
    lines.append("")

    if report.risk_distribution_original:
        lines.append("风险分布对比 (原始 vs 采样):")
        all_risk = set(list(report.risk_distribution_original.keys()) + list(report.risk_distribution_sampled.keys()))
        for risk in sorted(all_risk):
            orig = report.risk_distribution_original.get(risk, 0)
            samp = report.risk_distribution_sampled.get(risk, 0)
            orig_pct = orig / report.total_accounts * 100 if report.total_accounts > 0 else 0
            samp_pct = samp / report.sampled_count * 100 if report.sampled_count > 0 else 0
            lines.append(
                f"  {risk:<8s}: 原始 {orig:4d} ({orig_pct:5.1f}%) -> "
                f"采样 {samp:4d} ({samp_pct:5.1f}%)"
            )

    if report.feature_importance_weights:
        lines.append("")
        lines.append("采样权重参数:")
        for k, v in report.feature_importance_weights.items():
            lines.append(f"  {k}: {v}")

    lines.append("=" * 60)
    return "\n".join(lines)


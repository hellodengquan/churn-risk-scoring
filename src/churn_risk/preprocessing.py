from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from .models import AccountBehavior


NUMERIC_FEATURE_COLS = [
    "login_count_last_30d",
    "login_count_last_7d",
    "last_login_days_ago",
    "total_transactions_last_30d",
    "total_transactions_last_90d",
    "transaction_amount_last_30d",
    "avg_session_minutes",
    "support_tickets_last_30d",
    "refund_count_last_90d",
    "payment_failures_last_30d",
    "email_open_rate",
    "subscription_age_days",
]


def accounts_to_dataframe(accounts: List[AccountBehavior]) -> pd.DataFrame:
    records = []
    for a in accounts:
        record = {
            "account_id": a.account_id,
            "login_count_last_30d": a.login_count_last_30d,
            "login_count_last_7d": a.login_count_last_7d,
            "last_login_days_ago": a.last_login_days_ago,
            "total_transactions_last_30d": a.total_transactions_last_30d,
            "total_transactions_last_90d": a.total_transactions_last_90d,
            "transaction_amount_last_30d": a.transaction_amount_last_30d,
            "avg_session_minutes": a.avg_session_minutes,
            "support_tickets_last_30d": a.support_tickets_last_30d,
            "refund_count_last_90d": a.refund_count_last_90d,
            "payment_failures_last_30d": a.payment_failures_last_30d,
            "email_open_rate": a.email_open_rate,
            "subscription_age_days": a.subscription_age_days,
            "plan_level": a.plan_level,
        }
        records.append(record)
    return pd.DataFrame(records)


def dataframe_to_accounts(df: pd.DataFrame) -> List[AccountBehavior]:
    accounts = []
    for _, row in df.iterrows():
        account = AccountBehavior(
            account_id=str(row.get("account_id", "")),
            login_count_last_30d=int(row.get("login_count_last_30d", 0)),
            login_count_last_7d=int(row.get("login_count_last_7d", 0)),
            last_login_days_ago=int(row.get("last_login_days_ago", 999)),
            total_transactions_last_30d=int(row.get("total_transactions_last_30d", 0)),
            total_transactions_last_90d=int(row.get("total_transactions_last_90d", 0)),
            transaction_amount_last_30d=float(row.get("transaction_amount_last_30d", 0.0)),
            avg_session_minutes=float(row.get("avg_session_minutes", 0.0)),
            support_tickets_last_30d=int(row.get("support_tickets_last_30d", 0)),
            refund_count_last_90d=int(row.get("refund_count_last_90d", 0)),
            payment_failures_last_30d=int(row.get("payment_failures_last_30d", 0)),
            feature_usage_count={},
            email_open_rate=float(row.get("email_open_rate", 0.0)),
            subscription_age_days=int(row.get("subscription_age_days", 0)),
            plan_level=str(row.get("plan_level", "basic")),
        )
        accounts.append(account)
    return accounts


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    fill_values: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    result = df.copy()
    fill_map = fill_values.copy() if fill_values else {}

    for col in NUMERIC_FEATURE_COLS:
        if col not in result.columns:
            continue
        if col not in fill_map:
            if strategy == "median":
                fill_map[col] = float(result[col].median()) if not result[col].isna().all() else 0.0
            elif strategy == "mean":
                fill_map[col] = float(result[col].mean()) if not result[col].isna().all() else 0.0
            elif strategy == "zero":
                fill_map[col] = 0.0
            else:
                fill_map[col] = 0.0
        result[col] = result[col].fillna(fill_map[col])

    if "plan_level" in result.columns:
        result["plan_level"] = result["plan_level"].fillna("basic")
    if "account_id" in result.columns:
        result["account_id"] = result["account_id"].fillna("UNKNOWN")

    return result, fill_map


def detect_outliers_iqr(
    series: pd.Series,
    iqr_factor: float = 1.5,
) -> Tuple[np.ndarray, float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    mask = (series < lower_bound) | (series > upper_bound)
    return mask.values, lower_bound, upper_bound


def handle_outliers(
    df: pd.DataFrame,
    method: str = "clip",
    iqr_factor: float = 1.5,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], Dict[str, int]]:
    result = df.copy()
    computed_bounds = {}
    outlier_counts = {}

    for col in NUMERIC_FEATURE_COLS:
        if col not in result.columns:
            continue

        if bounds and col in bounds:
            lower, upper = bounds[col]
        else:
            _, lower, upper = detect_outliers_iqr(result[col], iqr_factor)

        computed_bounds[col] = (lower, upper)

        if method == "clip":
            outlier_mask = (result[col] < lower) | (result[col] > upper)
            outlier_counts[col] = int(outlier_mask.sum())
            result[col] = result[col].clip(lower=lower, upper=upper)
        elif method == "remove":
            outlier_mask = (result[col] < lower) | (result[col] > upper)
            outlier_counts[col] = int(outlier_mask.sum())
            result = result[~outlier_mask]
        elif method == "median":
            outlier_mask = (result[col] < lower) | (result[col] > upper)
            outlier_counts[col] = int(outlier_mask.sum())
            median_val = result[col].median()
            result.loc[outlier_mask, col] = median_val

    return result, computed_bounds, outlier_counts


def min_max_normalize(
    df: pd.DataFrame,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    result = df.copy()
    computed_ranges = {}

    for col in NUMERIC_FEATURE_COLS:
        if col not in result.columns:
            continue

        if ranges and col in ranges:
            min_val, max_val = ranges[col]
        else:
            min_val = float(result[col].min())
            max_val = float(result[col].max())

        computed_ranges[col] = (min_val, max_val)

        if max_val != min_val:
            result[col] = (result[col] - min_val) / (max_val - min_val)
        else:
            result[col] = 0.0

    return result, computed_ranges


def zscore_normalize(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    result = df.copy()
    computed_stats = {}

    for col in NUMERIC_FEATURE_COLS:
        if col not in result.columns:
            continue

        if stats and col in stats:
            mean, std = stats[col]
        else:
            mean = float(result[col].mean())
            std = float(result[col].std())

        computed_stats[col] = (mean, std)

        if std > 0:
            result[col] = (result[col] - mean) / std
        else:
            result[col] = 0.0

    return result, computed_stats


def robust_normalize(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float, float]]]:
    result = df.copy()
    computed_stats = {}

    for col in NUMERIC_FEATURE_COLS:
        if col not in result.columns:
            continue

        if stats and col in stats:
            median, q1, q3 = stats[col]
        else:
            median = float(result[col].median())
            q1 = float(result[col].quantile(0.25))
            q3 = float(result[col].quantile(0.75))

        computed_stats[col] = (median, q1, q3)
        iqr = q3 - q1

        if iqr > 0:
            result[col] = (result[col] - median) / iqr
        else:
            result[col] = 0.0

    return result, computed_stats


def preprocess_pipeline(
    accounts: List[AccountBehavior],
    missing_strategy: str = "median",
    outlier_method: str = "clip",
    normalization: str = "minmax",
) -> Dict:
    df = accounts_to_dataframe(accounts)

    df_clean, fill_map = handle_missing_values(df, strategy=missing_strategy)

    df_no_outliers, bounds, outlier_counts = handle_outliers(
        df_clean, method=outlier_method
    )

    norm_params = {}
    if normalization == "minmax":
        df_normalized, norm_params = min_max_normalize(df_no_outliers)
    elif normalization == "zscore":
        df_normalized, norm_params = zscore_normalize(df_no_outliers)
    elif normalization == "robust":
        df_normalized, norm_params = robust_normalize(df_no_outliers)
    else:
        df_normalized = df_no_outliers

    processed_accounts = dataframe_to_accounts(df_normalized)

    return {
        "original_df": df,
        "clean_df": df_clean,
        "normalized_df": df_normalized,
        "processed_accounts": processed_accounts,
        "fill_map": fill_map,
        "outlier_bounds": bounds,
        "outlier_counts": outlier_counts,
        "normalization_params": norm_params,
    }


def preprocess_report(preprocess_result: Dict) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append("特征预处理报告")
    lines.append("=" * 50)

    original_count = len(preprocess_result["original_df"])
    clean_count = len(preprocess_result["clean_df"])
    lines.append(f"原始样本数: {original_count}")
    lines.append(f"清洗后样本数: {clean_count}")
    lines.append("")

    lines.append("缺失值填充:")
    for col, val in preprocess_result["fill_map"].items():
        lines.append(f"  {col}: {val:.4f}")
    lines.append("")

    lines.append("异常值统计:")
    for col, count in preprocess_result["outlier_counts"].items():
        lines.append(f"  {col}: {count} 个异常值")
    lines.append("")

    lines.append("归一化参数:")
    for col, params in preprocess_result["normalization_params"].items():
        if isinstance(params, tuple):
            params_str = ", ".join(f"{p:.4f}" for p in params)
            lines.append(f"  {col}: ({params_str})")

    lines.append("=" * 50)
    return "\n".join(lines)

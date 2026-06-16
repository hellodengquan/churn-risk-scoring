from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .models import RiskScore
from .shap_analysis import SHAPGlobalResult, SHAPAccountResult
from .ltv import calculate_portfolio_summary


def _safe_matplotlib_import():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_risk_distribution(
    scores: List[RiskScore],
    output_path: str,
    title: str = "风险等级分布",
) -> bool:
    plt = _safe_matplotlib_import()
    if plt is None:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    levels = ["critical", "high", "medium", "low"]
    level_labels = ["严重风险", "高风险", "中风险", "低风险"]
    colors = ["#dc3545", "#fd7e14", "#ffc107", "#28a745"]

    counts = [sum(1 for s in scores if s.risk_level == lvl) for lvl in levels]
    total = sum(counts) if sum(counts) > 0 else 1
    percentages = [c / total * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(level_labels, counts, color=colors, edgecolor="white")
    ax1.set_title(f"{title} - 数量分布", fontsize=14, fontweight="bold")
    ax1.set_ylabel("账号数量")
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center", va="bottom", fontweight="bold",
        )

    wedges, texts, autotexts = ax2.pie(
        percentages, labels=level_labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
    )
    ax2.set_title(f"{title} - 占比", fontsize=14, fontweight="bold")
    for t in autotexts:
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def plot_score_histogram(
    scores: List[RiskScore],
    output_path: str,
    title: str = "风险评分分布直方图",
) -> bool:
    plt = _safe_matplotlib_import()
    if plt is None:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    score_values = [s.total_score for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins, patches = ax.hist(
        score_values, bins=20, color="#4e79a7", edgecolor="white", linewidth=1, alpha=0.85
    )

    for bin_left, patch in zip(bins, patches):
        if bin_left >= 80:
            patch.set_facecolor("#dc3545")
        elif bin_left >= 60:
            patch.set_facecolor("#fd7e14")
        elif bin_left >= 30:
            patch.set_facecolor("#ffc107")
        else:
            patch.set_facecolor("#28a745")

    ax.axvline(np.mean(score_values), color="#e15759", linestyle="--", linewidth=2, label=f"均值: {np.mean(score_values):.2f}")
    ax.axvline(np.median(score_values), color="#76b7b2", linestyle="--", linewidth=2, label=f"中位数: {np.median(score_values):.2f}")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("风险评分 (0-100)")
    ax.set_ylabel("账号数量")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def plot_feature_importance(
    shap_result: SHAPGlobalResult,
    output_path: str,
    title: str = "SHAP 全局特征重要性",
    top_n: int = 10,
) -> bool:
    plt = _safe_matplotlib_import()
    if plt is None:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    from .shap_analysis import FEATURE_LABELS

    ranking = shap_result.feature_ranking[:top_n]
    values = [shap_result.mean_abs_shap[f] for f in ranking]
    labels = [FEATURE_LABELS.get(f, f) for f in ranking]

    fig, ax = plt.subplots(figsize=(10, max(6, len(ranking) * 0.6)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color="#59a14f", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.6f}",
            va="center", fontsize=10,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("平均 |SHAP| 值")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def plot_ltv_vs_risk(
    actions: List[Dict],
    output_path: str,
    title: str = "LTV vs 流失风险 散点图",
) -> bool:
    plt = _safe_matplotlib_import()
    if plt is None:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    risk_colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#28a745",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for level, color in risk_colors.items():
        level_data = [a for a in actions if a["risk_level"] == level]
        if level_data:
            risks = [a["risk_score"] for a in level_data]
            ltvs = [a["predicted_ltv"] for a in level_data]
            sizes = [min(500, max(50, a["monthly_revenue"] / 10)) for a in level_data]
            ax.scatter(
                risks, ltvs, s=sizes, c=color, alpha=0.7, edgecolor="white", linewidth=0.5,
                label=f"{level} (n={len(level_data)})",
            )

    ax.axhline(y=10000, color="#b07aa1", linestyle="--", alpha=0.7, label="高价值线 (¥10,000)")
    ax.axvline(x=70, color="#e15759", linestyle="--", alpha=0.7, label="高风险线 (70分)")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("流失风险评分")
    ax.set_ylabel("预测 LTV (¥)")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def plot_shap_waterfall(
    shap_result: SHAPAccountResult,
    output_path: str,
    title: Optional[str] = None,
) -> bool:
    plt = _safe_matplotlib_import()
    if plt is None:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    from .shap_analysis import FEATURE_LABELS

    top_exps = shap_result.explanations[:8]
    labels = [FEATURE_LABELS.get(e.feature_name, e.feature_name) for e in top_exps]
    values = [e.shap_value for e in top_exps]

    fig, ax = plt.subplots(figsize=(10, max(5, len(top_exps) * 0.6)))

    y_pos = np.arange(len(labels))
    colors = ["#e15759" if v > 0 else "#59a14f" for v in values]

    current = shap_result.base_value
    for i in range(len(values)):
        start = current
        current += values[i]
        ax.barh(
            y_pos[i], values[i], left=start,
            color=colors[i], alpha=0.85, edgecolor="white", height=0.6,
        )

    ax.axvline(shap_result.base_value, color="gray", linestyle="--", alpha=0.7, label="基准值")
    ax.axvline(shap_result.predicted_score, color="#4e79a7", linestyle="-", linewidth=2, label="预测值")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    display_title = title or f"SHAP 瀑布图 - {shap_result.account_id}"
    ax.set_title(display_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("SHAP 值 (流失概率贡献)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def plot_timeseries(
    timeseries_data: Dict,
    output_path: str,
    title: Optional[str] = None,
) -> bool:
    plt = _safe_matplotlib_import()
    if plt is None:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    metrics_data = timeseries_data.get("metrics", [])
    if not metrics_data:
        return False

    n_metrics = len(metrics_data)
    fig, axes = plt.subplots(
        n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_result in zip(axes, metrics_data):
        values = metric_result.get("values", [])
        metric_name = metric_result.get("metric", "")
        direction = metric_result.get("trend_direction", "")
        is_alert = metric_result.get("is_alert", False)

        x = list(range(len(values)))
        color = "#dc3545" if is_alert else "#4e79a7"

        ax.plot(x, values, marker="o", markersize=4, color=color, linewidth=2, alpha=0.85)
        ax.fill_between(x, values, alpha=0.15, color=color)

        if len(values) >= 2:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", color="#e15759", alpha=0.7, linewidth=1.5, label="趋势线")

        alert_suffix = " ⚠" if is_alert else ""
        ax.set_title(f"{metric_name} ({direction}){alert_suffix}", fontweight="bold")
        ax.set_ylabel("值")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("时间段 (月)")

    display_title = title or f"时间序列趋势 - {timeseries_data.get('account_id', '')}"
    fig.suptitle(display_title, fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def generate_all_visualizations(
    scores: List[RiskScore],
    actions: Optional[List[Dict]] = None,
    shap_global: Optional[SHAPGlobalResult] = None,
    shap_accounts: Optional[List[SHAPAccountResult]] = None,
    timeseries_summaries: Optional[List[Dict]] = None,
    output_dir: str = "output/figures",
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}

    results["risk_distribution"] = plot_risk_distribution(
        scores, f"{output_dir}/risk_distribution.png"
    )

    results["score_histogram"] = plot_score_histogram(
        scores, f"{output_dir}/score_histogram.png"
    )

    if shap_global:
        results["feature_importance"] = plot_feature_importance(
            shap_global, f"{output_dir}/feature_importance.png"
        )

    if actions:
        results["ltv_vs_risk"] = plot_ltv_vs_risk(
            actions, f"{output_dir}/ltv_vs_risk.png"
        )

    if shap_accounts:
        for i, shap_acc in enumerate(shap_accounts[:3]):
            key = f"shap_waterfall_{shap_acc.account_id}"
            results[key] = plot_shap_waterfall(
                shap_acc, f"{output_dir}/shap_waterfall_{shap_acc.account_id}.png"
            )

    if timeseries_summaries:
        for i, ts in enumerate(timeseries_summaries[:2]):
            acc_id = ts.get("account_id", f"account_{i}")
            key = f"timeseries_{acc_id}"
            results[key] = plot_timeseries(
                ts, f"{output_dir}/timeseries_{acc_id}.png"
            )

    return results

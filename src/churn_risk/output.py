import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AccountBehavior, RiskScore


RISK_LEVEL_ORDER = ["critical", "high", "medium", "low"]
RISK_LEVEL_LABELS = {
    "critical": "严重风险",
    "high": "高风险",
    "medium": "中风险",
    "low": "低风险",
}


def filter_by_risk_level(scores: List[RiskScore], min_level: str = "low") -> List[RiskScore]:
    if min_level not in RISK_LEVEL_ORDER:
        raise ValueError(f"Invalid risk level: {min_level}")

    min_index = RISK_LEVEL_ORDER.index(min_level)
    return [s for s in scores if RISK_LEVEL_ORDER.index(s.risk_level) <= min_index]


def group_by_risk_level(scores: List[RiskScore]) -> Dict[str, List[RiskScore]]:
    groups = {level: [] for level in RISK_LEVEL_ORDER}
    for score in scores:
        if score.risk_level in groups:
            groups[score.risk_level].append(score)
    return groups


def get_top_n(scores: List[RiskScore], n: int) -> List[RiskScore]:
    return scores[:n]


def export_to_csv(scores: List[RiskScore], output_path: str, include_features: bool = True) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "account_id",
        "plan_level",
        "total_score",
        "risk_level",
        "risk_percentile",
    ]

    if include_features and scores:
        feature_keys = sorted(scores[0].feature_scores.keys())
        fieldnames.extend([f"feature_{k}" for k in feature_keys])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for score in scores:
            row = {
                "account_id": score.account_id,
                "plan_level": score.plan_level,
                "total_score": score.total_score,
                "risk_level": RISK_LEVEL_LABELS.get(score.risk_level, score.risk_level),
                "risk_percentile": score.risk_percentile,
            }
            if include_features:
                for k, v in score.feature_scores.items():
                    row[f"feature_{k}"] = v
            writer.writerow(row)


def export_to_json(scores: List[RiskScore], output_path: str, indent: int = 2) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data = []
    for score in scores:
        data.append({
            "account_id": score.account_id,
            "plan_level": score.plan_level,
            "total_score": score.total_score,
            "risk_level": score.risk_level,
            "risk_level_label": RISK_LEVEL_LABELS.get(score.risk_level, score.risk_level),
            "risk_percentile": score.risk_percentile,
            "feature_scores": score.feature_scores,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def format_summary(scores: List[RiskScore]) -> str:
    if not scores:
        return "无评分数据"

    total = len(scores)
    distribution = {level: 0 for level in RISK_LEVEL_ORDER}
    for s in scores:
        if s.risk_level in distribution:
            distribution[s.risk_level] += 1

    avg_score = sum(s.total_score for s in scores) / total

    lines = [
        "=" * 50,
        "流失风险评分汇总",
        "=" * 50,
        f"账号总数: {total}",
        f"平均风险分: {avg_score:.2f}",
        "",
        "风险等级分布:",
    ]

    for level in RISK_LEVEL_ORDER:
        count = distribution.get(level, 0)
        pct = count / total * 100 if total > 0 else 0
        label = RISK_LEVEL_LABELS.get(level, level)
        bar = "█" * int(pct / 2)
        lines.append(f"  {label:8s}: {count:4d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("TOP 5 高风险账号:")
    for i, s in enumerate(scores[:5], 1):
        label = RISK_LEVEL_LABELS.get(s.risk_level, s.risk_level)
        lines.append(f"  {i}. {s.account_id} - {s.total_score:.2f} 分 ({label})")

    lines.append("=" * 50)

    return "\n".join(lines)


def format_ranked_list(scores: List[RiskScore], max_items: Optional[int] = None) -> str:
    if not scores:
        return "无评分数据"

    display_scores = scores[:max_items] if max_items else scores

    lines = []
    lines.append(f"{'排名':>4s}  {'账号ID':<20s}  {'风险分':>8s}  {'等级':<8s}  {'百分位':>8s}")
    lines.append("-" * 60)

    for i, s in enumerate(display_scores, 1):
        label = RISK_LEVEL_LABELS.get(s.risk_level, s.risk_level)
        lines.append(
            f"{i:4d}  {s.account_id:<20s}  {s.total_score:8.2f}  {label:<8s}  {s.risk_percentile:7.2f}%"
        )

    return "\n".join(lines)


def get_terminal_width() -> int:
    try:
        import shutil
        size = shutil.get_terminal_size((120, 40))
        return int(size.columns)
    except Exception:
        return 120


def _truncate_text(text: str, max_len: int, ellipsis: str = "...") -> str:
    if len(text) <= max_len:
        return text
    if max_len <= len(ellipsis):
        return text[:max_len]
    return text[:max_len - len(ellipsis)] + ellipsis


def build_rich_risk_tree(
    scores: List[RiskScore],
    actions: Optional[List[Dict]] = None,
    accounts: Optional[List[AccountBehavior]] = None,
    terminal_width: Optional[int] = None,
) -> "Table":
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    from rich.panel import Panel
    from rich import box

    if terminal_width is None:
        terminal_width = get_terminal_width()

    use_wide_mode = terminal_width >= 120
    account_id_width = min(25, max(15, terminal_width // 6))
    action_width = min(60, max(25, terminal_width // 3)) if use_wide_mode else min(40, max(20, terminal_width // 3))

    risk_color_map = {
        "critical": "bold red",
        "high": "bold orange3",
        "medium": "bold yellow",
        "low": "bold green",
    }

    action_map = {}
    if actions:
        action_map = {a["account_id"]: a for a in actions}

    account_map = {}
    if accounts:
        account_map = {a.account_id: a for a in accounts}

    groups = group_by_risk_level(scores)

    root_label = Text("📊 客户流失风险分级清单", style="bold cyan")
    if use_wide_mode:
        root_label.append(f" (共 {len(scores)} 个账号)", style="dim")
    tree = Tree(root_label)

    for level in RISK_LEVEL_ORDER:
        level_scores = groups.get(level, [])
        if not level_scores:
            continue

        label = RISK_LEVEL_LABELS.get(level, level)
        color = risk_color_map.get(level, "white")
        count = len(level_scores)
        pct = count / len(scores) * 100 if scores else 0

        node_label = Text()
        node_label.append(f"{'█' * int(pct / 5):<10s} ", style=color)
        node_label.append(f"{label:<6s} ", style=f"bold {color}" if color != "white" else "bold")
        node_label.append(f"{count:4d} 个 ", style="white")
        node_label.append(f"({pct:5.1f}%)", style="dim")

        node = tree.add(node_label)

        display_count = min(10 if use_wide_mode else 5, len(level_scores))
        for s in level_scores[:display_count]:
            account_id = _truncate_text(s.account_id, account_id_width)

            leaf_label = Text()
            leaf_label.append(f"  {account_id:<{account_id_width + 2}s} ", style="white")
            leaf_label.append(f"{s.total_score:>6.1f} 分", style=color)

            if use_wide_mode and s.risk_percentile is not None:
                leaf_label.append(f"  [前 {s.risk_percentile:>5.1f}%]", style="dim")

            action = action_map.get(s.account_id)
            if action and use_wide_mode:
                action_text = _truncate_text(action.get("recommended_action", ""), action_width)
                ltv_tier = action.get("ltv_tier", "")
                if ltv_tier:
                    tier_color = {
                        "platinum": "bold magenta",
                        "gold": "bold yellow",
                        "silver": "bright_white",
                        "bronze": "yellow",
                    }.get(ltv_tier, "white")
                    leaf_label.append(f"  [{ltv_tier.upper():<8s}]", style=tier_color)
                leaf_label.append(f"  {action_text}", style="cyan")

            node.add(leaf_label)

        if len(level_scores) > display_count:
            more_label = Text()
            more_label.append(f"  ... 还有 {len(level_scores) - display_count} 个", style="dim")
            node.add(more_label)

    return tree


def build_rich_account_detail_tree(
    account_id: str,
    score: RiskScore,
    account: Optional[AccountBehavior] = None,
    ltv: Optional[Any] = None,
    shap_result: Optional[Any] = None,
    trend_result: Optional[Dict] = None,
    terminal_width: Optional[int] = None,
) -> "Tree":
    from rich.tree import Tree
    from rich.text import Text
    from rich import box

    if terminal_width is None:
        terminal_width = get_terminal_width()

    use_wide_mode = terminal_width >= 100

    risk_color_map = {
        "critical": "bold red",
        "high": "bold orange3",
        "medium": "bold yellow",
        "low": "bold green",
    }
    color = risk_color_map.get(score.risk_level, "white")

    root_label = Text()
    root_label.append(f"👤 账号详情: {account_id}", style="bold cyan")
    risk_label = RISK_LEVEL_LABELS.get(score.risk_level, score.risk_level)
    root_label.append(f"  [{risk_label}]", style=color)
    root_label.append(f"  {score.total_score:.1f} 分", style="white")
    root_label.append(f"  [前 {score.risk_percentile:.1f}%]", style="dim")

    tree = Tree(root_label)

    if account:
        info_node = tree.add(Text("📋 基础信息", style="bold blue"))
        info_node.add(f"  套餐等级: {account.plan_level}")
        info_node.add(f"  订阅时长: {account.subscription_age_days} 天")
        info_node.add(f"  最近登录: {account.last_login_days_ago} 天前")
        info_node.add(f"  30天登录: {account.login_count_last_30d} 次")
        info_node.add(f"  30天交易: {account.total_transactions_last_30d} 笔, ¥{account.transaction_amount_last_30d:,.2f}")

    if score.feature_scores and use_wide_mode:
        feat_node = tree.add(Text("📈 特征得分", style="bold blue"))
        for feat_name, feat_score in sorted(score.feature_scores.items(), key=lambda x: -x[1]):
            score_color = "red" if feat_score >= 70 else "yellow" if feat_score >= 40 else "green"
            bar = "█" * int(feat_score / 5)
            feat_node.add(
                f"  {feat_name:<25s} [{bar:<20s}] {feat_score:>6.1f}",
                style=score_color,
            )

    if ltv and hasattr(ltv, 'predicted_ltv'):
        ltv_node = tree.add(Text("💰 LTV 信息", style="bold magenta"))
        ltv_node.add(f"  预测 LTV: ¥{ltv.predicted_ltv:,.2f}")
        ltv_node.add(f"  历史 LTV: ¥{ltv.historical_ltv:,.2f}")
        ltv_node.add(f"  月收入: ¥{ltv.monthly_revenue:,.2f}")
        ltv_node.add(f"  流失成本: ¥{ltv.churn_cost:,.2f}")
        if hasattr(ltv, 'ltv_tier'):
            ltv_node.add(f"  LTV 等级: {ltv.ltv_tier.upper()}")

    if shap_result and hasattr(shap_result, 'top_positive_drivers'):
        shap_node = tree.add(Text("🧠 SHAP 特征解释", style="bold purple"))
        shap_node.add(f"  预测流失概率: {shap_result.predicted_score:.2%}")
        pos_node = shap_node.add(Text("  ⚠️  增加流失风险:", style="red"))
        for exp in shap_result.top_positive_drivers[:3]:
            pos_node.add(
                f"    {exp.feature_label}: {exp.shap_value:+.4f} (值: {exp.feature_value:.2f})"
            )
        neg_node = shap_node.add(Text("  ✅ 降低流失风险:", style="green"))
        for exp in shap_result.top_negative_drivers[:3]:
            neg_node.add(
                f"    {exp.feature_label}: {exp.shap_value:+.4f} (值: {exp.feature_value:.2f})"
            )

    if trend_result:
        trend_node = tree.add(Text("📉 趋势分析", style="bold yellow"))
        overall = trend_result.get("overall_trend_risk", "unknown")
        trend_color = "red" if overall == "high" else "yellow" if overall == "medium" else "green"
        trend_node.add(f"  整体趋势风险: {overall}", style=trend_color)
        trend_node.add(f"  告警数量: {trend_result.get('alert_count', 0)}")
        trend_node.add(f"  下降指标数: {trend_result.get('declining_metrics', 0)}")
        for alert in trend_result.get("active_alerts", [])[:3]:
            trend_node.add(f"    ⚠ {alert['metric']}: {alert['message']}")

    return tree


def print_terminal_width_info() -> str:
    width = get_terminal_width()
    mode = "宽屏模式" if width >= 120 else "标准模式" if width >= 80 else "窄屏模式"
    return f"终端宽度: {width} 列 | 显示模式: {mode}"


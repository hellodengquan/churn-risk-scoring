import csv
import json
from pathlib import Path
from typing import List, Dict, Optional

from .models import RiskScore


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

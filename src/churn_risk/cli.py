from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, save_default_config
from .features import load_accounts, collect_feature_summary
from .scoring import score_accounts, get_risk_distribution
from .output import (
    export_to_csv,
    export_to_json,
    format_summary,
    format_ranked_list,
    filter_by_risk_level,
    get_top_n,
    RISK_LEVEL_LABELS,
)


app = typer.Typer(
    name="churn-score",
    help="客户流失风险评分工具 - 识别高风险流失账号",
    add_completion=False,
)
console = Console()


@app.command()
def score(
    input_file: str = typer.Argument(..., help="输入数据文件路径 (CSV 或 JSON)"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    output_csv: Optional[str] = typer.Option(None, "--csv", help="导出 CSV 文件路径"),
    output_json: Optional[str] = typer.Option(None, "--json", help="导出 JSON 文件路径"),
    min_risk: str = typer.Option("low", "--min-risk", "-m", help="最低风险等级筛选 (low/medium/high/critical)"),
    top_n: Optional[int] = typer.Option(None, "--top", "-n", help="仅显示前 N 个高风险账号"),
    show_features: bool = typer.Option(False, "--features", "-f", help="显示各特征详细得分"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式，仅输出结果文件"),
):
    """对账号进行流失风险评分并输出分级清单"""

    try:
        config = load_config(config_file)
    except FileNotFoundError as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    if not accounts:
        typer.echo("警告: 未找到任何账号数据", err=True)
        raise typer.Exit(0)

    scores = score_accounts(accounts, config)

    if min_risk != "low":
        scores = filter_by_risk_level(scores, min_risk)

    if top_n:
        scores = get_top_n(scores, top_n)

    if not quiet:
        _print_results(scores, show_features)

    if output_csv:
        export_to_csv(scores, output_csv, include_features=show_features)
        if not quiet:
            typer.echo(f"\n✓ CSV 结果已保存至: {output_csv}")

    if output_json:
        export_to_json(scores, output_json)
        if not quiet:
            typer.echo(f"✓ JSON 结果已保存至: {output_json}")


@app.command("summary")
def show_summary(
    input_file: str = typer.Argument(..., help="输入数据文件路径 (CSV 或 JSON)"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """显示数据汇总和风险分布概览"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    config = load_config(config_file)
    scores = score_accounts(accounts, config)
    summary = format_summary(scores)
    typer.echo(summary)

    feature_summary = collect_feature_summary(accounts)
    typer.echo("\n特征统计:")
    for key, value in feature_summary.items():
        if isinstance(value, dict):
            typer.echo(f"  {key}:")
            for k, v in value.items():
                typer.echo(f"    {k}: {v}")
        else:
            typer.echo(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


@app.command("init-config")
def init_config(
    output_path: str = typer.Argument("config.yaml", help="配置文件输出路径"),
    force: bool = typer.Option(False, "--force", "-f", help="强制覆盖已存在的文件"),
):
    """生成默认配置文件模板"""

    path = Path(output_path)
    if path.exists() and not force:
        typer.echo(f"错误: 文件已存在: {output_path}", err=True)
        typer.echo("使用 --force 选项覆盖", err=True)
        raise typer.Exit(1)

    save_default_config(output_path)
    typer.echo(f"✓ 默认配置文件已生成: {output_path}")


def _print_results(scores, show_features: bool):
    if not scores:
        typer.echo("未找到匹配的账号")
        return

    summary = format_summary(scores)
    typer.echo(summary)
    typer.echo()

    table = Table(title="流失风险评分排行榜", show_lines=False)
    table.add_column("排名", justify="right", style="cyan", no_wrap=True)
    table.add_column("账号 ID", style="magenta")
    table.add_column("风险分", justify="right", style="yellow")
    table.add_column("风险等级", style="red")
    table.add_column("百分位", justify="right", style="green")

    risk_styles = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
    }

    for i, s in enumerate(scores[:20], 1):
        label = RISK_LEVEL_LABELS.get(s.risk_level, s.risk_level)
        style = risk_styles.get(s.risk_level, "")
        table.add_row(
            str(i),
            s.account_id,
            f"{s.total_score:.2f}",
            f"[{style}]{label}[/{style}]",
            f"{s.risk_percentile:.2f}%",
        )

    console.print(table)

    if len(scores) > 20:
        typer.echo(f"\n... 共 {len(scores)} 个账号，仅显示前 20 名")

    if show_features and scores:
        typer.echo("\n特征权重详情 (前 5 名):")
        feature_table = Table(show_header=True, header_style="bold blue")
        feature_table.add_column("特征", style="cyan")
        for i, s in enumerate(scores[:5], 1):
            feature_table.add_column(f"#{i} {s.account_id}", justify="right")

        feature_names = list(scores[0].feature_scores.keys())
        for feat in feature_names:
            row = [feat]
            for s in scores[:5]:
                row.append(f"{s.feature_scores.get(feat, 0):.2f}")
            feature_table.add_row(*row)

        console.print(feature_table)


if __name__ == "__main__":
    app()

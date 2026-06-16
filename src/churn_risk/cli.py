from pathlib import Path
from typing import List, Optional

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
from .preprocessing import preprocess_pipeline, preprocess_report
from .ml_weights import train_weights_auto, OnlineWeightAdjuster, WeightLearner
from .ltv import (
    calculate_ltv_for_account,
    assign_ltv_tiers,
    generate_action_list,
    calculate_portfolio_summary,
    format_action_list,
    format_portfolio_summary,
)
from .shap_analysis import (
    SHAPAnalyzer,
    format_shap_account_report,
    format_shap_global_report,
)
from .timeseries import (
    analyze_account_timeseries,
    get_account_trend_summary,
    get_portfolio_trend_summary,
    format_timeseries_report,
    format_account_trend_report,
)
from .visualization import generate_all_visualizations
from .data_source import load_from_database, AccountBehaviorDB, StreamProcessor


app = typer.Typer(
    name="churn-score",
    help="客户流失风险评分工具 - 识别高风险流失账号",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def score(
    input_file: str = typer.Argument(..., help="输入数据文件路径 (CSV/JSON) 或 数据库 URL"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    output_csv: Optional[str] = typer.Option(None, "--csv", help="导出 CSV 文件路径"),
    output_json: Optional[str] = typer.Option(None, "--json", help="导出 JSON 文件路径"),
    min_risk: str = typer.Option("low", "--min-risk", "-m", help="最低风险等级筛选 (low/medium/high/critical)"),
    top_n: Optional[int] = typer.Option(None, "--top", "-n", help="仅显示前 N 个高风险账号"),
    show_features: bool = typer.Option(False, "--features", "-f", help="显示各特征详细得分"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式"),
    dry_run: bool = typer.Option(False, "--dry-run", help="试运行模式：仅显示配置和数据概览，不写入文件"),
    preprocess: bool = typer.Option(True, "--preprocess/--no-preprocess", help="启用特征预处理"),
    ml_weights: bool = typer.Option(False, "--ml-weights", help="使用 ML 自动学习权重替代配置权重"),
    ml_model: str = typer.Option("logistic", "--ml-model", help="ML 模型类型: logistic/random_forest"),
    with_ltv: bool = typer.Option(False, "--ltv", help="启用 LTV 业务联动分析"),
    with_shap: bool = typer.Option(False, "--shap", help="启用 SHAP 可解释性分析"),
    shap_account: Optional[List[str]] = typer.Option(None, "--shap-account", help="指定 SHAP 分析的账号ID（可多次指定）"),
    with_timeseries: bool = typer.Option(False, "--timeseries", help="启用时间序列趋势分析"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="生成可视化图表"),
    output_dir: str = typer.Option("output", "--output-dir", help="输出目录"),
    from_db: bool = typer.Option(False, "--from-db", help="从数据库读取数据"),
    db_plan_filter: Optional[str] = typer.Option(None, "--db-plan", help="数据库查询: 按套餐类型筛选"),
    db_limit: Optional[int] = typer.Option(None, "--db-limit", help="数据库查询: 限制返回条数"),
):
    """对账号进行流失风险评分并输出分级清单"""

    if dry_run:
        typer.echo("[yellow]=== DRY-RUN 模式 ===[/yellow]")
        typer.echo(f"输入源: {input_file}")
        typer.echo(f"配置文件: {config_file or '(默认)'}")
        typer.echo(f"预处理: {'启用' if preprocess else '禁用'}")
        typer.echo(f"ML权重: {'启用 (' + ml_model + ')' if ml_weights else '禁用'}")
        typer.echo(f"LTV联动: {'启用' if with_ltv else '禁用'}")
        typer.echo(f"SHAP分析: {'启用' if with_shap else '禁用'}")
        typer.echo(f"时序分析: {'启用' if with_timeseries else '禁用'}")
        typer.echo(f"可视化: {'启用' if visualize else '禁用'}")
        typer.echo(f"输出目录: {output_dir}")
        typer.echo("[yellow]====================[/yellow]\n")

    try:
        config = load_config(config_file)
    except FileNotFoundError as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    try:
        if from_db:
            accounts = load_from_database(
                input_file, plan_filter=db_plan_filter, limit=db_limit
            )
        else:
            accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    if not accounts:
        typer.echo("警告: 未找到任何账号数据", err=True)
        raise typer.Exit(0)

    if dry_run:
        typer.echo(f"加载账号数: {len(accounts)}")
        feat_summary = collect_feature_summary(accounts)
        typer.echo(f"平均登录频次(30d): {feat_summary['avg_login_count_30d']:.2f}")
        typer.echo(f"平均最近登录(天前): {feat_summary['avg_last_login_days']:.2f}")
        typer.echo(f"平均交易频次(30d): {feat_summary['avg_transactions_30d']:.2f}")
        typer.echo("\n[green]✓ DRY-RUN 完成，未写入任何文件[/green]")
        return

    if preprocess:
        preprocess_result = preprocess_pipeline(accounts)
        if not quiet:
            typer.echo(preprocess_report(preprocess_result))
            typer.echo()
        scoring_accounts = preprocess_result["processed_accounts"]
    else:
        scoring_accounts = accounts

    if ml_weights:
        if not quiet:
            typer.echo(f"[cyan]使用 {ml_model} 模型训练权重中...[/cyan]")
        try:
            train_result = train_weights_auto(scoring_accounts, model_type=ml_model)
            config.weights = train_result.weights
            if not quiet:
                typer.echo(f"训练完成 - 准确率: {train_result.accuracy:.4f}, ROC-AUC: {train_result.roc_auc:.4f}")
                typer.echo("学习到的特征权重:")
                for feat, imp in train_result.feature_importance.items():
                    typer.echo(f"  {feat}: {imp:.4f}")
                typer.echo()
        except Exception as e:
            typer.echo(f"[yellow]ML 训练失败，使用默认权重: {e}[/yellow]")

    scores = score_accounts(scoring_accounts, config)

    if min_risk != "low":
        scores = filter_by_risk_level(scores, min_risk)

    if top_n:
        scores = get_top_n(scores, top_n)

    account_map = {a.account_id: a for a in accounts}
    score_accounts_data = [account_map[s.account_id] for s in scores if s.account_id in account_map]

    ltv_list = []
    actions = []
    if with_ltv and score_accounts_data:
        if not quiet:
            typer.echo("[cyan]计算 LTV 与业务联动...[/cyan]")
        ltv_list = [
            calculate_ltv_for_account(a, s)
            for a, s in zip(score_accounts_data, scores)
            if a.account_id == s.account_id
        ]
        ltv_list = assign_ltv_tiers(ltv_list)
        actions = generate_action_list(scores, score_accounts_data, ltv_list)

    shap_global = None
    shap_account_results = []
    if with_shap and scoring_accounts:
        if not quiet:
            typer.echo("[cyan]运行 SHAP 可解释性分析...[/cyan]")
        try:
            analyzer = SHAPAnalyzer()
            shap_global = analyzer.global_analysis(scoring_accounts)

            target_ids = shap_account if shap_account else [s.account_id for s in scores[:3]]
            target_accounts = [a for a in scoring_accounts if a.account_id in target_ids]
            shap_account_results = analyzer.analyze_accounts(scoring_accounts, target_account_ids=target_ids)

            if not quiet:
                typer.echo(format_shap_global_report(shap_global))
                for shap_res in shap_account_results:
                    typer.echo()
                    typer.echo(format_shap_account_report(shap_res))
        except Exception as e:
            typer.echo(f"[yellow]SHAP 分析失败: {e}[/yellow]")

    ts_portfolio_summary = None
    ts_account_summaries = []
    if with_timeseries and accounts:
        if not quiet:
            typer.echo("\n[cyan]时间序列趋势分析...[/cyan]")
        try:
            ts_results = analyze_account_timeseries(accounts)
            ts_portfolio_summary = get_portfolio_trend_summary(ts_results)
            if not quiet:
                typer.echo(format_timeseries_report(ts_portfolio_summary))

            top_risk_ids = [s.account_id for s in scores[:3]]
            for acc_id in top_risk_ids:
                acc_ts = get_account_trend_summary(ts_results, acc_id)
                if acc_ts:
                    ts_account_summaries.append(acc_ts)
                    if not quiet:
                        typer.echo()
                        typer.echo(format_account_trend_report(acc_ts))
        except Exception as e:
            typer.echo(f"[yellow]时序分析失败: {e}[/yellow]")

    if not quiet:
        _print_results(scores, show_features)

        if with_ltv and actions:
            typer.echo()
            typer.echo(format_portfolio_summary(calculate_portfolio_summary(actions)))
            typer.echo()
            typer.echo(format_action_list(actions, max_items=10))

    if visualize:
        if not quiet:
            typer.echo(f"\n[cyan]生成可视化图表至 {output_dir}/figures/...[/cyan]")
        try:
            fig_results = generate_all_visualizations(
                scores=scores,
                actions=actions if with_ltv else None,
                shap_global=shap_global,
                shap_accounts=shap_account_results if with_shap else None,
                timeseries_summaries=ts_account_summaries if with_timeseries else None,
                output_dir=f"{output_dir}/figures",
            )
            if not quiet:
                for name, success in fig_results.items():
                    status = "✓" if success else "✗"
                    typer.echo(f"  {status} {name}")
        except Exception as e:
            typer.echo(f"[yellow]可视化生成失败: {e}[/yellow]")

    if output_csv:
        csv_path = output_csv if output_csv.startswith("/") else f"{output_dir}/{output_csv}"
        export_to_csv(scores, csv_path, include_features=show_features)
        if not quiet:
            typer.echo(f"\n[green]✓[/green] CSV 结果已保存至: {csv_path}")

    if output_json:
        json_path = output_json if output_json.startswith("/") else f"{output_dir}/{output_json}"
        export_to_json(scores, json_path)
        if not quiet:
            typer.echo(f"[green]✓[/green] JSON 结果已保存至: {json_path}")


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


@app.command("preprocess")
def run_preprocess(
    input_file: str = typer.Argument(..., help="输入数据文件路径"),
    missing_strategy: str = typer.Option("median", "--missing", help="缺失值策略: median/mean/zero"),
    outlier_method: str = typer.Option("clip", "--outlier", help="异常值处理: clip/remove/median"),
    normalization: str = typer.Option("minmax", "--norm", help="归一化方式: minmax/zscore/robust/none"),
    output_csv: Optional[str] = typer.Option(None, "--csv", help="导出预处理后数据"),
):
    """特征预处理：缺失值、异常值、归一化"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    result = preprocess_pipeline(
        accounts,
        missing_strategy=missing_strategy,
        outlier_method=outlier_method,
        normalization=normalization if normalization != "none" else "none",
    )

    typer.echo(preprocess_report(result))

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result["normalized_df"].to_csv(output_csv, index=False)
        typer.echo(f"\n✓ 预处理数据已保存至: {output_csv}")


@app.command("train-weights")
def train_weights_command(
    input_file: str = typer.Argument(..., help="训练数据文件路径"),
    model_type: str = typer.Option("logistic", "--model", "-m", help="模型类型: logistic/random_forest"),
    output_config: Optional[str] = typer.Option(None, "--output", "-o", help="输出权重到配置文件"),
):
    """使用 ML 自动从数据中学习特征权重"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    if len(accounts) < 10:
        typer.echo("错误: 至少需要 10 个样本进行训练", err=True)
        raise typer.Exit(1)

    result = train_weights_auto(accounts, model_type=model_type)

    typer.echo("=" * 60)
    typer.echo(f"ML 权重训练结果 ({model_type})")
    typer.echo("=" * 60)
    typer.echo(f"训练样本: {result.training_samples}")
    typer.echo(f"准确率: {result.accuracy:.4f}")
    typer.echo(f"ROC-AUC: {result.roc_auc:.4f}")
    typer.echo("")
    typer.echo(f"{'特征':<25s}  {'重要性':>10s}")
    typer.echo("-" * 40)
    for feat, imp in sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"{feat:<25s}  {imp:10.4f}")
    typer.echo("=" * 60)

    if output_config:
        import yaml
        config_data = {
            "weights": {
                "login_frequency_30d": result.weights.login_frequency_30d,
                "login_trend": result.weights.login_trend,
                "recency": result.weights.recency,
                "transaction_frequency": result.weights.transaction_frequency,
                "transaction_amount": result.weights.transaction_amount,
                "engagement_depth": result.weights.engagement_depth,
                "support_tickets": result.weights.support_tickets,
                "payment_issues": result.weights.payment_issues,
                "email_engagement": result.weights.email_engagement,
            }
        }
        Path(output_config).parent.mkdir(parents=True, exist_ok=True)
        with open(output_config, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)
        typer.echo(f"\n✓ 权重配置已保存至: {output_config}")


@app.command("ltv")
def analyze_ltv(
    input_file: str = typer.Argument(..., help="输入数据文件路径"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="评分配置文件"),
    output_csv: Optional[str] = typer.Option(None, "--csv", help="导出挽留清单 CSV"),
    output_json: Optional[str] = typer.Option(None, "--json", help="导出挽留清单 JSON"),
):
    """LTV 业务联动分析与客户挽留优先级清单"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    config = load_config(config_file)
    scores = score_accounts(accounts, config)

    ltv_list = [
        calculate_ltv_for_account(a, s)
        for a, s in zip(accounts, scores)
    ]
    ltv_list = assign_ltv_tiers(ltv_list)
    actions = generate_action_list(scores, accounts, ltv_list)

    portfolio = calculate_portfolio_summary(actions)
    typer.echo(format_portfolio_summary(portfolio))
    typer.echo()
    typer.echo(format_action_list(actions))

    if output_csv:
        import csv
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            if actions:
                writer = csv.DictWriter(f, fieldnames=list(actions[0].keys()))
                writer.writeheader()
                writer.writerows(actions)
        typer.echo(f"\n✓ 挽留清单已保存至: {output_csv}")

    if output_json:
        import json
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"portfolio": portfolio, "actions": actions}, f, ensure_ascii=False, indent=2)
        typer.echo(f"✓ 挽留分析已保存至: {output_json}")


@app.command("shap")
def analyze_shap(
    input_file: str = typer.Argument(..., help="输入数据文件路径"),
    account_id: Optional[List[str]] = typer.Option(None, "--account", help="指定分析账号 (可多次指定)"),
    global_report: bool = typer.Option(True, "--global/--no-global", help="显示全局特征重要性"),
    output_dir: str = typer.Option("output/shap", "--output", "-o", help="SHAP 报告输出目录"),
):
    """SHAP 模型可解释性分析"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    analyzer = SHAPAnalyzer()

    if global_report:
        result = analyzer.global_analysis(accounts)
        typer.echo(format_shap_global_report(result))

    target_ids = account_id if account_id else [a.account_id for a in accounts[:5]]
    account_results = analyzer.analyze_accounts(accounts, target_account_ids=target_ids)

    for ar in account_results:
        typer.echo()
        typer.echo(format_shap_account_report(ar))


@app.command("timeseries")
def analyze_trends(
    input_file: str = typer.Argument(..., help="输入数据文件路径"),
    account_id: Optional[str] = typer.Option(None, "--account", help="指定单个账号详情分析"),
    periods: int = typer.Option(12, "--periods", help="回溯期数"),
):
    """时间序列趋势分析"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    results = analyze_account_timeseries(accounts, periods=periods)
    summary = get_portfolio_trend_summary(results)
    typer.echo(format_timeseries_report(summary))

    if account_id:
        acc_summary = get_account_trend_summary(results, account_id)
        if acc_summary:
            typer.echo()
            typer.echo(format_account_trend_report(acc_summary))
        else:
            typer.echo(f"\n未找到账号: {account_id}")


@app.command("db-init")
def init_database(
    db_url: str = typer.Argument("sqlite:///./churn_data.db", help="数据库 URL"),
    sample_file: Optional[str] = typer.Option(None, "--sample", help="导入示例数据文件"),
):
    """初始化数据库并可选导入示例数据"""

    db = AccountBehaviorDB(db_url=db_url)
    typer.echo(f"✓ 数据库初始化成功: {db_url}")

    if sample_file:
        try:
            accounts = load_accounts(sample_file)
            count = db.upsert_batch(accounts)
            typer.echo(f"✓ 已导入 {count} 条账号数据")
        except Exception as e:
            typer.echo(f"[yellow]导入数据失败: {e}[/yellow]")

    typer.echo(f"  当前账号总数: {db.get_account_count()}")


@app.command("stream-test")
def stream_test(
    input_file: str = typer.Argument(..., help="测试数据流的源文件"),
    db_url: Optional[str] = typer.Option(None, "--db", help="可选的数据库持久化"),
    batch_size: int = typer.Option(10, "--batch-size", help="批处理大小"),
):
    """测试流式数据处理"""

    try:
        accounts = load_accounts(input_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    db = AccountBehaviorDB(db_url=db_url) if db_url else None
    received = []

    def callback(account):
        received.append(account.account_id)

    typer.echo(f"启动流式处理器 (batch_size={batch_size})...")
    with StreamProcessor(db=db, callback=callback, batch_size=batch_size) as processor:
        for acc in accounts:
            processor.ingest(acc)

        import time
        time.sleep(1)
        stats = processor.stats

    typer.echo(f"处理完成:")
    typer.echo(f"  已处理: {stats['processed']}")
    typer.echo(f"  错误: {stats['errors']}")
    typer.echo(f"  回调接收: {len(received)}")


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

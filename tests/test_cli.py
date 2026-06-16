import pytest
from typer.testing import CliRunner

from churn_risk.cli import app


runner = CliRunner()


class TestCLIDryRun:
    def test_dry_run_mode(self):
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--dry-run",
                "--no-preprocess",
            ],
        )
        assert result.exit_code == 0
        assert "DRY-RUN" in result.stdout

    def test_dry_run_with_all_flags(self):
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--dry-run",
                "--ml-weights",
                "--ltv",
                "--shap",
                "--timeseries",
                "--visualize",
            ],
        )
        assert result.exit_code == 0
        assert "ML权重: 启用" in result.stdout
        assert "LTV联动: 启用" in result.stdout
        assert "SHAP分析: 启用" in result.stdout
        assert "时序分析: 启用" in result.stdout


class TestCLICore:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "score" in result.stdout
        assert "summary" in result.stdout
        assert "init-config" in result.stdout

    def test_score_basic(self):
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--no-preprocess",
                "--quiet",
            ],
        )
        assert result.exit_code == 0

    def test_score_top_n(self):
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--top", "5",
                "--no-preprocess",
                "--quiet",
            ],
        )
        assert result.exit_code == 0

    def test_score_min_risk(self):
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--min-risk", "high",
                "--no-preprocess",
                "--quiet",
            ],
        )
        assert result.exit_code == 0

    def test_summary_command(self):
        result = runner.invoke(app, ["summary", "examples/sample_accounts.csv"])
        assert result.exit_code == 0
        assert "风险等级分布" in result.stdout


class TestCLIInitConfig:
    def test_init_config(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        result = runner.invoke(app, ["init-config", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()

    def test_init_config_no_force(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("existing")
        result = runner.invoke(app, ["init-config", str(config_path)])
        assert result.exit_code != 0

    def test_init_config_force(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("existing")
        result = runner.invoke(app, ["init-config", str(config_path), "--force"])
        assert result.exit_code == 0


class TestCLIOutput:
    def test_csv_output(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--csv", str(csv_path),
                "--no-preprocess",
                "--quiet",
            ],
        )
        assert result.exit_code == 0
        assert csv_path.exists()

    def test_json_output(self, tmp_path):
        json_path = tmp_path / "results.json"
        result = runner.invoke(
            app,
            [
                "score",
                "examples/sample_accounts.csv",
                "--json", str(json_path),
                "--no-preprocess",
                "--quiet",
            ],
        )
        assert result.exit_code == 0
        assert json_path.exists()


class TestCLIPreprocess:
    def test_preprocess_command(self):
        result = runner.invoke(app, ["preprocess", "examples/sample_accounts.csv"])
        assert result.exit_code == 0
        assert "特征预处理报告" in result.stdout

    def test_preprocess_output_csv(self, tmp_path):
        output_csv = tmp_path / "preprocessed.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "examples/sample_accounts.csv",
                "--csv", str(output_csv),
            ],
        )
        assert result.exit_code == 0
        assert output_csv.exists()


class TestCLILTV:
    def test_ltv_command(self):
        result = runner.invoke(app, ["ltv", "examples/sample_accounts.csv"])
        assert result.exit_code == 0
        assert "客户组合风险与价值汇总" in result.stdout


class TestCLITimeseries:
    def test_timeseries_command(self):
        result = runner.invoke(app, ["timeseries", "examples/sample_accounts.csv"])
        assert result.exit_code == 0
        assert "时间序列趋势分析汇总" in result.stdout


class TestCLIDatabase:
    def test_db_init(self, tmp_path):
        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        result = runner.invoke(
            app,
            [
                "db-init",
                db_url,
                "--sample", "examples/sample_accounts.csv",
            ],
        )
        assert result.exit_code == 0
        assert "数据库初始化成功" in result.stdout


class TestCLIStream:
    def test_stream_test(self):
        result = runner.invoke(
            app,
            [
                "stream-test",
                "examples/sample_accounts.csv",
                "--batch-size", "5",
            ],
        )
        assert result.exit_code == 0
        assert "处理完成" in result.stdout

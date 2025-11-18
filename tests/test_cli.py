"""
Integration testing with the CLI
"""
import pytest
from click.testing import CliRunner
from cli.cli import cli


# Fixture
@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance for all CLI tests."""
    return CliRunner()


def test_help(runner):
    """Tests the command-line interface help message."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


# Testing of the predict_cli of the inference group
def test_predict_cli(runner):
    """Tests the command-line interface predict command."""
    result = runner.invoke(cli, ["inference", "predict", "sample.jpg"])
    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_predict_cli_with_custom_classes(runner):
    """Tests predict with custom class names."""
    result = runner.invoke(
        cli,
        ["inference", "predict", "sample.jpg", "--class-names", "cat,dog"]
    )
    assert result.exit_code == 0
    assert "Predicted class:" in result.output


# Testing of the resize_cli of the transform group
def test_resize_cli(runner):
    """Tests the command-line interface resize command."""
    result = runner.invoke(cli, ["transform", "resize", "sample.jpg", "32", "32"])
    assert result.exit_code == 0
    assert "32" in result.output


def test_resize_cli_invalid_width(runner):
    """Tests resize with invalid width."""
    result = runner.invoke(cli, ["transform", "resize", "sample.jpg", "0", "32"])
    assert result.exit_code == 0
    assert "Error" in result.output
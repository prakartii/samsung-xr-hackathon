"""tools/update_numbers.py: final metrics written into the ablation table and README."""

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_spec = importlib.util.spec_from_file_location("update_numbers", _TOOLS / "update_numbers.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules["update_numbers"] = tool
_spec.loader.exec_module(tool)

M = {
    "weighted": 0.31234,
    "coverage_frac": 0.3207,
    "pct_oracle": 0.52654,
    "map": 0.15,
    "f1": 0.2311,
    "label_acc": 0.8511,
}

ABLATION = (
    "Ablation on `report`.\n\n"
    "| variant | weighted | coverage_frac | pct_oracle | map | f1 | label_acc | coverage_any |\n"
    "|---|---|---|---|---|---|---|---|\n"
    "| full pipeline | 0.2985 | 0.3045 | 0.4999 | 0.1492 | 0.2256 | 0.8298 | 0.4163 |\n"
    "| no hand signal | 0.2964 | 0.3045 | 0.4999 | 0.1492 | 0.2173 | 0.8298 | 0.4163 |\n"
)

README = (
    "| | Weighted | Action coverage (40%) | Action mAP (25%) | Highlight F1 (25%) | Label |\n"
    "|---|---|---|---|---|---|\n"
    "| **REWIND** | **0.298** | 0.304 (50.0% of oracle) | 0.149 | 0.226 | 0.830 |\n"
    "| Uniform windows | 0.112 | 0.177 (29.0%) | 0 | 0.167 | 0 |\n"
    "match any standard frame rate. Over all 47, F1 is 0.222 and the weighted total about 0.298.\n"
    "- **85.7% of visual tokens pruned** (highlight seconds vs video seconds).\n"
)


def test_update_ablation_full_row_gets_new_metrics_rounded_to_4():
    out = tool.update_ablation(ABLATION, M)
    # 0.31234 -> 0.3123, 0.52654 -> 0.5265, 0.15 -> 0.1500; coverage_any untouched
    row = "| full pipeline | 0.3123 | 0.3207 | 0.5265 | 0.1500 | 0.2311 | 0.8511 | 0.4163 |"
    assert row in out.splitlines()


def test_update_ablation_other_rows_unchanged():
    out = tool.update_ablation(ABLATION, M)
    assert out.splitlines()[5] == ABLATION.splitlines()[5]
    assert len(out.splitlines()) == len(ABLATION.splitlines())


def test_update_ablation_missing_full_row_raises():
    broken = ABLATION.replace("| full pipeline |", "| full |")
    with pytest.raises(tool.TargetNotFound):
        tool.update_ablation(broken, M)


def test_update_readme_rewind_row_rounded_to_3_and_pct_to_1():
    out = tool.update_readme(README, M)
    # 0.52654 -> 52.7%; 0.31234 -> 0.312; 0.3207 -> 0.321; 0.8511 -> 0.851
    row = "| **REWIND** | **0.312** | 0.321 (52.7% of oracle) | 0.150 | 0.231 | 0.851 |"
    assert out.splitlines()[2] == row


def test_update_readme_without_extras_keeps_all47_and_pruned_lines():
    out = tool.update_readme(README, M)
    assert out.splitlines()[4:] == README.splitlines()[4:]


def test_update_readme_f1_all_recomputes_weighted_all47():
    out = tool.update_readme(README, M, f1_all=0.2271)
    # 0.31234 + 0.25 * (0.2271 - 0.2311) = 0.31234 - 0.001 = 0.31134 -> 0.311
    assert "Over all 47, F1 is 0.227 and the weighted total about 0.311." in out


def test_update_readme_tokens_pruned_bullet():
    out = tool.update_readme(README, M, pruned=0.8612)
    assert "**86.1% of visual tokens pruned**" in out


def test_main_dry_run_writes_nothing(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ablation_report.md").write_text(ABLATION, encoding="utf-8")
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    argv = ["--root", str(tmp_path), "--dry-run", "--weighted", "0.31234",
            "--coverage-frac", "0.3207", "--pct-oracle", "0.52654", "--map", "0.15",
            "--f1", "0.2311", "--label-acc", "0.8511"]  # fmt: skip
    assert tool.main(argv) == 0
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == README
    assert "README.md: 1 line(s) changed" in capsys.readouterr().out


def test_main_writes_both_files(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ablation_report.md").write_text(ABLATION, encoding="utf-8")
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    argv = ["--root", str(tmp_path), "--weighted", "0.31234", "--coverage-frac", "0.3207",
            "--pct-oracle", "0.52654", "--map", "0.15", "--f1", "0.2311",
            "--label-acc", "0.8511"]  # fmt: skip
    assert tool.main(argv) == 0
    assert "**0.312**" in (tmp_path / "README.md").read_text(encoding="utf-8")
    abl = (tmp_path / "results" / "ablation_report.md").read_text(encoding="utf-8")
    assert "| full pipeline | 0.3123 |" in abl


def test_main_missing_target_writes_nothing(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ablation_report.md").write_text(ABLATION, encoding="utf-8")
    bad_readme = README.replace("| **REWIND** |", "| REWIND |")
    (tmp_path / "README.md").write_text(bad_readme, encoding="utf-8")
    argv = ["--root", str(tmp_path), "--weighted", "0.31", "--coverage-frac", "0.32",
            "--pct-oracle", "0.5", "--map", "0.15", "--f1", "0.23",
            "--label-acc", "0.85"]  # fmt: skip
    assert tool.main(argv) == 1
    assert (tmp_path / "results" / "ablation_report.md").read_text(encoding="utf-8") == ABLATION

import subprocess
import sys


def test_run_real_demo_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "run_real_demo.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--provider" in result.stdout


def test_run_real_demo_requires_provenance_flags():
    result = subprocess.run(
        [sys.executable, "run_real_demo.py", "--input", "data/example.csv"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower()


def test_run_real_demo_missing_file_reports_clean_error():
    result = subprocess.run(
        [sys.executable, "run_real_demo.py", "--input", "does_not_exist.csv", "--name", "X", "--provider", "Y", "--source", "Z"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()


def test_run_real_demo_end_to_end_delegates_to_run_demo(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "run_real_demo.py",
            "--input", "data/example.csv",
            "--name", "Test Dataset",
            "--provider", "Test Provider",
            "--source", "https://example.org/test",
            "--no-figures",
        ],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "REAL-DATA PROVENANCE" in result.stdout
    assert "Test Dataset" in result.stdout
    assert "ENERGYOS RESEARCH DEMO" in result.stdout
    assert "AI ENERGY ENGINEER" in result.stdout

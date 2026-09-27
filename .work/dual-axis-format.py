"""Format only this task's explicit files into /output for review and application."""
import pathlib
import subprocess

paths = [
    "src/rosetta_reality/diagnostics/dual_axis.py",
    "src/rosetta_reality/diagnostics/dual_axis_torch.py",
    "src/rosetta_reality/diagnostics/dual_axis_bank.py",
    "src/rosetta_reality/vla/training/dual_axis.py",
    "src/rosetta_reality/vla/training/features.py",
    "src/rosetta_reality/vla/training/plan.py",
    "src/rosetta_reality/eval/dual_axis.py",
    "src/rosetta_reality/eval/reproducibility_capture.py",
    "scripts/prepare_dual_axis.py",
    "tests/test_dual_axis.py",
    "tests/test_dual_axis_runtime.py",
]
for name in paths:
    raw = pathlib.Path(name).read_bytes()
    fixed = subprocess.run(["python", "-m", "ruff", "check", "--fix", "--stdin-filename", name, "-"],
                           input=raw, capture_output=True)
    formatted = subprocess.run(["python", "-m", "ruff", "format", "--stdin-filename", name, "-"],
                               input=fixed.stdout, capture_output=True, check=True)
    target = pathlib.Path("/output/formatted") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(formatted.stdout)
    print(name)

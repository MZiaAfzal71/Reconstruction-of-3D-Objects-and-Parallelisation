
import subprocess
import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "code"


SCRIPTS = [
    "info.py",
    "reconstruct_python_loops_cpu.py",
    "reconstruct_numpy_vec_cpu.py",
    "reconstruct_numba_cpu.py",
    "reconstruct_numba_st_cpu.py",
    "reconstruct_pytorch_cpu.py",
    "precision_continuity_analysis.py",
    "isolated_peak_memory_python.py",
    "isolated_peak_memory_numba.py",
    "isolated_peak_memory_pytorch.py",
]

for script in SCRIPTS:
    print(f"\n===== Running {script} =====\n")

    result = subprocess.run(
        [sys.executable, "-u", SCRIPT_DIR / script],
        check=True
    )

print("\n✅ All scripts completed successfully.")
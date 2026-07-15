
import subprocess
import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "code"


SCRIPTS = [
    "info.py",
    "aggregate_results.py",
    "generate_figures.py",
]

for script in SCRIPTS:
    print(f"\n===== Running {script} =====\n")

    result = subprocess.run(
        [sys.executable, "-u", SCRIPT_DIR / script],
        check=True
    )

print("\n✅ All scripts completed successfully.")
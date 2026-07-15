import subprocess
import sys
from pathlib import Path
from numba import cuda

if not cuda.is_available():
    # Prints the message and shuts down with an exit status code of 1
    sys.exit("Error: Please enable the GPU before running these scripts.")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "code"


SCRIPTS = [
    "info.py",
    "reconstruct_numba_gpu.py",
    "reconstruct_pytorch_gpu.py",
]

for script in SCRIPTS:
    print(f"\n===== Running {script} =====\n")

    result = subprocess.run(
        [sys.executable, "-u", SCRIPT_DIR / script],
        check=True
    )

print("\n✅ All scripts completed successfully.")
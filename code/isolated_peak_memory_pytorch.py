# === Subprocess-isolated CPU peak-memory measurement (PyTorch notebook) ===
# Same rationale as the other two harnesses: ru_maxrss is a monotonic
# high-water mark for the whole process, so 10 runs back-to-back in one
# long-lived process badly undercounts memory after the first large run.
# Each run here executes in a genuinely fresh subprocess instead.
#
# Covers PyTorch CPU only. GPU memory does NOT need this fix: PyTorch's
# torch.cuda.max_memory_allocated() with reset_peak_memory_stats() before
# each run is already correctly isolated per-run (it's not a process-wide
# high-water mark the way ru_maxrss is) -- that's why the existing GPU
# memory numbers in your earlier upload looked sane while the CPU ones
# didn't.

from curves.curves_pytorch import curve_goodman_torch
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_pytorch import *

import subprocess, sys, json, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import cloudpickle

if "COLAB_GPU" in os.environ:
    platform_env = 'Colab'
elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    platform_env = 'Kaggle'
else:
    platform_env = 'Unknown'
    print("Running locally or in another environment")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Target paths
isolated_mem_file_path = PROJECT_ROOT / "results" / "csv files" / f"torch_cpu_memory_isolated_{platform_env}.csv"

# Safely create directories
isolated_mem_file_path.parent.mkdir(parents=True, exist_ok=True)

WORKER_SCRIPT = r'''
import sys, time, json, resource, pickle
import torch
import numpy as np

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--funcs', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--obj', required=True)
    p.add_argument('--n1', type=int, required=True)
    p.add_argument('--n2', type=int, required=True)
    p.add_argument('--dtype', required=True)
    args = p.parse_args()

    # The worker runs from /tmp. Add /code before cloudpickle tries
    # to import data, curves, and surfaces during unpickling.
    if args.project_root not in sys.path:
        sys.path.insert(0, args.project_root)

    import cloudpickle
    with open(args.funcs, 'rb') as f:
        F = cloudpickle.load(f)

    device = torch.device('cpu')
    dt = torch.float32 if args.dtype == 'float32' else torch.float64
    n1, ds = args.n1, args.obj

    # IMPORTANT: surf_pts_inplace_vectorized reads the *global* `n2`, not
    # its `n` parameter -- replicate that exactly here.
    global n2
    n2 = args.n2

    t0 = time.perf_counter()
    I, Z, Null_Hts = F['data_3d_shape'](ds, backend='torch', dtype=dt, device=device)
    tot_pts, seg_pts = F['t_no_pts'](I, n1)
    N = len(seg_pts)
    M = 4
    step = tot_pts // M

    r = [F['curve_goodman_torch'](I[k], seg_pts[k]) for k in range(len(I))]
    r = torch.stack(r).to(device)
    R = F['match_parameters_torch_seq'](r, N, tot_pts, M)
    B_Point, C_Point = F['base_crown_pt'](R, N, tot_pts, M, step)

    if ds == 'apple':
        B, T = Null_Hts[0], Null_Hts[1]; bt = ct = 'n'
    else:
        B, T = F['base_crown_ht'](R, N, tot_pts, M, step, Z, Null_Hts); bt = ct = 'y'

    gR, gz, gRB, gRC, fb, fc = F['surf_tangent'](R, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)

    FR = torch.zeros((N + 1, tot_pts + 1, n2 + 1, 2), dtype=dt, device=device)
    Fz = torch.zeros((N + 1, tot_pts + 1, n2 + 1), dtype=dt, device=device)
    F['surf_pts_inplace_vectorized'](R, N, tot_pts, Z, B_Point, C_Point, B, T,
                                      gRB, gRC, fb, fc, gR, gz, FR, Fz, bt, ct, n2)

    t1 = time.perf_counter()
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({'elapsed_s': t1 - t0, 'peak_mem_mb': peak_kb / 1024.0}))

if __name__ == "__main__":
    main()
'''

NEEDED_FUNCS = ['data_3d_shape', 't_no_pts', 'curve_goodman_torch', 'match_parameters_torch_seq',
                'base_crown_pt', 'base_crown_ht', 'surf_tangent', 'surf_pts_inplace_vectorized']


def build_worker(worker_path, funcs_path):
    ns = globals()
    missing = [n for n in NEEDED_FUNCS if n not in ns]
    if missing:
        raise RuntimeError(f"These functions must be defined in this notebook before running "
                           f"the memory sweep: {missing}")
    funcs = {name: ns[name] for name in NEEDED_FUNCS}
    with open(funcs_path, 'wb') as f:
        cloudpickle.dump(funcs, f)
    with open(worker_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    compile(WORKER_SCRIPT, worker_path, 'exec')


def measure_peak_memory_isolated(worker_path, funcs_path, ds, n1, n2, dtype_str, timeout=600):
    # The worker is located in /tmp, so explicitly expose the directory
    # containing the curves, data, and surfaces packages.
    source_dir = str(Path(__file__).resolve().parent)

    proc = subprocess.run(
        [sys.executable, worker_path, '--funcs', funcs_path, '--project-root', source_dir, '--obj', ds,
         '--n1', str(n1), '--n2', str(n2), '--dtype', dtype_str],
        capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (obj={ds}, n1={n1}, n2={n2}, dtype={dtype_str}):\n"
                           f"{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_isolated_memory_sweep(N1, N2, dtypes, objects, repeats=2,
                              out_path=isolated_mem_file_path):
    worker_path = os.path.join(tempfile.gettempdir(), "torch_mem_worker.py")
    funcs_path = os.path.join(tempfile.gettempdir(), "torch_mem_worker_funcs.pkl")
    build_worker(worker_path, funcs_path)

    rows = []
    for n1, n2 in zip(N1, N2):
        for dt_str in dtypes:
            for ds in objects:
                peaks = []
                for r in range(repeats):
                    result = measure_peak_memory_isolated(worker_path, funcs_path, ds, n1, n2, dt_str)
                    peaks.append(result['peak_mem_mb'])
                    print(f"[PyTorch CPU] {ds} n2={n2} dtype={dt_str} rep={r + 1}/{repeats}: "
                          f"peak_mem={result['peak_mem_mb']:.1f}MB elapsed={result['elapsed_s']:.3f}s")
                rows.append({
                    'implementation': 'PyTorch CPU', 'object': ds, 'n1': n1, 'n2': n2, 'dtype': dt_str,
                    'peak_mem_mb_mean': np.mean(peaks), 'peak_mem_mb_std': np.std(peaks),
                    'peak_mem_mb_all_reps': str(peaks),
                })

    report = pd.DataFrame(rows)
    report.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return report


# --- SMOKE TEST FIRST (recommended): one small config before the full sweep ---
# smoke = run_isolated_memory_sweep(N1=[300], N2=[1200], dtypes=['float32'],
#                                    objects=['banana'], repeats=1,
#                                    out_path="torch_cpu_memory_smoketest.xlsx")
# print(smoke)

# --- Full sweep ---
N1 = [300, 625, 1250]
N2 = [1200, 2500, 5000]
dtypes = ['float32', 'float64']
objects = ['banana', 'apple', 'vase']

print("\n" + "=" * 75)
print("🚀 Subprocess-isolated CPU peak-memory measurement (PyTorch notebook)")
print("=" * 75)
print("☕ Please be patient while computations are in progress...\n")

report = run_isolated_memory_sweep(N1, N2, dtypes, objects)
report
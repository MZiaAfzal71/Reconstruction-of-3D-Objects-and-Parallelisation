# === Subprocess-isolated CPU peak-memory measurement (pure-Python notebook) ===
# Fixes the ru_maxrss-is-monotonic problem: each run below executes in a
# genuinely FRESH subprocess, so its peak RSS reflects only that run, not a
# high-water mark carried over from earlier (possibly larger) iterations in
# the same long-lived process. Covers "Serial Python" and "NumPy-vectorized
# (1 thread)" -- both implementations live in this notebook.
#
# Memory is far more deterministic than timing for a fixed input size, so
# this uses only 2 repeats per config (not 10) to keep the subprocess count
# manageable.
#
# Uses cloudpickle (not inspect.getsource) to ship the actual function
# objects to the worker subprocess: inspect.getsource relies on linecache
# entries that only exist for functions defined in a real IPython/Jupyter
# cell, and is fragile/version-dependent even then. cloudpickle serializes
# by bytecode instead, which works regardless of how the function was
# defined (interactive session, exec, or normal import) and was verified
# to work in a plain, non-notebook Python process during development.

from curves.curves_python_loops import curve_goodman
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_python_loops import *
from surfaces.surfaces_numpy_vec import *

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
isolated_mem_file_path = PROJECT_ROOT / "results" / "csv files" / f"cpu_memory_isolated_{platform_env}.csv"

# Safely create directories
isolated_mem_file_path.parent.mkdir(parents=True, exist_ok=True)


WORKER_SCRIPT = r'''
import sys, time, json, resource, pickle

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--funcs', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--impl', required=True, choices=['serial', 'npvec'])
    p.add_argument('--obj', required=True)
    p.add_argument('--n1', type=int, required=True)
    p.add_argument('--n2', type=int, required=True)
    p.add_argument('--dtype', required=True)
    args = p.parse_args()

    # The worker runs from /tmp. Add /code before cloudpickle tries
    # to import data, curves, and surfaces during unpickling.
    if args.project_root not in sys.path:
        sys.path.insert(0, args.project_root)


    with open(args.funcs, 'rb') as f:
        import cloudpickle
        F = cloudpickle.load(f)

    import numpy as np
    dt = np.float32 if args.dtype == 'float32' else np.float64
    n1, n2, ds = args.n1, args.n2, args.obj

    t0 = time.perf_counter()
    I, Z, Null_Hts = F['data_3d_shape'](ds, dtype=dt)
    tot_pts, seg_pts = F['t_no_pts'](I, n1)
    N = len(seg_pts)
    M = 4
    step = tot_pts // M
    r = np.stack([F['curve_goodman'](I[k], seg_pts[k]) for k in range(len(I))])
    R_mat = F['match_parameters'](r, N, tot_pts, M)

    if args.impl == 'serial':
        B_Point, C_Point = F['base_crown_pt'](R_mat, N, tot_pts, M, step)
    else:
        B_Point, C_Point = F['base_crown_pt_npvec'](R_mat, N, tot_pts, M, step)

    if ds == 'apple':
        B, T = Null_Hts[0], Null_Hts[1]; bt = ct = 'n'
    else:
        B, T = F['base_crown_ht'](R_mat, N, tot_pts, M, step, Z, Null_Hts); bt = ct = 'y'

    if args.impl == 'serial':
        gR, gz, gRB, gRC, fb, fc = F['surf_tangent'](R_mat, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)
        FR, Fz = F['surf_pts'](R_mat, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2)
    else:
        gR, gz, gRB, gRC, fb, fc = F['surf_tangent_npvec'](R_mat, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)
        FR, Fz = F['surf_pts_npvec'](R_mat, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2, dtype=dt)

    t1 = time.perf_counter()
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({'elapsed_s': t1 - t0, 'peak_mem_mb': peak_kb / 1024.0}))

if __name__ == "__main__":
    main()
'''


def build_worker(worker_path, funcs_path):
    needed = ['data_3d_shape', 't_no_pts', 'curve_goodman', 'match_parameters',
              'base_crown_pt', 'base_crown_ht', 'surf_tangent', 'surf_pts',
              'base_crown_pt_npvec', 'surf_tangent_npvec', 'surf_pts_npvec']
    ns = globals()
    missing = [n for n in needed if n not in ns]
    if missing:
        raise RuntimeError(f"These functions must be defined in this notebook before running "
                            f"the memory sweep: {missing}")
    funcs = {name: ns[name] for name in needed}
    with open(funcs_path, 'wb') as f:
        cloudpickle.dump(funcs, f)
    with open(worker_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    compile(WORKER_SCRIPT, worker_path, 'exec')  # sanity check before launching


def measure_peak_memory_isolated(worker_path, funcs_path, impl, ds, n1, n2, dtype_str, timeout=600):
    source_dir = str(Path(__file__).resolve().parent)

    proc = subprocess.run(
        [sys.executable, worker_path, '--funcs', funcs_path,'--project-root', source_dir, '--impl', impl, '--obj', ds,
         '--n1', str(n1), '--n2', str(n2), '--dtype', dtype_str],
        capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (impl={impl}, obj={ds}, n1={n1}, n2={n2}, dtype={dtype_str}):\n"
                            f"{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_isolated_memory_sweep(N1, N2, dtypes, objects, implementations=('serial', 'npvec'),
                               repeats=2, out_path=isolated_mem_file_path):
    worker_path = os.path.join(tempfile.gettempdir(), "mem_worker.py")
    funcs_path = os.path.join(tempfile.gettempdir(), "mem_worker_funcs.pkl")
    build_worker(worker_path, funcs_path)

    rows = []
    for impl in implementations:
        for n1, n2 in zip(N1, N2):
            for dt_str in dtypes:
                for ds in objects:
                    peaks = []
                    for r in range(repeats):
                        result = measure_peak_memory_isolated(worker_path, funcs_path, impl, ds, n1, n2, dt_str)
                        peaks.append(result['peak_mem_mb'])
                        print(f"[{impl}] {ds} n2={n2} dtype={dt_str} rep={r+1}/{repeats}: "
                              f"peak_mem={result['peak_mem_mb']:.1f}MB elapsed={result['elapsed_s']:.3f}s")
                    rows.append({
                        'implementation': 'Serial Python' if impl == 'serial' else 'NumPy-vectorized (1 thread)',
                        'object': ds, 'n1': n1, 'n2': n2, 'dtype': dt_str,
                        'peak_mem_mb_mean': np.mean(peaks), 'peak_mem_mb_std': np.std(peaks),
                        'peak_mem_mb_all_reps': str(peaks),
                    })

    report = pd.DataFrame(rows)
    report.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return report


# --- Run it ---
N1 = [300, 625, 1250]
N2 = [1200, 2500, 5000]
dtypes = ['float32', 'float64']
objects = ['banana', 'apple', 'vase']

print("\n" + "=" * 75)
print("🚀 Subprocess-isolated CPU peak-memory measurement (pure-Python notebook)")
print("🚀 Memory is far more deterministic than timing for a fixed input size, so")
print("🚀 this uses only 2 repeats per config (not 10) to keep the subprocess count")
print("🚀 manageable.")
print("=" * 75)
print("☕ Please be patient while computations are in progress...\n")

report = run_isolated_memory_sweep(N1, N2, dtypes, objects)
report

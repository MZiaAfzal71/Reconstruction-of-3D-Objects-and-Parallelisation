# === Subprocess-isolated CPU peak-memory measurement (Numba notebook) ===
# Same rationale as the pure-Python version: ru_maxrss is a monotonic
# high-water mark for the whole process, so measuring 10 runs back-to-back
# in one long-lived process badly undercounts memory for every run after
# the first large one. Each run here executes in a genuinely fresh
# subprocess instead.
#
# IMPORTANT DIFFERENCE from the pure-Python harness: @njit dispatcher
# objects have their own pickle behavior (by reference to the original
# module+qualname), which would fail the same way plain pickle fails for
# functions defined interactively (no real importable module backs them).
# The robust fix: ship the UNDECORATED Python function via
# dispatcher.py_func (a stable, long-documented part of Numba's public API)
# through cloudpickle, then re-apply @njit with the matching options
# *inside* the worker subprocess, where it JIT-compiles fresh.

from curves.curves_numba import curve_goodman_numba_f32, curve_goodman_numba_f64
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_numba_st import *
from surfaces.surfaces_numba import *

import subprocess, json, tempfile, os
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
isolated_mem_file_path = PROJECT_ROOT / "results" / "csv files" / f"numba_cpu_memory_isolated_{platform_env}.csv"

# Safely create directories
isolated_mem_file_path.parent.mkdir(parents=True, exist_ok=True)

# njit options each function was originally compiled with (MT = multithreaded,
# ST = single-threaded per Step 8's *_st kernels)
NJIT_OPTS = {
    'curve_goodman_numba_f32': dict(parallel=True, fastmath=True),
    'curve_goodman_numba_f64': dict(parallel=True, fastmath=True),
    'base_crown_pt_numba_float': dict(parallel=True, fastmath=True),
    'base_crown_ht_numba_f32': dict(parallel=True, fastmath=True),
    'base_crown_ht_numba_f64': dict(parallel=True, fastmath=True),
    'surf_tangent_numba_f32': dict(parallel=True, fastmath=True),
    'surf_tangent_numba_f64': dict(parallel=True, fastmath=True),
    'surf_pts_numba_f32': dict(parallel=True, fastmath=True),
    'surf_pts_numba_f64': dict(parallel=True, fastmath=True),
    # single-threaded (_st) variants: no parallel=True, matching Step 8
    'base_crown_pt_numba_float_st': dict(fastmath=True),
    'base_crown_ht_numba_f32_st': dict(fastmath=True),
    'base_crown_ht_numba_f64_st': dict(fastmath=True),
    'surf_tangent_numba_f32_st': dict(fastmath=True),
    'surf_tangent_numba_f64_st': dict(fastmath=True),
    'surf_pts_numba_f32_st': dict(fastmath=True),
    'surf_pts_numba_f64_st': dict(fastmath=True),
}
# plain (non-njit) helpers, shipped as-is via cloudpickle (no py_func needed)
PLAIN_FUNCS = ['data_3d_shape', 't_no_pts', 'match_parameters']

WORKER_SCRIPT = r'''
import sys, time, json, resource, pickle
import numpy as np
from numba import njit, set_num_threads

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--funcs', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--njit_opts', required=True)
    p.add_argument('--impl', required=True, choices=['mt', 'st'])
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
        raw = cloudpickle.load(f)   # {name: plain python function (py_func or as-is)}
    with open(args.njit_opts, 'rb') as f:
        opts = cloudpickle.load(f)  # {name: kwargs dict for njit(), or None for plain funcs}

    if args.impl == 'st':
        set_num_threads(1)

    F = {}
    for name, fn in raw.items():
        if name in opts and opts[name] is not None:
            F[name] = njit(**opts[name])(fn)   # re-JIT fresh in this process
        else:
            F[name] = fn

    suffix = '_st' if args.impl == 'st' else ''
    dt = np.float32 if args.dtype == 'float32' else np.float64
    n1, n2, ds = args.n1, args.n2, args.obj

    t0 = time.perf_counter()
    I, Z, Null_Hts = F['data_3d_shape'](ds, dtype=dt)
    tot_pts, seg_pts = F['t_no_pts'](I, n1)
    N = len(seg_pts)
    M = 4
    step = tot_pts // M

    cg = F['curve_goodman_numba_f32'] if args.dtype == 'float32' else F['curve_goodman_numba_f64']
    r = np.stack([cg(I[k], seg_pts[k]) for k in range(len(I))])
    R_mat = F['match_parameters'](r, N, tot_pts, M)
    B_Point, C_Point = F['base_crown_pt_numba_float' + suffix](R_mat, N, tot_pts, M, step)

    if ds == 'apple':
        B, T = Null_Hts[0], Null_Hts[1]; bt = ct = 'n'
    else:
        bch = F['base_crown_ht_numba_f32' + suffix] if args.dtype == 'float32' else F['base_crown_ht_numba_f64' + suffix]
        B, T = bch(R_mat, N, tot_pts, M, step, Z, Null_Hts); bt = ct = 'y'

    st_fn = F['surf_tangent_numba_f32' + suffix] if args.dtype == 'float32' else F['surf_tangent_numba_f64' + suffix]
    gR, gz, gRB, gRC, fb, fc = st_fn(R_mat, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)

    sp_fn = F['surf_pts_numba_f32' + suffix] if args.dtype == 'float32' else F['surf_pts_numba_f64' + suffix]
    FR, Fz = sp_fn(R_mat, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2)

    t1 = time.perf_counter()
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({'elapsed_s': t1 - t0, 'peak_mem_mb': peak_kb / 1024.0}))

if __name__ == "__main__":
    main()
'''


def build_worker(worker_path, funcs_path, opts_path):
    ns = globals()
    needed_njit = list(NJIT_OPTS.keys())
    missing = [n for n in needed_njit + PLAIN_FUNCS if n not in ns]
    if missing:
        raise RuntimeError(f"These must be defined in this notebook before running "
                           f"the memory sweep: {missing}")

    raw = {}
    for name in needed_njit:
        dispatcher = ns[name]
        if not hasattr(dispatcher, 'py_func'):
            raise RuntimeError(f"{name} does not expose .py_func -- is it actually "
                               f"an @njit-decorated function? Cannot proceed safely.")
        raw[name] = dispatcher.py_func
    for name in PLAIN_FUNCS:
        raw[name] = ns[name]

    with open(funcs_path, 'wb') as f:
        cloudpickle.dump(raw, f)
    with open(opts_path, 'wb') as f:
        cloudpickle.dump(NJIT_OPTS, f)
    with open(worker_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    compile(WORKER_SCRIPT, worker_path, 'exec')


def measure_peak_memory_isolated(worker_path, funcs_path, opts_path, impl, ds, n1, n2, dtype_str, timeout=900):
    # The worker is located in /tmp, so explicitly expose the directory
    # containing the curves, data, and surfaces packages.
    source_dir = str(Path(__file__).resolve().parent)

    proc = subprocess.run(
        [sys.executable, worker_path, '--funcs', funcs_path, '--project-root', source_dir, '--njit_opts', opts_path,
         '--impl', impl, '--obj', ds, '--n1', str(n1), '--n2', str(n2), '--dtype', dtype_str],
        capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (impl={impl}, obj={ds}, n1={n1}, n2={n2}, dtype={dtype_str}):\n"
                           f"{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_isolated_memory_sweep(N1, N2, dtypes, objects, implementations=('mt', 'st'),
                              repeats=2, out_path=isolated_mem_file_path):
    worker_path = os.path.join(tempfile.gettempdir(), "numba_mem_worker.py")
    funcs_path = os.path.join(tempfile.gettempdir(), "numba_mem_worker_funcs.pkl")
    opts_path = os.path.join(tempfile.gettempdir(), "numba_mem_worker_opts.pkl")
    build_worker(worker_path, funcs_path, opts_path)

    rows = []
    for impl in implementations:
        for n1, n2 in zip(N1, N2):
            for dt_str in dtypes:
                for ds in objects:
                    peaks = []
                    for r in range(repeats):
                        result = measure_peak_memory_isolated(worker_path, funcs_path, opts_path,
                                                              impl, ds, n1, n2, dt_str)
                        peaks.append(result['peak_mem_mb'])
                        print(f"[{impl}] {ds} n2={n2} dtype={dt_str} rep={r + 1}/{repeats}: "
                              f"peak_mem={result['peak_mem_mb']:.1f}MB elapsed={result['elapsed_s']:.3f}s "
                              f"(includes fresh JIT compile -- ignore for timing)")
                    rows.append({
                        'implementation': 'Numba (multithreaded)' if impl == 'mt' else 'Numba (1 thread)',
                        'object': ds, 'n1': n1, 'n2': n2, 'dtype': dt_str,
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
#                                    out_path="numba_cpu_memory_smoketest.xlsx")
# print(smoke)

# --- Full sweep ---
N1 = [300, 625, 1250]
N2 = [1200, 2500, 5000]
dtypes = ['float32', 'float64']
objects = ['banana', 'apple', 'vase']

print("\n" + "=" * 75)
print("🚀 Subprocess-isolated CPU peak-memory measurement (Numba notebook)")
print("=" * 75)
print("☕ Please be patient while computations are in progress...\n")

report = run_isolated_memory_sweep(N1, N2, dtypes, objects)
report

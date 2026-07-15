"""
Aggregates all benchmark .csv result files into:
  1) A long-format master dataframe (one row per implementation x
     object x n2 x dtype).
  2) A merged summary table (largest N only, all implementations as columns)
     as both .csv
  3) Figures: log-log runtime scaling (Fig. 1), and (if the precision/
     continuity report is present) the precision + continuity figure (Fig. 3).

USAGE:
  Put all the .csv files described below in one folder (RESULTS_DIR below),
  then run this script. It does not require numpy/numba/torch to be
  installed beyond pandas/matplotlib/openpyxl.

EXPECTED FILES per object in
{"banana", "apple", "vase"}):
  Python_CPU_stats_{object}.csv        (serial)
  Python_NPVEC_ST_stats_{object}.csv   (NumPy-vectorized, 1 thread)
  Numba_CPU_ST_stats_{object}.csv      (Numba, 1 thread)
  Numba_CPU_stats_{object}.csv         (Numba, multithreaded)
  Numba_GPU_stats_{object}.csv         (Numba CUDA)
  TensorV_CPU_stats_{object}.csv       (PyTorch CPU)
  TensorV_GPU_stats_{object}.csv       (PyTorch CUDA)

If your actual filenames differ (e.g. "stat" vs "stats", different
capitalization), edit FILE_PATTERNS below -- everything else adapts
automatically.
"""

import os
import glob, sys
import numpy as np
import pandas as pd
from pathlib import Path

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
RESULTS_DIR = PROJECT_ROOT / "results" / "csv files"  # <- change to wherever you put the .csv files


# Target paths
prec_cont_file_path = RESULTS_DIR / f"precision_continuity_report_{platform_env}.csv"
master_file_path = RESULTS_DIR / f"master_long_format_{platform_env}.csv"
merged_table_path = RESULTS_DIR / f"merged_table_{platform_env}.csv"
mem_trans_table_path = RESULTS_DIR / f"memory_transfer_table_{platform_env}.csv"
mem_trans_table_gpu_path = RESULTS_DIR / f"memory_transfer_table_gpu_transfer_vs_kernel_{platform_env}.csv"
merged_table_prec_path = RESULTS_DIR / f"merged_table_with_precision_{platform_env}.csv"
# Safely create directories
master_file_path.parent.mkdir(parents=True, exist_ok=True)
merged_table_path.parent.mkdir(parents=True, exist_ok=True)
mem_trans_table_path.parent.mkdir(parents=True, exist_ok=True)
mem_trans_table_gpu_path.parent.mkdir(parents=True, exist_ok=True)
merged_table_prec_path.parent.mkdir(parents=True, exist_ok=True)

PLATFORMS = ["Colab", "Kaggle"]
OBJECTS = ["banana", "apple", "vase"]
N2_TO_N1 = {1200: 300, 2500: 625, 5000: 1250}  # from the fixed (N1, N2) pairing used in every driver

# implementation label -> (filename prefix, schema kind)
FILE_PATTERNS = {
    "Serial Python":              ("Python_CPU_stats",       "cpu_staged"),
    "NumPy-vectorized (1 thread)": ("Python_NPVEC_ST_stats",  "cpu_staged"),
    "Numba (1 thread)":           ("Numba_CPU_ST_stats",     "cpu_staged"),
    "Numba (multithreaded)":      ("Numba_CPU_stats",        "cpu_staged"),
    "Numba CUDA":                 ("Numba_GPU_stats",        "gpu_numba"),
    "PyTorch CPU":                ("TensorV_CPU_stats",      "gpu_torch"),
    "PyTorch CUDA":               ("TensorV_GPU_stats",      "gpu_torch"),
}


def _find_file(prefix, platform, obj):
    # tolerant match: allow "stats"/"stat", case variants
    candidates = glob.glob(os.path.join(RESULTS_DIR, f"{prefix}*{obj}*{platform}*.csv"))
    if not candidates:
        candidates = glob.glob(os.path.join(RESULTS_DIR, f"*{prefix}*{obj}*{platform}*.csv"))
    return candidates[0] if candidates else None


def _load_cpu_staged(path, impl, platform, obj):
    df = pd.read_csv(path)
    required = {'Start time', 'Surface ET/End Time', 'n2', 'dtype'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}. "
                          f"Found: {list(df.columns)}")
    df = df.copy()
    df['total_s'] = df['Surface ET/End Time'] - df['Start time']
    df['kernel_s'] = np.nan       # not separable from total for CPU-staged schema
    df['transfer_s'] = 0.0        # no device transfer for CPU-only implementations
    df['n1'] = df['n2'].map(N2_TO_N1)
    df['peak_mem_mb'] = df.get('peak_mem_mb', np.nan)
    return _summarize(df, impl, platform, obj)


def _load_gpu_numba(path, impl, platform, obj):
    df = pd.read_csv(path)
    required = {'t_total', 'n2', 'dtype', 't_kernel_surf_pts', 't_h2d_transfer', 't_d2h_transfer'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}. "
                          f"Found: {list(df.columns)}")
    df = df.copy()
    df['total_s'] = df['t_total']
    df['kernel_s'] = df['t_kernel_surf_pts']
    df['transfer_s'] = df['t_h2d_transfer'] + df['t_d2h_transfer']
    df['n1'] = df['n2'].map(N2_TO_N1)
    df['peak_mem_mb'] = df.get('peak_gpu_mem_mb', np.nan)
    return _summarize(df, impl, platform, obj)


def _load_gpu_torch(path, impl, platform, obj):
    df = pd.read_csv(path)
    required = {'t_total', 'n2', 'dtype', 't_kernel_surf_pts', 't_h2d_transfer', 't_d2h_transfer'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}. "
                          f"Found: {list(df.columns)}")
    df = df.copy()
    df['total_s'] = df['t_total']
    df['kernel_s'] = df['t_kernel_surf_pts']
    df['transfer_s'] = df['t_h2d_transfer'] + df['t_d2h_transfer']
    df['n1'] = df['n2'].map(N2_TO_N1)
    df['peak_mem_mb'] = df.get('peak_mem_mb', np.nan)
    return _summarize(df, impl, platform, obj)


LOADERS = {"cpu_staged": _load_cpu_staged, "gpu_numba": _load_gpu_numba, "gpu_torch": _load_gpu_torch}


def _normalize_dtype(val):
    """Collapses however dtype ended up serialized in the xlsx (raw numpy
    type object -> "<class 'numpy.float32'>", torch dtype -> "torch.float32",
    or already a clean string) into a single clean label: 'float32'/'float64'."""
    s = str(val).lower()
    if 'float32' in s or s in ('f32',):
        return 'float32'
    if 'float64' in s or s in ('f64',):
        return 'float64'
    return s


def _summarize(df, impl, platform, obj):
    rows = []
    df = df.copy()
    df['dtype'] = df['dtype'].map(_normalize_dtype)
    for (n2, dtype), g in df.groupby(['n2', 'dtype']):
        rows.append({
            'implementation': impl, 'platform': platform, 'object': obj,
            'n1': N2_TO_N1.get(int(n2), np.nan), 'n2': int(n2), 'dtype': str(dtype),
            'mean_total_s': g['total_s'].mean(), 'std_total_s': g['total_s'].std(),
            'mean_kernel_s': g['kernel_s'].mean(), 'mean_transfer_s': g['transfer_s'].mean(),
            'mean_peak_mem_mb': g['peak_mem_mb'].mean(), 'n_runs': len(g),
        })
    return pd.DataFrame(rows)


def build_master():
    all_rows = []
    missing_files = []
    for impl, (prefix, kind) in FILE_PATTERNS.items():
        for platform in PLATFORMS:
            for obj in OBJECTS:
                path = _find_file(prefix, platform, obj)
                if path is None:
                    missing_files.append((impl, platform, obj))
                    continue
                all_rows.append(LOADERS[kind](path, impl, platform, obj))
    if missing_files:
        print(f"WARNING: {len(missing_files)} expected files not found (skipped):")
        for m in missing_files:
            print("  ", m)
    if not all_rows:
        raise RuntimeError(f"No result files found in {RESULTS_DIR}. "
                            f"Check RESULTS_DIR and FILE_PATTERNS.")
    master = pd.concat(all_rows, ignore_index=True)
    return master


def build_merged_table(master, n2_target=5000, out_prefix=merged_table_path):
    """Largest-N-only summary table: one row per (object, dtype), columns =
    mean_total_s per implementation. This is the combined
    Table 1 + Table 2 replacement."""
    sub = master[master['n2'] == n2_target].copy()
    sub['col'] = sub['implementation'] + ' (' + sub['platform'] + ')'
    wide = sub.pivot_table(index=['object', 'dtype'], columns='col',
                            values='mean_total_s', aggfunc='first')
    wide = wide.reindex(sorted(wide.index), )
    wide.reset_index().to_csv(out_prefix, index=False)
    print(f"Saved: {out_prefix}")

    return wide


def build_memory_transfer_table(master, n2_target=5000, out_prefix1=mem_trans_table_path, out_prefix2=mem_trans_table_gpu_path):
    """Separate compact table for Step 11 (memory) / Step 12 (transfer vs
    kernel) -- only implementations where these are meaningful (GPU rows for
    transfer; all rows for memory)."""
    sub = master[master['n2'] == n2_target].copy()
    mem = sub.pivot_table(index=['object', 'dtype'],
                           columns=['implementation', 'platform'],
                           values='mean_peak_mem_mb', aggfunc='first')
    mem.to_csv(out_prefix1)

    gpu = sub[sub['implementation'].isin(['Numba CUDA', 'PyTorch CUDA'])]
    transfer = gpu.pivot_table(index=['object', 'dtype'],
                                columns=['implementation', 'platform'],
                                values=['mean_kernel_s', 'mean_transfer_s'], aggfunc='first')
    transfer.to_csv(out_prefix2)
    print(f"Saved: {out_prefix1} and {out_prefix2}")
    return mem, transfer


def add_precision_continuity_columns(wide, report_path=prec_cont_file_path,
                                      out_prefix=merged_table_prec_path):
    """Folds the f32-vs-f64 error and tangent-magnitude verification numbers
    (R1 #2 and weakest-aspect #3) into the merged table as extra columns,
    in addition to Fig. 3 -- belt-and-suspenders reporting since these are
    two of the most specific things R1 asked to see quantified."""
    import os
    if not os.path.exists(report_path):
        print(f"Skipping precision/continuity columns: {report_path} not found "
              f"(run the notebook's precision/continuity cell first).")
        return wide
    report = pd.read_csv(report_path)
    report = report.set_index('object')[
        ['FR_max_abs_err', 'FR_max_rel_err', 'tangent_mag_min', 'n_points_near_cusp']
    ]
    report = report.add_prefix('precision_')
    # wide is indexed by (object, dtype); reset, merge on 'object', restore the index
    wide_flat = wide.reset_index()
    merged = wide_flat.merge(report, left_on='object', right_index=True, how='left')
    merged.to_csv(out_prefix, index=False)
    print(f"Saved: {out_prefix}")
    return merged.set_index(['object', 'dtype'])


def merge_isolated_cpu_memory(master, isolated_paths, out_path=master_file_path):
    """Overrides the unreliable in-process ru_maxrss-based mean_peak_mem_mb
    for CPU implementations with the subprocess-isolated measurements
    (see isolated_memory_*.py). GPU rows are untouched -- their memory
    tracking (torch.cuda.max_memory_allocated / cuda memory-info, both
    reset per run) was already correctly isolated and doesn't need this.

    Merges on platform too, so a Colab-only isolated-memory run does not
    get broadcast onto Kaggle's rows. Files from before the harness scripts
    recorded a `platform` column are NOT auto-merged here -- tag them first
    (add a 'platform' column with the correct value) or they'll be skipped
    with a warning, since guessing wrong would silently corrupt the table."""

    import os
    frames = []
    for p in isolated_paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"Note: {p} not found, skipping.")
    if not frames:
        print("No isolated memory files found; master's original (unreliable) "
              "CPU memory numbers are left as-is.")
        return master

    iso = pd.concat(frames, ignore_index=True)
    iso['platform'] = platform_env
    if 'platform' not in iso.columns:
        raise ValueError(
            "One or more isolated-memory files have no 'platform' column. "
            "Add one before merging (e.g. df['platform'] = 'Colab'; "
            "df.to_excel(...)) -- merging without it would silently apply "
            "one platform's numbers to both Colab and Kaggle rows."
        )

    iso = iso.rename(columns={'peak_mem_mb_mean': 'peak_mem_mb_isolated'})
    iso['dtype'] = iso['dtype'].map(_normalize_dtype)

    merged = master.merge(
        iso[['implementation', 'object', 'n2', 'dtype', 'platform', 'peak_mem_mb_isolated']],
        on=['implementation', 'object', 'n2', 'dtype', 'platform'], how='left'
    )
    merged['mean_peak_mem_mb'] = merged['peak_mem_mb_isolated'].combine_first(merged['mean_peak_mem_mb'])
    merged = merged.drop(columns=['peak_mem_mb_isolated'])
    merged.to_csv(out_path, index=False)
    print(f"Saved (memory-corrected): {out_path}")
    return merged


if __name__ == "__main__":
    master = build_master()
    master = merge_isolated_cpu_memory(master, [
        os.path.join(RESULTS_DIR, f"cpu_memory_isolated_{platform_env}.csv"),
        os.path.join(RESULTS_DIR, f"numba_cpu_memory_isolated_{platform_env}.csv"),
        os.path.join(RESULTS_DIR, f"torch_cpu_memory_isolated_{platform_env}.csv"),
    ])
    print("Saved: master_long_format.csv (", len(master), "rows )")

    wide = build_merged_table(master, n2_target=5000)
    print(wide)
    wide = add_precision_continuity_columns(wide)

    mem, transfer = build_memory_transfer_table(master, n2_target=5000)

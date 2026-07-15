# === Precision (float32 vs float64) and C1->C0 continuity analysis ===
# Addresses:
#   - R1 #2 (quantify float32 vs float64 error)
#   - R1 weakest-aspect #3 (verify the resulting surface is actually C1)
#   - Step 3 discussion (non-monotonic bends can locally reduce C1 -> C0)
#
# Two things are measured per object, at a fixed (n1, n2):
#   1) f32-vs-f64 error: the f32 pipeline is run, then compared to the f64
#      pipeline (cast to f64) on the final surface points (FR, Fz) -- max
#      and mean absolute/relative error.
#   2) Continuity indicator: at each interior junction (i, j), the tangent
#      denominator computed inside surf_tangent is the quantity that goes
#      toward zero exactly where the surface locally loses C1 continuity
#      (see Step 3). This cell recomputes that denominator directly (not
#      just gR, gz) and reports, per object, the minimum value and the
#      (i, j) location where it occurs, plus a flag for near-degenerate
#      points below a configurable relative threshold.

from curves.curves_python_loops import curve_goodman
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_numpy_vec import *

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path


if "google.colab" in sys.modules:
    platform_env = 'Colab'
elif "kaggle_secrets" in sys.modules or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
    platform_env = 'Kaggle'
else:
    platform_env = 'Unknown'
    print("Running locally or in another environment")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Target paths
prec_cont_file_path = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_{platform_env}.csv"
prec_cont_tangent_mag_banana = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_tangent_magnitudes_banana_{platform_env}.csv"
prec_cont_tangent_mag_apple = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_tangent_magnitudes_apple_{platform_env}.csv"
prec_cont_tangent_mag_vase = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_tangent_magnitudes_vase_{platform_env}.csv"
# Safely create directories
prec_cont_file_path.parent.mkdir(parents=True, exist_ok=True)
prec_cont_tangent_mag_banana.parent.mkdir(parents=True, exist_ok=True)
prec_cont_tangent_mag_apple.parent.mkdir(parents=True, exist_ok=True)
prec_cont_tangent_mag_vase.parent.mkdir(parents=True, exist_ok=True)

CONTINUITY_REL_THRESHOLD = 1e-3  # flag points where denom < threshold * median(denom)


def compute_tangent_magnitudes(gR, gz, N):
    """The correct C1->C0 risk indicator: the combined tangent-vector
    magnitude ||(gR, gz)|| at each interior junction (i, j). Unlike the
    algebraic denominator inside surf_tangent (which is only exactly zero
    in the fully-degenerate A=B=C case, already special-cased), the
    *numerator* of gR and gz can shrink toward zero on its own when a
    contour bends inward -- the forward/backward segment-direction vectors
    partially cancel -- driving gR and gz toward zero together even while
    the denominator stays healthy. A near-zero combined tangent magnitude
    means the parametrization has a near-stationary point at that (i, j):
    a visible crease/edge rather than a smooth blend, regardless of why
    the numerator shrank. gR has shape (N, M, 2), gz has shape (N, M)."""
    mag = np.sqrt(gR[..., 0] ** 2 + gR[..., 1] ** 2 + gz ** 2)  # (N, M)
    mag[0, :] = np.nan  # base row: different formula, not an interior junction
    mag[N - 1, :] = np.nan  # crown row: same
    return mag


def run_precision_continuity_analysis(objects, n1, n2, out_path=prec_cont_file_path):
    rows = []
    magnitude_records = {}

    for ds in objects:
        # --- f64 pipeline ---
        I64, Z64, NH64 = data_3d_shape(ds, dtype=np.float64)
        tot_pts, seg_pts = t_no_pts(I64, n1)
        N = len(seg_pts);
        M = 4;
        step = tot_pts // M
        r64 = np.stack([curve_goodman(I64[k], seg_pts[k]) for k in range(len(I64))])
        R64 = match_parameters(r64, N, tot_pts, M)
        B64, C64 = base_crown_pt_npvec(R64, N, tot_pts, M, step)
        if ds == 'apple':
            Bh64, T64 = NH64[0], NH64[1];
            bt = ct = 'n'
        else:
            Bh64, T64 = base_crown_ht(R64, N, tot_pts, M, step, Z64, NH64);
            bt = ct = 'y'
        gR64, gz64, gRB64, gRC64, fb64, fc64 = surf_tangent_npvec(
            R64, N, tot_pts, Z64, NH64, B64, C64, Bh64, T64, bt, ct)
        FR64, Fz64 = surf_pts_npvec(R64, N, tot_pts, Z64, B64, C64, Bh64, T64,
                                    gRB64, gRC64, fb64, fc64, gR64, gz64, bt, ct, n2,
                                    dtype=np.float64)

        # --- f32 pipeline ---
        I32, Z32, NH32 = data_3d_shape(ds, dtype=np.float32)
        r32 = np.stack([curve_goodman(I32[k], seg_pts[k]) for k in range(len(I32))]).astype(np.float32)
        R32 = match_parameters(r32, N, tot_pts, M).astype(np.float32)
        B32, C32 = base_crown_pt_npvec(R32, N, tot_pts, M, step)
        if ds == 'apple':
            Bh32, T32 = np.float32(NH32[0]), np.float32(NH32[1])
        else:
            Bh32, T32 = base_crown_ht(R32, N, tot_pts, M, step, Z32, NH32)
        gR32, gz32, gRB32, gRC32, fb32, fc32 = surf_tangent_npvec(
            R32, N, tot_pts, Z32.astype(np.float32), NH32, B32, C32, Bh32, T32, bt, ct)
        FR32, Fz32 = surf_pts_npvec(R32, N, tot_pts, Z32.astype(np.float32), B32, C32, Bh32, T32,
                                    gRB32, gRC32, fb32, fc32, gR32, gz32, bt, ct, n2,
                                    dtype=np.float32)

        # --- f32 vs f64 error on final surface ---
        FR_err = np.abs(FR32.astype(np.float64) - FR64)
        Fz_err = np.abs(Fz32.astype(np.float64) - Fz64)
        scale = max(np.max(np.abs(FR64)), 1e-12)
        rows.append({
            'object': ds, 'n1': n1, 'n2': n2,
            'FR_max_abs_err': np.max(FR_err), 'FR_mean_abs_err': np.mean(FR_err),
            'FR_max_rel_err': np.max(FR_err) / scale,
            'Fz_max_abs_err': np.max(Fz_err), 'Fz_mean_abs_err': np.mean(Fz_err),
        })

        # --- continuity: combined tangent-vector magnitude (f64 pipeline) ---
        mag = compute_tangent_magnitudes(gR64, gz64, N)  # (N, M), rows 0 and N-1 are NaN
        flat = mag[1:N - 1, :].ravel()
        med = np.median(flat)
        min_val = np.min(flat)
        min_idx = np.unravel_index(np.nanargmin(mag), mag.shape)
        n_flagged = int(np.sum(flat < CONTINUITY_REL_THRESHOLD * med))
        magnitude_records[ds] = mag
        rows[-1].update({
            'tangent_mag_median': med,
            'tangent_mag_min': min_val,
            'tangent_mag_min_at_i_j': str(min_idx),
            'n_points_near_cusp': n_flagged,
            'frac_points_near_cusp': n_flagged / flat.size,
        })

        print(f"[{ds}] FR max abs err (f32 vs f64) = {np.max(FR_err):.3e}  "
              f"min tangent magnitude = {min_val:.3e} at (i,j)={min_idx}  "
              f"near-cusp points = {n_flagged}/{flat.size}")

    report = pd.DataFrame(rows)
    report.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save the raw per-junction tangent-magnitude arrays (one CSV per
    # object) -- this is what Fig. 3 (precision/continuity figure) plots:
    # tangent magnitude vs. i, flagging the bend location where it dips
    # toward zero.
    for ds, mag in magnitude_records.items():
        if ds == 'banana':
            pd.DataFrame(mag).to_csv(prec_cont_tangent_mag_banana, index=False)
            print(f"Saved: {prec_cont_tangent_mag_banana} (per-junction tangent-magnitude grid for plotting)")
        elif ds == 'apple':
            pd.DataFrame(mag).to_csv(prec_cont_tangent_mag_apple, index=False)
            print(f"Saved: {prec_cont_tangent_mag_apple} (per-junction tangent-magnitude grid for plotting)")
        else:
            pd.DataFrame(mag).to_csv(prec_cont_tangent_mag_vase, index=False)
            print(f"Saved: {prec_cont_tangent_mag_vase} (per-junction tangent-magnitude grid for plotting)")

    return report, magnitude_records


# --- Run it (n1=1250, n2=5000 matches the largest/most demanding case
# already used elsewhere in the paper) ---

print("\n" + "=" * 75)
print("🚀 Precision (float32 vs float64) and C1->C0 continuity analysis")
print("🚀 n1=900 and n2=3600 for each object: banana, apple, and vase.")
print("=" * 75)

report, magnitude_records = run_precision_continuity_analysis(['banana', 'apple', 'vase'], n1=900, n2=3600)
report
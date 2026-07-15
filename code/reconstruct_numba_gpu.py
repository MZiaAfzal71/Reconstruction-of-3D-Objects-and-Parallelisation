from curves.curves_numba import curve_goodman_numba_f32, curve_goodman_numba_f64
from matplotlib.backends.backend_pdf import PdfPages
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_numba import *
import matplotlib.pyplot as plt
from pathlib import Path
from numba import cuda
import pandas as pd
import numpy as np
import matplotlib
import gc, sys, os
import math
import time

if "google.colab" in sys.modules:
    platform_env = 'Colab'
elif "kaggle_secrets" in sys.modules or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
    platform_env = 'Kaggle'
else:
    platform_env = 'Unknown'
    print("Running locally or in another environment")

matplotlib.use("Agg")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Target paths
banana_path = PROJECT_ROOT / "results" / "csv files" / f"Numba_GPU_stats_banana_{platform_env}.csv"
apple_path = PROJECT_ROOT / "results" / "csv files" / f"Numba_GPU_stats_apple_{platform_env}.csv"
vase_path = PROJECT_ROOT / "results" / "csv files" / f"Numba_GPU_stats_vase_{platform_env}.csv"
fig_path = PROJECT_ROOT / "results" / "pdf fig files" / f"numba_gpu_{platform_env}.pdf"

# Safely create directories
banana_path.parent.mkdir(parents=True, exist_ok=True)
apple_path.parent.mkdir(parents=True, exist_ok=True)
vase_path.parent.mkdir(parents=True, exist_ok=True)
fig_path.parent.mkdir(parents=True, exist_ok=True)


def gpu_mem_used_mb():
    free_b, total_b = cuda.current_context().get_memory_info()
    return (total_b - free_b) / (1024 ** 2)


columns = ['run_idx', 'object', 'n1', 'n2', 'dtype',
           't_curves', 't_h2d_transfer', 't_kernel_surf_pts', 't_d2h_transfer',
           't_total', 'peak_gpu_mem_mb']
numba_stats_banana = pd.DataFrame(columns=columns)
numba_stats_apple = pd.DataFrame(columns=columns)
numba_stats_vase = pd.DataFrame(columns=columns)

dtypes = [np.float64, np.float32]

N1 = [300, 625, 1250]  # The number of points generated horizontally on each countour = 60 * n1
N2 = [1200, 2500, 5000]  # The number of points generated in vertical directions

print("\n" + "=" * 75)
print("🚀 Surface Reconstruction Benchmark (Numba | GPU)")
print("=" * 75)
print("🔬 Purpose  : Performance benchmarking for surface reconstruction")
print("📊 Datasets : Banana, Apple, Vase")
print("🧮 Parameters:")
print(f"   • n1 (horizontal points) : {N1}")
print(f"   • n2 (vertical points)   : {N2}")
print(f"   • Data types             : {[dt.__name__ for dt in dtypes]}")
print("\n⏳ This is a GPU-intensive benchmark.")

WARMUP = 1
R = 10

start_time_loops = time.perf_counter()
for n1, n2 in zip(N1, N2):
    for dt in dtypes:
        for ds in ['banana', 'apple', 'vase']:
            for i in range(-WARMUP, R):
                is_warmup = (i < 0)
                mem_before = gpu_mem_used_mb()

                t0 = time.perf_counter()
                I, Z, Null_Hts = data_3d_shape(ds, dtype=dt)
                Z = np.ascontiguousarray(Z)
                tot_pts, seg_pts = t_no_pts(I, n1)
                N = len(seg_pts)
                M = np.int64(4) if dt == np.float32 else 4
                step = tot_pts // M

                r = []
                if dt == np.float64:
                    for k in range(len(I)):
                        r.append(curve_goodman_numba_f64(I[k], seg_pts[k]))
                else:
                    for k in range(len(I)):
                        r.append(curve_goodman_numba_f32(I[k], seg_pts[k]))
                r = np.stack(r)
                t_curves = time.perf_counter() - t0

                R_mat = match_parameters(r, N, tot_pts, M)
                B_Point, C_Point = base_crown_pt_numba_float(R_mat, N, tot_pts, M, step)
                if ds == 'apple':
                    B = Null_Hts[0];
                    T = Null_Hts[1];
                    bt = ct = 'n'
                else:
                    if dt == np.float64:
                        B, T = base_crown_ht_numba_f64(R_mat, N, tot_pts, M, step, Z, Null_Hts)
                    else:
                        B, T = base_crown_ht_numba_f32(R_mat, N, tot_pts, M, step, Z, Null_Hts)
                    bt = ct = 'y'

                if dt == np.float64:
                    gR, gz, gRB, gRC, fb, fc = surf_tangent_numba_f64(
                        R_mat, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)
                else:
                    gR, gz, gRB, gRC, fb, fc = surf_tangent_numba_f32(
                        R_mat, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)

                # --- H2D transfer (properly synchronized by cuda.to_device being
                # blocking for host->device copies of pinned/pageable memory in the
                # Numba CUDA API; we still bracket + synchronize explicitly for
                # clarity and consistency with the kernel timing below) ---
                t0 = time.perf_counter()
                R_gpu = cuda.to_device(R_mat)
                Z_gpu = cuda.to_device(Z)
                RB_gpu = cuda.to_device(B_Point)
                RC_gpu = cuda.to_device(C_Point)
                gRB_gpu = cuda.to_device(gRB)
                gRC_gpu = cuda.to_device(gRC)
                fb_gpu = cuda.to_device(fb)
                fc_gpu = cuda.to_device(fc)
                gR_gpu = cuda.to_device(gR)
                gz_gpu = cuda.to_device(gz)

                u = np.linspace(0, 1, n2 + 1, dtype=np.float32)
                L0 = 1 - 3 * u ** 2 + 2 * u ** 3
                L1 = 3 * u ** 2 - 2 * u ** 3
                H0 = u - 2 * u ** 2 + u ** 3
                H1 = -u ** 2 + u ** 3
                u_gpu, L0_gpu, L1_gpu, H0_gpu, H1_gpu = (cuda.to_device(x) for x in (u, L0, L1, H0, H1))

                FR_gpu = cuda.device_array((N + 1, tot_pts + 1, n2 + 1, 2), dtype=np.float32)
                Fz_gpu = cuda.device_array((N + 1, tot_pts + 1, n2 + 1), dtype=np.float32)
                cuda.synchronize()
                t_h2d_transfer = time.perf_counter() - t0

                threadsperblock = (4, 8, 8)
                blockspergrid = (
                    -(-(N + 1) // threadsperblock[0]),
                    -(-(tot_pts + 1) // threadsperblock[1]),
                    -(-(n2 + 1) // threadsperblock[2]),
                )

                t0 = time.perf_counter()
                surf_pts_gpu[blockspergrid, threadsperblock](
                    R_gpu, Z_gpu, RB_gpu, RC_gpu, B, T, gRB_gpu, gRC_gpu, fb_gpu, fc_gpu,
                    gR_gpu, gz_gpu, ord(bt), ord(ct), FR_gpu, Fz_gpu,
                    u_gpu, L0_gpu, L1_gpu, H0_gpu, H1_gpu)
                cuda.synchronize()
                t_kernel_surf_pts = time.perf_counter() - t0

                # --- D2H transfer of the final result (was missing entirely before) ---
                t0 = time.perf_counter()
                FR_host = FR_gpu.copy_to_host()
                Fz_host = Fz_gpu.copy_to_host()
                cuda.synchronize()
                t_d2h_transfer = time.perf_counter() - t0

                peak_gpu_mem_mb = gpu_mem_used_mb() - mem_before

                t_total = t_curves + t_h2d_transfer + t_kernel_surf_pts + t_d2h_transfer

                if not is_warmup:
                    row = [i, ds, n1, n2, dt.__name__, t_curves, t_h2d_transfer, t_kernel_surf_pts,
                           t_d2h_transfer, t_total, peak_gpu_mem_mb]
                    if ds == 'banana':
                        numba_stats_banana.loc[len(numba_stats_banana)] = row
                    elif ds == 'apple':
                        numba_stats_apple.loc[len(numba_stats_apple)] = row
                    else:
                        numba_stats_vase.loc[len(numba_stats_vase)] = row
                    print(f"[{ds}] dtype={dt.__name__} n1={n1} n2={n2} run {i + 1}/{R}: "
                          f"total={t_total:.4f}s kernel={t_kernel_surf_pts:.4f}s "
                          f"h2d={t_h2d_transfer:.5f}s d2h={t_d2h_transfer:.5f}s "
                          f"gpu_mem={peak_gpu_mem_mb:.1f}MB")

                del r, R_mat, gR, gz, gRB, gRC, fb, fc, FR_host, Fz_host
                del u_gpu, L0_gpu, L1_gpu, H0_gpu, H1_gpu
                del R_gpu, gR_gpu, gz_gpu, gRB_gpu, gRC_gpu, fb_gpu, fc_gpu, FR_gpu, Fz_gpu

end_time_loops = time.perf_counter()
time_taken = end_time_loops - start_time_loops

numba_stats_banana.to_csv(banana_path, index=False)
numba_stats_apple.to_csv(apple_path, index=False)
numba_stats_vase.to_csv(vase_path, index=False)

print(f"\nSaved: {banana_path}, {apple_path}, and {vase_path}.\n")

print("\n" + "-" * 75)
print("⏱️ Benchmarking Phase Completed")
print("-" * 75)
print(f"🕒 Total benchmarking time: {time_taken / 60:.2f} minutes")
print("📈 Timing data collected for all datasets and parameter combinations.")
print("-" * 75 + "\n")

# Surface reconstruction of all three shapes [apple, banana, vase] using selected parameters to visualize the final 3D shape.
print("\n" + "=" * 75)
print("🎨 Final Surface Reconstruction & Visualization")
print("=" * 75 + "\n")

for ds in ['banana', 'apple', 'vase']:
    n1 = 500  # The number of points generated horizontally on each countour = 60 * n1
    n2 = 2000  # The number of points generated in vertical directions

    dt = np.float64

    I, Z, Null_Hts = data_3d_shape(ds=ds)
    Z = np.ascontiguousarray(Z)

    tot_pts, seg_pts = t_no_pts(I, n1)

    N = len(seg_pts)
    M = 4
    step = tot_pts // M

    r = []
    for k in range(len(I)):
        r.append(curve_goodman_numba_f64(I[k], seg_pts[k]))
    r = np.stack(r)

    R = match_parameters(r, N, tot_pts)
    B_Point, C_Point = base_crown_pt_numba_float(R, N, tot_pts, M, step)

    if ds == 'apple':
        B = Null_Hts[0]
        T = Null_Hts[1]
        bt = ct = 'n'
    else:
        B, T = base_crown_ht_numba_f64(R, N, tot_pts, M, step, Z, Null_Hts)
        bt = ct = 'y'

    gR, gz, gRB, gRC, fb, fc = surf_tangent_numba_f64(R, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)

    R_gpu = cuda.to_device(R)
    Z_gpu = cuda.to_device(Z)
    RB_gpu = cuda.to_device(B_Point)
    RC_gpu = cuda.to_device(C_Point)
    gRB_gpu = cuda.to_device(gRB)
    gRC_gpu = cuda.to_device(gRC)
    fb_gpu = cuda.to_device(fb)
    fc_gpu = cuda.to_device(fc)
    gR_gpu = cuda.to_device(gR)
    gz_gpu = cuda.to_device(gz)

    # Hermite coefficients (reused in all kernels)
    u = np.linspace(0, 1, n2 + 1, dtype=np.float32)
    L0 = 1 - 3 * u ** 2 + 2 * u ** 3
    L1 = 3 * u ** 2 - 2 * u ** 3
    H0 = u - 2 * u ** 2 + u ** 3
    H1 = -u ** 2 + u ** 3
    u_gpu = cuda.to_device(u)
    L0_gpu = cuda.to_device(L0)
    L1_gpu = cuda.to_device(L1)
    H0_gpu = cuda.to_device(H0)
    H1_gpu = cuda.to_device(H1)

    # Output arrays on GPU
    FR_gpu = cuda.device_array((N + 1, tot_pts + 1, n2 + 1, 2), dtype=np.float32)
    Fz_gpu = cuda.device_array((N + 1, tot_pts + 1, n2 + 1), dtype=np.float32)

    # === 3. Define GPU grid/block configuration ===
    threadsperblock = (4, 8, 8)
    blockspergrid_x = math.ceil((N + 1) / threadsperblock[0])
    blockspergrid_y = math.ceil((tot_pts + 1) / threadsperblock[1])
    blockspergrid_z = math.ceil((n2 + 1) / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # === 4. Run the CUDA kernel ===

    surf_pts_gpu[blockspergrid, threadsperblock](
        R_gpu, Z_gpu, RB_gpu, RC_gpu,
        B, T, gRB_gpu, gRC_gpu, fb_gpu, fc_gpu,
        gR_gpu, gz_gpu,
        ord(bt), ord(ct),  # base_circular, crown_circular
        FR_gpu, Fz_gpu,
        u_gpu, L0_gpu, L1_gpu, H0_gpu, H1_gpu
    )

    cuda.synchronize()

    if ds == 'banana':
        # === 5. Copy final results back to CPU ===
        FR_banana = FR_gpu.copy_to_host()
        Fz_banana = Fz_gpu.copy_to_host()
    elif ds == 'apple':
        # === 5. Copy final results back to CPU ===
        FR_apple = FR_gpu.copy_to_host()
        Fz_apple = Fz_gpu.copy_to_host()
    else:
        # === 5. Copy final results back to CPU ===
        FR_vase = FR_gpu.copy_to_host()
        Fz_vase = Fz_gpu.copy_to_host()

with PdfPages(fig_path) as pdf:
    # ================= FIGURE 1 =================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_banana.shape[0]):
        ax.plot_wireframe(
            FR_banana[i, :, :, 0],
            FR_banana[i, :, :, 1],
            Fz_banana[i, :, :],
            rcount=10,
            ccount=3,
            linewidth=0.7,
            edgecolor='k'
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=10, azim=10)
    ax.axis('off')

    fig.suptitle(
        "Figure 1: Banana's reconstruction\n"
        "Interface: Numba (GPU), loop based implementation",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 2 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, 2):
        ax.plot_surface(
            FR_banana[i, :, :, 0],
            FR_banana[i, :, :, 1],
            Fz_banana[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    ax.view_init(elev=-70, azim=70)
    ax.axis('off')

    fig.suptitle(
        "Figure 2: Banana's base\n"
        "Interface: Numba (GPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 3 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(6, 8):
        ax.plot_surface(
            FR_banana[i, :, :, 0],
            FR_banana[i, :, :, 1],
            Fz_banana[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    ax.view_init(elev=40, azim=20)
    ax.axis('off')

    fig.suptitle(
        "Figure 3: Banana's crown\n"
        "Interface: Numba (GPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 4 =================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_apple.shape[0]):
        ax.plot_wireframe(
            FR_apple[i, :, :, 0],
            FR_apple[i, :, :, 1],
            Fz_apple[i, :, :],
            rcount=10,
            ccount=3,
            linewidth=0.7,
            edgecolor='k'
        )

    plt.axis('equal')
    ax.view_init(elev=10, azim=10)
    ax.axis('off')

    fig.suptitle(
        "Figure 4: Apple's reconstruction\n"
        "Interface: Numba (GPU), loop based implementation",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 5 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, 4):
        ax.plot_surface(
            FR_apple[i, :, :, 0],
            FR_apple[i, :, :, 1],
            Fz_apple[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    plt.axis('equal')
    ax.view_init(elev=60, azim=100)
    ax.axis('off')

    fig.suptitle(
        "Figure 5: Apple's base\n"
        "Interface: Numba (GPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 6 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(6, 10):
        ax.plot_surface(
            FR_apple[i, :, :, 0],
            FR_apple[i, :, :, 1],
            Fz_apple[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    plt.axis('equal')
    ax.view_init(elev=-50, azim=100)
    ax.axis('off')

    fig.suptitle(
        "Figure 6: Apple's crown\n"
        "Interface: Numba (GPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 7 =================
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_vase.shape[0] - 1):
        ax.plot_wireframe(
            FR_vase[i, :, :, 0],
            FR_vase[i, :, :, 1],
            Fz_vase[i, :, :],
            rcount=10,
            ccount=3,
            linewidth=0.7,
            edgecolor='k'
        )

    plt.axis('square')
    ax.view_init(elev=30, azim=10)
    ax.axis('off')

    fig.suptitle(
        "Figure 7: Vase's reconstruction\n"
        "Interface: Numba (GPU), loop based implementation",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 8 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, 5):
        ax.plot_surface(
            FR_vase[i, :, :, 0],
            FR_vase[i, :, :, 1],
            Fz_vase[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    plt.axis('square')
    ax.view_init(elev=-60, azim=100)
    ax.axis('off')

    fig.suptitle(
        "Figure 8: Vase's base\n"
        "Interface: Numba (GPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

print("\n" + "=" * 75)
print("📄 Visualization Completed Successfully 🎉")
print("=" * 75 + "\n\n")


"""
Generates the three figures for the revision:
  Fig. 1: log-log runtime-vs-N scaling, one line per implementation,
          small multiples per object (validates the O(Nmn) complexity
          claim empirically -- Step 10).
  Fig. 2: speedup vs. serial baseline, grouped bar chart per object,
          one bar per implementation at the largest N (visualizes the
          existing "Summary" bullet list).
  Fig. 3: precision + continuity -- f32 vs f64 error alongside the
          tangent-vector-magnitude profile, addressing R1 #2 (quantify
          f32/f64 error) and R1 weakest-aspect #3 (verify the resulting
          C1 surface is actually smooth). Report the real finding either
          way -- this is a verification, not a foregone conclusion.

Run this AFTER aggregate_results.py (needs master_long_format.xlsx) and
AFTER the precision/continuity notebook cell (needs
precision_continuity_report.csv and
precision_continuity_report_tangent_magnitudes_banana.csv,
precision_continuity_report_tangent_magnitudes_apple.csv, and
precision_continuity_report_tangent_magnitudes_vase.csv).
"""
import os, sys
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if "COLAB_GPU" in os.environ:
    platform_env = 'Colab'
elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    platform_env = 'Kaggle'
else:
    platform_env = 'Unknown'
    print("Running locally or in another environment")

matplotlib.use("Agg")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Target paths
prec_cont_file_path = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_{platform_env}.csv"
prec_cont_tangent_mag_banana = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_tangent_magnitudes_banana_{platform_env}.csv"
prec_cont_tangent_mag_apple = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_tangent_magnitudes_apple_{platform_env}.csv"
prec_cont_tangent_mag_vase = PROJECT_ROOT / "results" / "csv files" / f"precision_continuity_report_tangent_magnitudes_vase_{platform_env}.csv"

fig_path = PROJECT_ROOT / "results" / "pdf fig files" / f"papers_three_figures_{platform_env}.pdf"
# Safely create directories
fig_path.parent.mkdir(parents=True, exist_ok=True)



MASTER_PATH = PROJECT_ROOT / "results" / "csv files" / f"master_long_format_{platform_env}.csv"
PRECISION_REPORT_PATH = prec_cont_file_path
TANGENT_MAG_PATHS = {
    "banana": prec_cont_tangent_mag_banana,
    "apple": prec_cont_tangent_mag_apple,
    "vase": prec_cont_tangent_mag_vase,
}

IMPLEMENTATION_STYLE = {
    "Serial Python":               dict(color="#888888", marker="o", ls="--"),
    "NumPy-vectorized (1 thread)": dict(color="#1f77b4", marker="s", ls="--"),
    "Numba (1 thread)":            dict(color="#2ca02c", marker="^", ls="--"),
    "Numba (multithreaded)":       dict(color="#2ca02c", marker="^", ls="-"),
    "PyTorch CPU":                 dict(color="#ff7f0e", marker="D", ls="-"),
    "Numba CUDA":                  dict(color="#d62728", marker="v", ls="-"),
    "PyTorch CUDA":                dict(color="#9467bd", marker="P", ls="-"),
}


def fig1_scaling(master_path=MASTER_PATH, platform="Colab", dtype="float32",
                  out_path=fig_path):
    master = pd.read_csv(master_path)
    sub = master[(master['platform'] == platform) & (master['dtype'] == dtype)]

    objects = sorted(sub['object'].unique())
    fig, axes = plt.subplots(1, len(objects), figsize=(5 * len(objects), 4.2), sharey=True)
    if len(objects) == 1:
        axes = [axes]

    for ax, obj in zip(axes, objects):
        og = sub[sub['object'] == obj]
        for impl, style in IMPLEMENTATION_STYLE.items():
            g = og[og['implementation'] == impl].sort_values('n2')
            if g.empty:
                continue
            ax.errorbar(g['n2'], g['mean_total_s'], yerr=g['std_total_s'],
                        label=impl, capsize=2, **style)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$n$ (vertical resolution)')
        ax.set_title(obj)
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('Mean runtime (s)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f'Runtime scaling by implementation ({platform}, {dtype})')
    fig.tight_layout()
    out_path.savefig(fig)
    plt.close(fig)


def fig2_speedup(master_path=MASTER_PATH, platform="Colab", dtype="float32",
                  n2_target=5000, baseline="Serial Python", out_path="fig2_speedup.png"):
    master = pd.read_csv(master_path)
    sub = master[(master['platform'] == platform) & (master['dtype'] == dtype)
                 & (master['n2'] == n2_target)]

    objects = sorted(sub['object'].unique())
    implementations = [k for k in IMPLEMENTATION_STYLE if k != baseline]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(implementations)
    x = np.arange(len(objects))

    for k, impl in enumerate(implementations):
        speedups = []
        for obj in objects:
            base_row = sub[(sub['object'] == obj) & (sub['implementation'] == baseline)]
            impl_row = sub[(sub['object'] == obj) & (sub['implementation'] == impl)]
            if base_row.empty or impl_row.empty:
                speedups.append(np.nan)
                continue
            speedups.append(base_row['mean_total_s'].iloc[0] / impl_row['mean_total_s'].iloc[0])
        color = IMPLEMENTATION_STYLE[impl]['color']
        ax.bar(x + k * width, speedups, width=width, label=impl, color=color)

    ax.set_yscale('log')
    ax.set_xticks(x + width * (len(implementations) - 1) / 2)
    ax.set_xticklabels(objects)
    ax.set_ylabel(f'Speedup vs. {baseline} (log scale)')
    ax.set_title(f'Speedup at N={n2_target} ({platform}, {dtype})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    out_path.savefig(fig)
    plt.close(fig)


def fig3_precision_continuity(report_path=PRECISION_REPORT_PATH,
                               tangent_mag_paths=TANGENT_MAG_PATHS,
                               out_path="fig3_precision_continuity.png"):
    """Restored per R1's explicit requests: #2 (quantify f32 vs f64 error)
    and weakest-aspect #3 (verify the resulting C1 surface is actually
    smooth). Report whatever this actually shows on the real data -- do not
    pre-decide the conclusion. If min tangent magnitude stays comfortably
    bounded away from zero for all three objects, this figure is the
    evidence for a full-C1 claim. If it dips near zero at a bend (e.g. the
    vase), that is itself a legitimate, quantified finding worth stating
    plainly rather than hiding."""
    report = pd.read_csv(report_path)
    mag_sheets = {obj: pd.read_csv(path, index_col=0)
                  for obj, path in tangent_mag_paths.items()}

    objects = list(mag_sheets.keys())
    fig, axes = plt.subplots(2, len(objects), figsize=(5 * len(objects), 7))

    for col, obj in enumerate(objects):
        mag = mag_sheets[obj].values  # shape (N, M)
        with np.errstate(invalid='ignore'):
            min_per_i = np.nanmin(mag, axis=1)
        ax0 = axes[0, col]
        ax0.plot(min_per_i, color='#d62728')
        ax0.axhline(0, color='k', lw=0.5)
        ax0.set_title(f'{obj}: min tangent magnitude per contour')
        ax0.set_xlabel('contour index $i$')
        ax0.set_ylabel('|tangent|' if col == 0 else '')
        row = report[report['object'] == obj]
        if not row.empty:
            ax0.axhline(row['tangent_mag_median'].iloc[0] * 1e-3, color='gray', ls=':',
                        label='near-cusp threshold')
            ax0.legend(fontsize=8)

        ax1 = axes[1, col]
        if not row.empty:
            ax1.bar(['FR max abs err', 'Fz max abs err'],
                    [row['FR_max_abs_err'].iloc[0], row['Fz_max_abs_err'].iloc[0]],
                    color=['#1f77b4', '#ff7f0e'])
            ax1.set_yscale('log')
            ax1.set_title(f'{obj}: f32 vs f64 error')

    fig.tight_layout()
    out_path.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    import os
    with PdfPages(fig_path) as pdf:
        if os.path.exists(MASTER_PATH):
            fig1_scaling(platform=platform_env, out_path=pdf)
            fig2_speedup(platform=platform_env, out_path=pdf)
        else:
            print(f"Skipping Fig. 1/2: {MASTER_PATH} not found (run aggregate_results.py first).")

        if os.path.exists(PRECISION_REPORT_PATH) and all(os.path.exists(p) for p in TANGENT_MAG_PATHS.values()):
            fig3_precision_continuity(out_path=pdf)
        else:
            print(f"Skipping Fig. 3: run the precision/continuity notebook cell first "
                f"to produce {PRECISION_REPORT_PATH} and {', '.join(TANGENT_MAG_PATHS.values())}.")
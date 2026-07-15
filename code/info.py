import subprocess, platform

print("\n" + "=" * 75)
print("🚀 Diagnostic cell: captures exact hardware/software identity for the")
print("🚀 Experimental Setup section (CPU model name, CPUID family/model number,")
print("🚀 full Linux distribution, NumPy/Numba SIMD dispatch).")
print("=" * 75)

def get_full_env_info():
    """Diagnostic cell: captures exact hardware/software identity for the
    Experimental Setup section (CPU model name, CPUID family/model number,
    full Linux distribution, NumPy/Numba SIMD dispatch). Run once per
    environment (Colab/Kaggle)."""
    info = {}
    # Full CPU identity: read the whole file once, then scan for each field
    # independently (do NOT reuse a single sequential iterator across
    # separate `for line in f` loops -- that consumes lines and silently
    # picks up fields from a *later* logical CPU block instead of the
    # first one, since cpuinfo repeats one block per core).
    try:
        with open('/proc/cpuinfo') as f:
            lines = f.readlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('model name') and 'cpu_model_name' not in info:
                info['cpu_model_name'] = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('cpu family') and 'cpu_family' not in info:
                info['cpu_family'] = stripped.split(':', 1)[1].strip()
            elif (stripped.startswith('model') and not stripped.startswith('model name')
                  and 'cpu_model_number' not in info):
                info['cpu_model_number'] = stripped.split(':', 1)[1].strip()
            if all(k in info for k in ('cpu_model_name', 'cpu_family', 'cpu_model_number')):
                break  # got everything from the first CPU block; stop early
    except FileNotFoundError:
        info['cpu_model_name'] = info['cpu_family'] = info['cpu_model_number'] = 'unavailable'

    # Full Linux distribution (not just kernel version)
    try:
        with open('/etc/os-release') as f:
            os_release = dict(
                l.strip().split('=', 1) for l in f if '=' in l
            )
        info['distro'] = os_release.get('PRETTY_NAME', 'unknown').strip('"')
    except FileNotFoundError:
        info['distro'] = 'unavailable'

    info['kernel'] = platform.release()

    # NumPy SIMD baseline/dispatch actually active on this CPU
    try:
        import numpy as np
        info['numpy_version'] = np.__version__
        cfg = np.show_config(mode='dicts')
        info['numpy_simd'] = cfg.get('SIMD Extensions', 'see np.show_config()')
    except Exception as e:
        info['numpy_simd'] = f'error: {e}'

    # Numba target CPU (confirms host-CPU JIT targeting, e.g. AVX2/AVX-512)
    try:
        import numba
        info['numba_version'] = numba.__version__
        info['numba_target_cpu'] = numba.config.CPU_NAME or 'host (auto-detected)'
        info['numba_target_features'] = numba.config.CPU_FEATURES
    except Exception as e:
        info['numba_target_cpu'] = f'error: {e}'

    return info

for k, v in get_full_env_info().items():
    print(f"{k}: {v}")
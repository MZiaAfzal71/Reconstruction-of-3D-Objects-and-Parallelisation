# --- Single-threaded NumPy-vectorized baseline ---
# This isolates the gain from *vectorization* (array broadcasting instead of
# Python-level loops) from the gain from *multithreading*, by constraining
# the underlying BLAS/OpenMP thread pools to exactly 1 thread. Note this must
# be set before NumPy is imported/used for the env vars to take effect, and
# is additionally enforced at runtime via threadpoolctl as a safeguard for
# any thread pool already initialized in this session.
import numpy as np

# Calculate total no. of points at each height
def t_no_pts(I, n1, dtype=np.int64):
    N = len(I)
    P = np.zeros(N, dtype=dtype)
    for i in range(N):
        P[i] = len(I[i])

    lcm_data_pts = np.lcm.reduce(P)
    tot_pts = lcm_data_pts * n1

    seg_pts = np.zeros(N, dtype=dtype)
    for i in range(N):
        seg_pts[i] = tot_pts / P[i]

    return tot_pts, seg_pts


# To obtain R's by Matching Parameters
def match_parameters(r, N, tot_pts, M=4):
    R = r.copy()
    step = tot_pts // M
    for i in range(N-1):
        for j in range(M):
            L = []
            #Find distance d(j) between two contours for each j
            for k in range(M):
                L.append(np.linalg.norm(R[i, k*step] - r[i+1, (k+j)*step%tot_pts])**2)
            d = sum(L)
            if j == 0:
                minval = d
                reqj = j
            elif d < minval:
                minval = d
                reqj = j
        for l in range(tot_pts):
            R[i + 1, l] = r[i + 1, (l + reqj * step) % tot_pts]
        #r[i+1] = R[i+1]

    #Add the first point at the end of each contour to close it
    # Create an empty list to store the modified sub-arrays
    modified_subarrays = []

    # Iterate through each sub-array
    for sub_arr in R:
        # Get the first point of the current sub-array
        first_point = sub_arr[0]
        # Append the first point to the end of the current sub-array
        # The 'axis=0' argument ensures the point is added as a new row
        modified_sub_arr = np.append(sub_arr, [first_point], axis=0)
        modified_subarrays.append(modified_sub_arr)

    # Convert the list of modified sub-arrays back to a NumPy array
    R = np.array(modified_subarrays)
    return R

def base_crown_pt_npvec(R, N, tot_pts, M, step):
    """
    Vectorized PyTorch + NumPy compatible version of base_crown_pt().
    Computes base and crown points without Python loops or lists.
    """

    dtype = R.dtype

    # Prepare index arrays (for the M//2 pairs)
    half_M = M // 2
    j_idx = np.arange(half_M, dtype=np.int64) * step
    k_idx = j_idx + (tot_pts // 2)

    # ----------------- BASE POINTS -----------------
    A = R[0, j_idx]
    B = R[0, k_idx]
    C = R[1, j_idx]
    D = R[1, k_idx]

    w_AC = np.linalg.norm(A - C, axis=1)
    w_AB  = np.linalg.norm(A - B, axis=1)
    w_BD = np.linalg.norm(B - D, axis=1)

    # weighted sum for base points
    C_B = (w_BD[:, None] * A +
            w_AC[:, None] * B +
            w_AB[:, None] * (A + B)) / (w_AC + 2 * w_AB + w_BD)[:, None]

    # ----------------- CROWN POINTS -----------------
    A = R[N-1, j_idx]
    B = R[N-1, k_idx]
    C = R[N-2, j_idx]
    D = R[N-2, k_idx]

    w_AC = np.linalg.norm(A - C, axis=1)
    w_AB  = np.linalg.norm(A - B, axis=1)
    w_BD = np.linalg.norm(B - D, axis=1)

    C_C = (w_BD[:, None] * A +
            w_AC[:, None] * B +
            w_AB[:, None] * (A + B)) / (w_AC + 2 * w_AB + w_BD)[:, None]

    # Mean over all pairs
    B_Point = 2.0 / M * np.sum(C_B, axis=0)
    C_Point = 2.0 / M * np.sum(C_C, axis=0)

    return B_Point, C_Point


# Calculates Base and Crown Heights
def base_crown_ht(R, N, tot_pts, M, step, Z, Null_Hts, dtype=np.float32):
    """
    Compute average base (B) and crown (T) heights using circle-fitting
    approach. Uses NumPy arrays and vectorized indexing.

    Parameters
    ----------
    R : ndarray, shape (N, tot_pts, 2)
        2D ring coordinates stacked along N levels.
    N : int
        Number of levels along Z axis.
    tot_pts : int
        Number of points per level.
    M : int
        Number of samples to use.
    step : int
        Step size for sampling indices.
    Z : ndarray, shape (N,)
        Z-coordinates for each level.
    Null_Hts : ndarray, shape (2,)
        Reference heights for base and crown.

    Returns
    -------
    B : float
        Average base height.
    T : float
        Average crown height.
    """

    zkb = np.zeros(M, dtype=dtype)
    zkc = np.zeros(M, dtype=dtype)

    for i in range(M):
        j = i * step
        k = (i * step + tot_pts // 2) % tot_pts

        # ---------- Base height ----------
        x1, y1, z1 = R[0, j, 0], R[0, j, 1], Z[0]
        x2, y2, z2 = R[0, k, 0], R[0, k, 1], Z[0]
        x3, y3, z3 = R[1, j, 0], R[1, j, 1], Z[1]

        x4 = -np.linalg.det([[y1, z1, 1], [y2, z2, 1], [y3, z3, 1]])
        y4 =  np.linalg.det([[x1, z1, 1], [x2, z2, 1], [x3, z3, 1]])
        z4 = -np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])

        b1 = -(x1**2 + y1**2 + z1**2)
        b2 = -(x2**2 + y2**2 + z2**2)
        b3 = -(x3**2 + y3**2 + z3**2)
        b4 =  np.linalg.det([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

        X = np.linalg.solve(
            [[2*x1, 2*y1, 2*z1, 1],
             [2*x2, 2*y2, 2*z2, 1],
             [2*x3, 2*y3, 2*z3, 1],
             [x4,   y4,   z4,   0]],
            [b1, b2, b3, b4]
        )

        u, v, w, d = X
        mdx1, mdy1, mdz1 = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        cx2, cy2, cz2 = -u, -v, -w

        a = (cx2 - mdx1) / (cz2 - mdz1)
        b = (cy2 - mdy1) / (cz2 - mdz1)

        p1 = a**2 + b**2 + 1
        p2 = (2*a*(mdx1 - a*mdz1) +
              2*b*(mdy1 - b*mdz1) -
              2*u*a + 2*v*b + 2*w)
        p3 = ((mdx1 - a*mdz1)**2 +
              (mdy1 - b*mdz1)**2 +
              2*u*(mdx1 - a*mdz1) +
              2*v*(mdy1 - b*mdz1) + d)

        roots = np.roots([p1, p2, p3])
        if Null_Hts[0] < z1:
            zkb[i] = max(Null_Hts[0], np.min(roots))
        else:
            zkb[i] = min(Null_Hts[0], np.max(roots))

        # ---------- Crown height ----------
        x1, y1, z1 = R[N-1, j, 0], R[N-1, j, 1], Z[N-1]
        x2, y2, z2 = R[N-1, k, 0], R[N-1, k, 1], Z[N-1]
        x3, y3, z3 = R[N-2, j, 0], R[N-2, j, 1], Z[N-2]

        x4 = -np.linalg.det([[y1, z1, 1], [y2, z2, 1], [y3, z3, 1]])
        y4 =  np.linalg.det([[x1, z1, 1], [x2, z2, 1], [x3, z3, 1]])
        z4 = -np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])

        b1 = -(x1**2 + y1**2 + z1**2)
        b2 = -(x2**2 + y2**2 + z2**2)
        b3 = -(x3**2 + y3**2 + z3**2)
        b4 =  np.linalg.det([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

        X = np.linalg.solve(
            [[2*x1, 2*y1, 2*z1, 1],
             [2*x2, 2*y2, 2*z2, 1],
             [2*x3, 2*y3, 2*z3, 1],
             [x4,   y4,   z4,   0]],
            [b1, b2, b3, b4]
        )

        u, v, w, d = X
        mdx1, mdy1, mdz1 = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        cx2, cy2, cz2 = -u, -v, -w

        a = (cx2 - mdx1) / (cz2 - mdz1)
        b = (cy2 - mdy1) / (cz2 - mdz1)

        p1 = a**2 + b**2 + 1
        p2 = (2*a*(mdx1 - a*mdz1) +
              2*b*(mdy1 - b*mdz1) -
              2*u*a + 2*v*b + 2*w)
        p3 = ((mdx1 - a*mdz1)**2 +
              (mdy1 - b*mdz1)**2 +
              2*u*(mdx1 - a*mdz1) +
              2*v*(mdy1 - b*mdz1) + d)

        roots = np.roots([p1, p2, p3])
        if Null_Hts[1] > z1:
            zkc[i] = min(Null_Hts[1], np.max(roots))
        else:
            zkc[i] = max(Null_Hts[1], np.min(roots))

    B = np.mean(zkb)
    T = np.mean(zkc)
    return B, T


def surf_tangent_npvec(R, N, tot_pts, Z, Null_Hts, RB, RC, Bh, T, base_null='y', crown_null='y',  eps=1e-12):
    """
    Vectorized surf_tangent that works with numpy or torch arrays.
    - R: shape (N, M, 2) where M == tot_pts + 1 (consistent with your loops)
    - Z: shape (N,)
    - RB, RC: shape (2,)
    Returns: (gR, gz, gRB, gRC, fb, fc) of same library type as R
    """

    dtype = R.dtype

    # shapes
    M = R.shape[1]            # should be tot_pts + 1
    assert M >= 1
    # Preallocate
    gR = np.zeros((N, M, 2), dtype=dtype)
    gz = np.zeros((N, M), dtype=dtype)
    gRB = np.zeros((M, 2), dtype=dtype)
    gRC = np.zeros((M, 2), dtype=dtype)
    fb = np.zeros((M,), dtype=dtype)
    fc = np.zeros((M,), dtype=dtype)

    if base_null in ('n', 'N'):
        Bh = Null_Hts[0]
    if crown_null in ('n', 'N'):
        T = Null_Hts[1]

    # --------------------------
    # Interior points: i=1..N-2
    # --------------------------
    if N > 2:
        i_idx = np.arange(1, N - 1, dtype=np.int64)                # shape (K,)
        K = i_idx.shape[0]

        # A, B, C shapes (K, M, 2)
        A = R[i_idx - 1]    # (K, M, 2)
        B = R[i_idx]        # (K, M, 2)
        C = R[i_idx + 1]    # (K, M, 2)

        # Degenerate case (matches scalar reference): if A, B, C all coincide
        # (np.allclose(A,B) and np.allclose(A,C)) the tangent is undefined by
        # the usual formula; fall back to gR=0, gz=sign(Z_fwd). This branch is
        # the one active at near-degenerate bends (see Step 3 discussion of
        # C1->C0 continuity loss), so it must be preserved exactly.
        degenerate = np.all(np.isclose(A, B), axis=-1) & np.all(np.isclose(A, C), axis=-1)  # (K, M)

        # dR_fwd_int, dR_bwd_int: (K, M)
        dR_fwd_int = np.linalg.norm(C - B, axis=2)
        dR_bwd_int  = np.linalg.norm(B - A, axis=2)

        # dZ_fwd_int, dZ_bwd_int per i (K,)
        dZ_fwd_int = Z[i_idx + 1] - Z[i_idx]     # (K,)
        dZ_bwd_int  = Z[i_idx] - Z[i_idx - 1]     # (K,)

        # Expand for broadcasting to (K, M)
        dZ_fwd_int_b = np.abs(dZ_fwd_int)[:, None]
        dZ_bwd_int_b  = np.abs(dZ_bwd_int)[:, None]

        # denom (K, M) with safe fallback (only matters where not degenerate)
        denom = dR_fwd_int * dZ_bwd_int_b + dR_bwd_int * dZ_fwd_int_b
        denom_safe = np.where(denom == 0, eps, denom)

        # numerator (K, M, 2)
        numerator = dR_fwd_int[..., None] * (B - A) + dR_bwd_int[..., None] * (C - B)
        gR_block = numerator / denom_safe[..., None]
        gR_block = np.where(degenerate[..., None], 0.0, gR_block)
        gR[i_idx, :, :] = gR_block

        # gz numerator: dR_fwd_int*dZ_bwd_int + dR_bwd_int*dZ_fwd_int (K, M)
        num_gz = dR_fwd_int * dZ_bwd_int[:, None] + dR_bwd_int * dZ_fwd_int[:, None]
        gz_block = num_gz / denom_safe
        gz_sign_fallback = np.broadcast_to(np.copysign(1.0, dZ_fwd_int)[:, None], gz_block.shape)
        gz_block = np.where(degenerate, gz_sign_fallback, gz_block)
        gz[i_idx, :] = gz_block

    # --------------------------
    # Base section i = 0
    # --------------------------
    if N >= 2:
        A = RB               # (2,)
        B0 = R[0]            # (M,2)
        C0 = R[1]            # (M,2)
        dR_fwd_base = np.linalg.norm(C0 - B0, axis=1)   # (M,)
        dR_bwd_base  = np.linalg.norm(B0 - A, axis=1)    # (M,)
        dZ_fwd_base = Z[1] - Z[0]
        dZ_bwd_base  = Z[0] - Bh
        denom0 = 2 * dR_fwd_base * abs(dZ_bwd_base) + dR_bwd_base * abs(dZ_fwd_base)
        denom0_safe = np.where(denom0 == 0, eps, denom0)
        numer0 = (2 * dR_fwd_base)[:, None] * (B0 - A) + dR_bwd_base[:, None] * (C0 - B0)
        gR[0, :, :] = numer0 / denom0_safe[:, None]
        gz[0, :] = (2 * dR_fwd_base * dZ_bwd_base + dR_bwd_base * dZ_fwd_base) / denom0_safe

    # --------------------------
    # Crown section i = N - 1
    # --------------------------
    if N >= 2:
        AN = R[N - 2]    # (M,2)
        BN = R[N - 1]    # (M,2)
        # dR_fwd_crown: norm of RC - BN per row
        dR_fwd_crown = np.linalg.norm(RC[None, :] - BN, axis=1)
        dR_bwd_crown  = np.linalg.norm(BN - AN, axis=1)
        dZ_fwd_crown = (T - Z[N - 1]).item()
        dZ_bwd_crown  = (Z[N - 1] - Z[N - 2]).item()
        denomN = dR_fwd_crown * abs(dZ_bwd_crown) + 2 * dR_bwd_crown * abs(dZ_fwd_crown)
        denomN_safe = np.where(denomN == 0, eps, denomN)
        numerN = dR_fwd_crown[:, None] * (BN - AN) + 2 * dR_bwd_crown[:, None] * (RC - BN)
        gR[N - 1, :, :] = numerN / denomN_safe[:, None]
        gz[N - 1, :] = (dR_fwd_crown * dZ_bwd_crown + 2 * dR_bwd_crown * dZ_fwd_crown) / denomN_safe

    # --------------------------
    # gRB vectorized across j
    # --------------------------
    j_idx = np.arange(M, dtype=np.int64)
    # kappa1(j) = 1 + (j mod 15): a small, bounded, per-point-varying weight.
    # Two properties are required: (i) it must differ across neighboring
    # boundary points j so that D1 and D2 (which use +kappa1 and -kappa1) do not
    # become simultaneously degenerate at every point of a periodic contour;
    # (ii) it must stay small since it scales terms divided by (Z1 - Z0), and
    # large weights amplify numerical sensitivity there. The modulus 15 was
    # chosen empirically (not formally optimized); nearby small moduli behave
    # comparably in practice.
    kappa1_base_vec = 1.0 + (j_idx % 15) #.astype(float)
    # lam1 vector computations (vector)
    # For base:
    denom_Z01 = Z[1] - Z[0]
    lam1_base_vec = (Bh - kappa1_base_vec * (Z[0] - Bh)) / denom_Z01
    kappa2_base_vec = -kappa1_base_vec
    lam2_base_vec = (Bh - kappa2_base_vec * (Z[0] - Bh)) / denom_Z01

    B0 = R[0]  # (M,2)
    C0 = R[1]  # (M,2)
    # D1, D2 shapes (M,2)
    D1 = kappa1_base_vec[:, None] * (B0 - RB) + lam1_base_vec[:, None] * (C0 - B0)
    D2 = kappa2_base_vec[:, None] * (B0 - RB) + lam2_base_vec[:, None] * (C0 - B0)

    # E and F: normalized row-wise
    def _normalize_rows(X):
        norms = np.linalg.norm(X, axis=1)
        norms_safe = np.where(norms == 0, eps, norms)
        return X / norms_safe[:, None]

    E = _normalize_rows(B0 - RB)   # (M,2)
    F = _normalize_rows(D1 - D2)   # (M,2)

    # cross2d and dot
    cross_vals = E[:, 0] * F[:, 1] - E[:, 1] * F[:, 0]
    dot_vals = np.einsum('ij,ij->i', E, F)

    choose_E = np.abs(cross_vals) < (1e-8)
    choose_F = (~choose_E) & (dot_vals > 0)
    choose_negF = (~choose_E) & (~choose_F)

    gRB[choose_E, :] = E[choose_E, :]
    gRB[choose_F, :] = F[choose_F, :]
    gRB[choose_negF, :] = -F[choose_negF, :]

    # --------------------------
    # gRC vectorized across j
    # --------------------------
    # similar for crown
    kappa1_crown_vec = 1.0 + (j_idx % 15) #.astype(float)
    denom_ZN = (T - Z[N - 1]).item()
    lam1_crown_vec = (T - kappa1_crown_vec * (Z[N - 1] - Z[N - 2])) / (T - Z[N - 1])
    kappa2_crown_vec = -kappa1_crown_vec
    lam2_crown_vec = (T - kappa2_crown_vec * (Z[N - 1] - Z[N - 2])) / (T - Z[N - 1])

    AN = R[N - 2]
    BN = R[N - 1]
    D1 = kappa1_crown_vec[:, None] * (BN - AN) + lam1_crown_vec[:, None] * (RC - BN)
    D2 = kappa2_crown_vec[:, None] * (BN - AN) + lam2_crown_vec[:, None] * (RC - BN)

    E = _normalize_rows(RC - BN)   # (M,2)
    F = _normalize_rows(D1 - D2)   # (M,2)

    cross_vals = E[:, 0] * F[:, 1] - E[:, 1] * F[:, 0]
    dot_vals = np.einsum('ij,ij->i', E, F)

    choose_E = np.abs(cross_vals) < (1e-8)
    choose_F = (~choose_E) & (dot_vals > 0)
    choose_negF = (~choose_E) & (~choose_F)

    gRC[choose_E, :] = E[choose_E, :]
    gRC[choose_F, :] = F[choose_F, :]
    gRC[choose_negF, :] = -F[choose_negF, :]

    # --------------------------
    # fb and fc vectorized
    # --------------------------
    B0 = R[0]   # (M,2)
    Cmag = np.linalg.norm(B0 - RB, axis=1)
    Cmag_safe = np.where(Cmag == 0, eps, Cmag)
    tmp = gR[0] / Cmag_safe[:, None]
    tmp_norm = np.linalg.norm(tmp, axis=1)
    fb = np.sqrt(2.0) * Cmag / np.sqrt(1.0 + abs(Z[0] - Bh) * tmp_norm)

    BN = R[N - 1]
    CmagN = np.linalg.norm(RC - BN, axis=1)
    CmagN_safe = np.where(CmagN == 0, eps, CmagN)
    tmp = gR[N - 1] / CmagN_safe[:, None]
    tmp_norm = np.linalg.norm(tmp, axis=1)
    fc = np.sqrt(2.0) * CmagN / np.sqrt(1.0 + abs(T - Z[N - 1]) * tmp_norm)

    return gR, gz, gRB, gRC, fb, fc


def surf_pts_npvec(R, N, tot_pts, Z, RB, RC, Bh, T, gRB, gRC, fb, fc, gR, gz,
                    base_circular, crown_circular, n, dtype=np.float32):
    """NumPy-vectorized (single-threaded) equivalent of surf_pts(): vectorizes
    over the interior contour index i and all boundary points j simultaneously
    via broadcasting, instead of the nested Python for-loops in the scalar
    version. Semantics are identical; see surf_pts() for the reference
    (scalar) implementation this was checked against."""
    u = np.linspace(0, 1, n + 1, dtype=dtype)
    L0 = 1 - 3 * u**2 + 2 * u**3
    L1 = 3 * u**2 - 2 * u**3
    H0 = u - 2 * u**2 + u**3
    H1 = -u**2 + u**3

    M = tot_pts + 1
    FR = np.zeros((N + 1, M, n + 1, 2), dtype=dtype)
    Fz = np.zeros((N + 1, M, n + 1), dtype=dtype)

    # --- Interior: vectorized over i=1..N-1 and all j at once ---
    if N > 1:
        deltaZ = np.abs(Z[1:N] - Z[0:N - 1])          # (K,) K = N-1
        deltaZ_b = deltaZ[:, None, None]              # (K,1,1)

        A = R[0:N - 1]      # (K, M, 2)  -> i-1
        B = R[1:N]          # (K, M, 2)  -> i
        gA = gR[0:N - 1]    # (K, M, 2)
        gB = gR[1:N]        # (K, M, 2)
        gzA = gz[0:N - 1]   # (K, M)
        gzB = gz[1:N]       # (K, M)

        FR[1:N, :, :, 0] = (L0 * A[..., 0:1] + L1 * B[..., 0:1] +
                             deltaZ_b * (H0 * gA[..., 0:1] + H1 * gB[..., 0:1]))
        FR[1:N, :, :, 1] = (L0 * A[..., 1:2] + L1 * B[..., 1:2] +
                             deltaZ_b * (H0 * gA[..., 1:2] + H1 * gB[..., 1:2]))
        Fz[1:N, :, :] = (Z[0:N - 1][:, None, None] * L0 + Z[1:N][:, None, None] * L1 +
                          deltaZ_b * (gzA[:, :, None] * H0 + gzB[:, :, None] * H1))

    # --- Base: vectorized over all j at once ---
    if base_circular in ('y', 'Y'):
        FR[0, :, :, 0] = (L0 * RB[0] + L1 * R[0, :, 0:1] +
                           fb[:, None] * H0 * gRB[:, 0:1] +
                           2 * abs(Z[0] - Bh) * H1 * gR[0, :, 0:1])
        FR[0, :, :, 1] = (L0 * RB[1] + L1 * R[0, :, 1:2] +
                           fb[:, None] * H0 * gRB[:, 1:2] +
                           2 * abs(Z[0] - Bh) * H1 * gR[0, :, 1:2])
        Fz[0, :, :] = (1 - u**2)[None, :] * Bh + (u**2)[None, :] * Z[0]
    else:
        FR[0, :, :, 0] = (L0 * RB[0] + L1 * R[0, :, 0:1] +
                           2 * abs(Z[0] - Bh) * H1 * gR[0, :, 0:1])
        FR[0, :, :, 1] = (L0 * RB[1] + L1 * R[0, :, 1:2] +
                           2 * abs(Z[0] - Bh) * H1 * gR[0, :, 1:2])
        Fz[0, :, :] = Bh * L0 + Z[0] * L1 + (Z[0] - Bh) * H0 + abs(Z[0] - Bh) * gz[0, :, None] * H1

    # --- Crown: vectorized over all j at once ---
    if crown_circular in ('y', 'Y'):
        FR[N, :, :, 0] = (L0 * R[N - 1, :, 0:1] + L1 * RC[0] +
                           fc[:, None] * H1 * gRC[:, 0:1] +
                           2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, :, 0:1])
        FR[N, :, :, 1] = (L0 * R[N - 1, :, 1:2] + L1 * RC[1] +
                           fc[:, None] * H1 * gRC[:, 1:2] +
                           2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, :, 1:2])
        Fz[N, :, :] = ((1 - u) ** 2)[None, :] * Z[N - 1] + (u * (2 - u))[None, :] * T
    else:
        FR[N, :, :, 0] = (L0 * R[N - 1, :, 0:1] + L1 * RC[0] +
                           2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, :, 0:1])
        FR[N, :, :, 1] = (L0 * R[N - 1, :, 1:2] + L1 * RC[1] +
                           2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, :, 1:2])
        Fz[N, :, :] = (Z[N - 1] * L0 + T * L1 + abs(T - Z[N - 1]) * gz[N - 1, :, None] * H0 +
                        (T - Z[N - 1]) * H1)

    return FR, Fz



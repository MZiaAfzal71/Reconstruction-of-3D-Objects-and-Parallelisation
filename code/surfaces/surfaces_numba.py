import numpy as np
from numba import cuda, njit, prange


# --- Helper functions (all float32-safe) ---
@njit(fastmath=True)
def cross2d_float(a, b):
    # a, b are length-2 arrays (float32)
    return a[0]*b[1] - a[1]*b[0]


@njit(fastmath=True)
def vec_norm_float(a):
    # Euclidean norm for a length-2 float32 vector
    return np.sqrt(a[0]*a[0] + a[1]*a[1])


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
    R = np.array(modified_subarrays, dtype=r.dtype)
    return R

@njit(parallel=True, fastmath=True)
def base_crown_pt_numba_float(R, N, tot_pts, M, step):
    dtype = R.dtype
    B_acc = np.zeros(2, dtype=dtype)
    C_acc = np.zeros(2, dtype=dtype)

    count = M // 2  # total iterations

    for i in range(count):
        j = i * step
        k = j + tot_pts // 2

        # -------- Base Point --------
        A = R[0, j]
        B = R[0, k]
        C = R[1, j]
        D = R[1, k]
        alpha = np.sqrt(np.sum((A - C) * (A - C)))
        beta  = np.sqrt(np.sum((A - B) * (A - B)))
        gamma = np.sqrt(np.sum((B - D) * (B - D)))
        denom = alpha + 2 * beta + gamma
        B_acc += (gamma * A + alpha * B + beta * (A + B)) / denom

        # -------- Crown Point --------
        A = R[N-1, j]
        B = R[N-1, k]
        C = R[N-2, j]
        D = R[N-2, k]
        alpha = np.sqrt(np.sum((A - C) * (A - C)))
        beta  = np.sqrt(np.sum((A - B) * (A - B)))
        gamma = np.sqrt(np.sum((B - D) * (B - D)))
        denom = alpha + 2 * beta + gamma
        C_acc += (gamma * A + alpha * B + beta * (A + B)) / denom

    scale = 2.0 / M
    B_Point = scale * B_acc
    C_Point = scale * C_acc
    return B_Point, C_Point

# ---------------------
# small helpers
# ---------------------
@njit(inline='always', fastmath=True)
def det3(a11, a12, a13,
         a21, a22, a23,
         a31, a32, a33):
    """
    Determinant of a 3x3 matrix given entries (float32).
    """
    return (a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31))

# float32 safe quadratic solver
@njit(inline='always', fastmath=True)
def safe_quad_roots_f32(p1, p2, p3):
    """
    Solve p1*z^2 + p2*z + p3 = 0 for real roots.
    If discriminant < 0, returns two identical fallback roots (NaN-safe).
    Returns (rmin, rmax) such that rmin <= rmax.
    """
    dt = np.float32

    # p1 = dtype(p1)
    # p2 = dtype(p2)
    # p3 = dtype(p3)
    zero = dt(0.0)
    two = dt(2.0)
    four = dt(4.0)
    eps = dt(1e-8)

    if p1 == zero:
        # linear case p2*z + p3 = 0 -> z = -p3/p2 (if p2 != 0)
        if p2 == zero:
            return (p3, p3)  # degenerate; return p3 as fallback
        r = -p3 / p2
        return (r, r)
    disc = p2 * p2 - four * p1 * p3
    if disc < zero:
        # no real roots; return a fallback pair (use -p2/(2*p1))
        r = -p2 / (two * p1)
        return (r, r)
    sd = np.sqrt(disc)
    r1 = (-p2 - sd) / (two * p1)
    r2 = (-p2 + sd) / (two * p1)
    if r1 <= r2:
        return (r1, r2)
    else:
        return (r2, r1)

# float64 safe quadratic solver
@njit(inline='always', fastmath=True)
def safe_quad_roots_f64(p1, p2, p3):
    """
    Solve p1*z^2 + p2*z + p3 = 0 for real roots.
    If discriminant < 0, returns two identical fallback roots (NaN-safe).
    Returns (rmin, rmax) such that rmin <= rmax.
    """
    dt = np.float64

    zero = dt(0.0)
    two = dt(2.0)
    four = dt(4.0)
    eps = dt(1e-8)

    if p1 == zero:
        # linear case p2*z + p3 = 0 -> z = -p3/p2 (if p2 != 0)
        if p2 == zero:
            return (p3, p3)  # degenerate; return p3 as fallback
        r = -p3 / p2
        return (r, r)
    disc = p2 * p2 - four * p1 * p3
    if disc < zero:
        # no real roots; return a fallback pair (use -p2/(2*p1))
        r = -p2 / (two * p1)
        return (r, r)
    sd = np.sqrt(disc)
    r1 = (-p2 - sd) / (two * p1)
    r2 = (-p2 + sd) / (two * p1)
    if r1 <= r2:
        return (r1, r2)
    else:
        return (r2, r1)

# ---------------------
# float32 specialized main function
# ---------------------
@njit(parallel=True, fastmath=True)
def base_crown_ht_numba_f32(R, N, tot_pts, M, step, Z, Null_Hts):
    """
    Float32/64-safe Numba implementation to compute average base (B) and crown (T) heights
    using a circle/sphere-fitting-like approach from three sample points per sample.

    Inputs:
      R : float32/64 ndarray shape (N, tot_pts, 2)
      N, tot_pts, M, step : ints
      Z : float32/64 ndarray shape (N,)
      Null_Hts : float32 ndarray shape (2,)  # [Null_base, Null_crown]

    Returns:
      B, T : float32/float64 scalars (average base and crown heights)
    """
    dt = np.float32

    zero = dt(0.0)
    one = dt(1.0)
    two = dt(2.0)
    half = dt(0.5)

    zkb = np.zeros(M, dtype=dt)
    zkc = np.zeros(M, dtype=dt)

    for i in prange(M):
        j = i * step
        k = (i * step + tot_pts // 2) % tot_pts

        # ---------- Base height (using points on levels 0 and 1) ----------
        x1 = R[0, j, 0]; y1 = R[0, j, 1]; z1 = Z[0]
        x2 = R[0, k, 0]; y2 = R[0, k, 1]; z2 = Z[0]
        x3 = R[1, j, 0]; y3 = R[1, j, 1]; z3 = Z[1]

        # 3x3 determinants for x4,y4,z4 (float32)
        x4 = -det3(y1, z1, one,
                   y2, z2, one,
                   y3, z3, one)
        y4 =  det3(x1, z1, one,
                  x2, z2, one,
                  x3, z3, one)
        z4 = -det3(x1, y1, one,
                   x2, y2, one,
                   x3, y3, one)

        b1 = -(x1*x1 + y1*y1 + z1*z1)
        b2 = -(x2*x2 + y2*y2 + z2*z2)
        b3 = -(x3*x3 + y3*y3 + z3*z3)
        b4 = det3(x1, y1, z1,
                  x2, y2, z2,
                  x3, y3, z3)

        # Build 4x4 system A X = B
        A = np.empty((4,4), dtype=dt)
        Bvec = np.empty(4, dtype=dt)

        A[0,0] = two * x1; A[0,1] = two * y1; A[0,2] = two * z1; A[0,3] = one
        A[1,0] = two * x2; A[1,1] = two * y2; A[1,2] = two * z2; A[1,3] = one
        A[2,0] = two * x3; A[2,1] = two * y3; A[2,2] = two * z3; A[2,3] = one
        A[3,0] = x4;       A[3,1] = y4;       A[3,2] = z4;       A[3,3] = zero

        Bvec[0] = b1; Bvec[1] = b2; Bvec[2] = b3; Bvec[3] = b4

        # Solve for X = [u,v,w,d]
        # np.linalg.solve is supported in Numba for fixed-size arrays of floats
        X = np.linalg.solve(A, Bvec)  # returns float32/float64 array depending on the input types
        u = X[0]; v = X[1]; w = X[2]; d = X[3]

        mdx1 = (x1 + x2) * half
        mdy1 = (y1 + y2) * half
        mdz1 = (z1 + z2) * half
        cx2 = -u; cy2 = -v; cz2 = -w

        # avoid division by zero for cz2 - mdz1
        denom_cz = cz2 - mdz1
        if denom_cz == zero:
            a = zero
            b = zero
        else:
            a = (cx2 - mdx1) / denom_cz
            b = (cy2 - mdy1) / denom_cz

        p1 = a*a + b*b + one
        p2 = (two*a*(mdx1 - a*mdz1) +
              two*b*(mdy1 - b*mdz1) -
              two*u*a + two*v*b + two*w)
        p3 = ((mdx1 - a*mdz1)*(mdx1 - a*mdz1) +
              (mdy1 - b*mdz1)*(mdy1 - b*mdz1) +
              two*u*(mdx1 - a*mdz1) +
              two*v*(mdy1 - b*mdz1) + d)

        # Solve quadratic p1*z^2 + p2*z + p3 = 0
        rmin, rmax = safe_quad_roots_f32(p1, p2, p3)

        # Choose solution as original logic
        if Null_Hts[0] < z1:
            # prefer the smaller root, but not below Null_Hts[0]
            # use max(Null_Hts[0], min(roots))
            # ensure comparisons in float32/float64
            if rmin < Null_Hts[0]:
                zkb[i] = Null_Hts[0]
            else:
                zkb[i] = rmin
        else:
            # prefer the larger root but not above Null_Hts[0]
            if rmax > Null_Hts[0]:
                zkb[i] = Null_Hts[0]
            else:
                zkb[i] = rmax

        # ---------- Crown height (top levels) ----------
        x1 = R[N-1, j, 0]; y1 = R[N-1, j, 1]; z1 = Z[N-1]
        x2 = R[N-1, k, 0]; y2 = R[N-1, k, 1]; z2 = Z[N-1]
        x3 = R[N-2, j, 0]; y3 = R[N-2, j, 1]; z3 = Z[N-2]

        x4 = -det3(y1, z1, one,
                   y2, z2, one,
                   y3, z3, one)
        y4 =  det3(x1, z1, one,
                  x2, z2, one,
                  x3, z3, one)
        z4 = -det3(x1, y1, one,
                   x2, y2, one,
                   x3, y3, one)

        b1 = -(x1*x1 + y1*y1 + z1*z1)
        b2 = -(x2*x2 + y2*y2 + z2*z2)
        b3 = -(x3*x3 + y3*y3 + z3*z3)
        b4 = det3(x1, y1, z1,
                  x2, y2, z2,
                  x3, y3, z3)

        A[0,0] = two * x1; A[0,1] = two * y1; A[0,2] = two * z1; A[0,3] = one
        A[1,0] = two * x2; A[1,1] = two * y2; A[1,2] = two * z2; A[1,3] = one
        A[2,0] = two * x3; A[2,1] = two * y3; A[2,2] = two * z3; A[2,3] = one
        A[3,0] = x4;       A[3,1] = y4;       A[3,2] = z4;       A[3,3] = zero

        Bvec[0] = b1; Bvec[1] = b2; Bvec[2] = b3; Bvec[3] = b4

        X = np.linalg.solve(A, Bvec)
        u = X[0]; v = X[1]; w = X[2]; d = X[3]

        mdx1 = (x1 + x2) * half
        mdy1 = (y1 + y2) * half
        mdz1 = (z1 + z2) * half
        cx2 = -u; cy2 = -v; cz2 = -w

        denom_cz = cz2 - mdz1
        if denom_cz == zero:
            a = zero
            b = zero
        else:
            a = (cx2 - mdx1) / denom_cz
            b = (cy2 - mdy1) / denom_cz

        p1 = a*a + b*b + one
        p2 = (two*a*(mdx1 - a*mdz1) +
              two*b*(mdy1 - b*mdz1) -
              two*u*a + two*v*b + two*w)
        p3 = ((mdx1 - a*mdz1)*(mdx1 - a*mdz1) +
              (mdy1 - b*mdz1)*(mdy1 - b*mdz1) +
              two*u*(mdx1 - a*mdz1) +
              two*v*(mdy1 - b*mdz1) + d)

        rmin, rmax = safe_quad_roots_f32(p1, p2, p3)

        if Null_Hts[1] > z1:
            # prefer max(Null_Hts[1], min(roots))
            if rmax < Null_Hts[1]:
                zkc[i] = rmax
            else:
                zkc[i] = Null_Hts[1]
        else:
            # prefer min(Null_Hts[1], max(roots))
            if rmin > Null_Hts[1]:
                zkc[i] = rmin
            else:
                zkc[i] = Null_Hts[1]

    # final averages (return float32)
    B = np.mean(zkb)
    T = np.mean(zkc)
    return B, T

# ---------------------
# float64 specialized main function
# ---------------------
@njit(parallel=True, fastmath=True)
def base_crown_ht_numba_f64(R, N, tot_pts, M, step, Z, Null_Hts):
    """
    Float32/64-safe Numba implementation to compute average base (B) and crown (T) heights
    using a circle/sphere-fitting-like approach from three sample points per sample.

    Inputs:
      R : float32/64 ndarray shape (N, tot_pts, 2)
      N, tot_pts, M, step : ints
      Z : float32/64 ndarray shape (N,)
      Null_Hts : float32 ndarray shape (2,)  # [Null_base, Null_crown]

    Returns:
      B, T : float32/float64 scalars (average base and crown heights)
    """
    dt = np.float64

    zero = dt(0.0)
    one = dt(1.0)
    two = dt(2.0)
    half = dt(0.5)

    zkb = np.zeros(M, dtype=dt)
    zkc = np.zeros(M, dtype=dt)

    for i in prange(M):
        j = i * step
        k = (i * step + tot_pts // 2) % tot_pts

        # ---------- Base height (using points on levels 0 and 1) ----------
        x1 = R[0, j, 0]; y1 = R[0, j, 1]; z1 = Z[0]
        x2 = R[0, k, 0]; y2 = R[0, k, 1]; z2 = Z[0]
        x3 = R[1, j, 0]; y3 = R[1, j, 1]; z3 = Z[1]

        # 3x3 determinants for x4,y4,z4 (float32)
        x4 = -det3(y1, z1, one,
                   y2, z2, one,
                   y3, z3, one)
        y4 =  det3(x1, z1, one,
                  x2, z2, one,
                  x3, z3, one)
        z4 = -det3(x1, y1, one,
                   x2, y2, one,
                   x3, y3, one)

        b1 = -(x1*x1 + y1*y1 + z1*z1)
        b2 = -(x2*x2 + y2*y2 + z2*z2)
        b3 = -(x3*x3 + y3*y3 + z3*z3)
        b4 = det3(x1, y1, z1,
                  x2, y2, z2,
                  x3, y3, z3)

        # Build 4x4 system A X = B
        A = np.empty((4,4), dtype=dt)
        Bvec = np.empty(4, dtype=dt)

        A[0,0] = two * x1; A[0,1] = two * y1; A[0,2] = two * z1; A[0,3] = one
        A[1,0] = two * x2; A[1,1] = two * y2; A[1,2] = two * z2; A[1,3] = one
        A[2,0] = two * x3; A[2,1] = two * y3; A[2,2] = two * z3; A[2,3] = one
        A[3,0] = x4;       A[3,1] = y4;       A[3,2] = z4;       A[3,3] = zero

        Bvec[0] = b1; Bvec[1] = b2; Bvec[2] = b3; Bvec[3] = b4

        # Solve for X = [u,v,w,d]
        # np.linalg.solve is supported in Numba for fixed-size arrays of floats
        X = np.linalg.solve(A, Bvec)  # returns float32/float64 array depending on the input types
        u = X[0]; v = X[1]; w = X[2]; d = X[3]

        mdx1 = (x1 + x2) * half
        mdy1 = (y1 + y2) * half
        mdz1 = (z1 + z2) * half
        cx2 = -u; cy2 = -v; cz2 = -w

        # avoid division by zero for cz2 - mdz1
        denom_cz = cz2 - mdz1
        if denom_cz == zero:
            a = zero
            b = zero
        else:
            a = (cx2 - mdx1) / denom_cz
            b = (cy2 - mdy1) / denom_cz

        p1 = a*a + b*b + one
        p2 = (two*a*(mdx1 - a*mdz1) +
              two*b*(mdy1 - b*mdz1) -
              two*u*a + two*v*b + two*w)
        p3 = ((mdx1 - a*mdz1)*(mdx1 - a*mdz1) +
              (mdy1 - b*mdz1)*(mdy1 - b*mdz1) +
              two*u*(mdx1 - a*mdz1) +
              two*v*(mdy1 - b*mdz1) + d)

        # Solve quadratic p1*z^2 + p2*z + p3 = 0
        rmin, rmax = safe_quad_roots_f64(p1, p2, p3)

        # Choose solution as original logic
        if Null_Hts[0] < z1:
            # prefer the smaller root, but not below Null_Hts[0]
            # use max(Null_Hts[0], min(roots))
            # ensure comparisons in float32/float64
            if rmin < Null_Hts[0]:
                zkb[i] = Null_Hts[0]
            else:
                zkb[i] = rmin
        else:
            # prefer the larger root but not above Null_Hts[0]
            if rmax > Null_Hts[0]:
                zkb[i] = Null_Hts[0]
            else:
                zkb[i] = rmax

        # ---------- Crown height (top levels) ----------
        x1 = R[N-1, j, 0]; y1 = R[N-1, j, 1]; z1 = Z[N-1]
        x2 = R[N-1, k, 0]; y2 = R[N-1, k, 1]; z2 = Z[N-1]
        x3 = R[N-2, j, 0]; y3 = R[N-2, j, 1]; z3 = Z[N-2]

        x4 = -det3(y1, z1, one,
                   y2, z2, one,
                   y3, z3, one)
        y4 =  det3(x1, z1, one,
                  x2, z2, one,
                  x3, z3, one)
        z4 = -det3(x1, y1, one,
                   x2, y2, one,
                   x3, y3, one)

        b1 = -(x1*x1 + y1*y1 + z1*z1)
        b2 = -(x2*x2 + y2*y2 + z2*z2)
        b3 = -(x3*x3 + y3*y3 + z3*z3)
        b4 = det3(x1, y1, z1,
                  x2, y2, z2,
                  x3, y3, z3)

        A[0,0] = two * x1; A[0,1] = two * y1; A[0,2] = two * z1; A[0,3] = one
        A[1,0] = two * x2; A[1,1] = two * y2; A[1,2] = two * z2; A[1,3] = one
        A[2,0] = two * x3; A[2,1] = two * y3; A[2,2] = two * z3; A[2,3] = one
        A[3,0] = x4;       A[3,1] = y4;       A[3,2] = z4;       A[3,3] = zero

        Bvec[0] = b1; Bvec[1] = b2; Bvec[2] = b3; Bvec[3] = b4

        X = np.linalg.solve(A, Bvec)
        u = X[0]; v = X[1]; w = X[2]; d = X[3]

        mdx1 = (x1 + x2) * half
        mdy1 = (y1 + y2) * half
        mdz1 = (z1 + z2) * half
        cx2 = -u; cy2 = -v; cz2 = -w

        denom_cz = cz2 - mdz1
        if denom_cz == zero:
            a = zero
            b = zero
        else:
            a = (cx2 - mdx1) / denom_cz
            b = (cy2 - mdy1) / denom_cz

        p1 = a*a + b*b + one
        p2 = (two*a*(mdx1 - a*mdz1) +
              two*b*(mdy1 - b*mdz1) -
              two*u*a + two*v*b + two*w)
        p3 = ((mdx1 - a*mdz1)*(mdx1 - a*mdz1) +
              (mdy1 - b*mdz1)*(mdy1 - b*mdz1) +
              two*u*(mdx1 - a*mdz1) +
              two*v*(mdy1 - b*mdz1) + d)

        rmin, rmax = safe_quad_roots_f64(p1, p2, p3)

        if Null_Hts[1] > z1:
            # prefer max(Null_Hts[1], min(roots))
            if rmax < Null_Hts[1]:
                zkc[i] = rmax
            else:
                zkc[i] = Null_Hts[1]
        else:
            # prefer min(Null_Hts[1], max(roots))
            if rmin > Null_Hts[1]:
                zkc[i] = rmin
            else:
                zkc[i] = Null_Hts[1]

    # final averages (return float32)
    B = np.mean(zkb)
    T = np.mean(zkc)
    return B, T


# ---------------------
# float32 surface tangents
# ---------------------
@njit(parallel=True, fastmath=True)
def surf_tangent_numba_f32(R, N, tot_pts, Z, Null_Hts, RB, RC, Bh, T, base_null='y', crown_null='y'):
    """
    Float32-safe, numba-jittable version of surf_tangent.
    Inputs:
      R         : (N, tot_pts+1, 2) float32 - aligned contours (assumed float32)
      N         : int
      tot_pts   : int
      Z         : (N,) float32
      Null_Hts  : (2,) float32
      RB, RC    : (2,) float32 base and crown 2D points
      Bh, T     : float32 base and crown heights (may be overridden)
      base_null, crown_null: 'y'/'n' flags (single-character strings)
    Returns:
      gR  : (N, tot_pts+1, 2) float32  radial tangents
      gz  : (N, tot_pts+1)      float32  axial tangents
      gRB : (tot_pts+1, 2)      float32  base boundary tangents
      gRC : (tot_pts+1, 2)      float32  crown boundary tangents
      fb  : (tot_pts+1,)        float32  base scalar factors
      fc  : (tot_pts+1,)        float32  crown scalar factors
    """

    dtype = np.float32

    zero = dtype(0.0)
    one = dtype(1.0)
    two = dtype(2.0)
    m_one = dtype(-1.0)
    tol = dtype(1e-6)

    # allocate outputs (ensure float32 dtype)
    gR = np.zeros((N, tot_pts + 1, 2), dtype=dtype)
    gz = np.zeros((N, tot_pts + 1), dtype=dtype)
    gRB = np.zeros((tot_pts + 1, 2), dtype=dtype)
    gRC = np.zeros((tot_pts + 1, 2), dtype=dtype)
    fb = np.zeros(tot_pts + 1, dtype=dtype)
    fc = np.zeros(tot_pts + 1, dtype=dtype)

    # adjust Bh and T if null flags set
    if base_null == 'n' or base_null == 'N':
        Bh = Null_Hts[0]
    if crown_null == 'n' or crown_null == 'N':
        T = Null_Hts[1]

    # --- Interior points ---
    for i in prange(1, N - 1):
        z_i = Z[i]
        z_ip1 = Z[i + 1]
        z_im1 = Z[i - 1]
        for j in range(tot_pts + 1):
            # get 2D points
            A0 = R[i - 1, j, 0]; A1 = R[i - 1, j, 1]
            B0 = R[i, j, 0];     B1 = R[i, j, 1]
            C0 = R[i + 1, j, 0]; C1 = R[i + 1, j, 1]

            # check all close: use component-wise tolerance
            if (abs(A0 - B0) < tol and abs(A1 - B1) < tol and
                abs(A0 - C0) < tol and abs(A1 - C1) < tol):
                gR[i, j, 0] = zero
                gR[i, j, 1] = zero
                # sign of z difference
                diffz = z_ip1 - z_i
                if diffz >= zero:
                    gz[i, j] = one
                else:
                    gz[i, j] = m_one
            else:
                # compute norms (float32)
                alphaR = vec_norm_float([C0 - B0, C1 - B1])
                betaR  = vec_norm_float([B0 - A0, B1 - A1])
                alphaZ = z_ip1 - z_i
                betaZ  = z_i - z_im1

                # numerator = alphaR * (B - A) + betaR * (C - B)
                num0 = alphaR * (B0 - A0) + betaR * (C0 - B0)
                num1 = alphaR * (B1 - A1) + betaR * (C1 - B1)

                denom = alphaR * abs(betaZ) + betaR * abs(alphaZ)
                # safe division
                if denom == zero:
                    gR[i, j, 0] = zero
                    gR[i, j, 1] = zero
                    gz[i, j] = zero
                else:
                    gR[i, j, 0] = num0 / denom
                    gR[i, j, 1] = num1 / denom
                    gz[i, j] = (alphaR * betaZ + betaR * alphaZ) / denom

    # --- Base section (i=0) ---
    z0 = Z[0]; z1 = Z[1]
    for j in prange(tot_pts + 1):
        A0 = RB[0]; A1 = RB[1]
        B0 = R[0, j, 0]; B1 = R[0, j, 1]
        C0 = R[1, j, 0]; C1 = R[1, j, 1]

        alphaR = vec_norm_float([C0 - B0, C1 - B1])
        betaR = vec_norm_float([B0 - A0, B1 - A1])
        alphaZ = z1 - z0
        betaZ = z0 - Bh

        num0 = two * alphaR * (B0 - A0) + betaR * (C0 - B0)
        num1 = two * alphaR * (B1 - A1) + betaR * (C1 - B1)
        denom = two * alphaR * abs(betaZ) + betaR * abs(alphaZ)

        if denom == zero:
            gR[0, j, 0] = zero; gR[0, j, 1] = zero
            gz[0, j] = zero
        else:
            gR[0, j, 0] = num0 / denom
            gR[0, j, 1] = num1 / denom
            gz[0, j] = (two * alphaR * betaZ + betaR * alphaZ) / denom

    # --- Crown section (i=N-1) ---
    zN1 = Z[N - 1]; zN2 = Z[N - 2]
    for j in prange(tot_pts + 1):
        A0 = R[N - 2, j, 0]; A1 = R[N - 2, j, 1]
        B0 = R[N - 1, j, 0]; B1 = R[N - 1, j, 1]
        C0 = RC[0]; C1 = RC[1]

        alphaR = vec_norm_float([C0 - B0, C1 - B1])
        betaR = vec_norm_float([B0 - A0, B1 - A1])
        alphaZ = T - zN1
        betaZ = zN1 - zN2

        num0 = alphaR * (B0 - A0) + two * betaR * (C0 - B0)
        num1 = alphaR * (B1 - A1) + two * betaR * (C1 - B1)
        denom = alphaR * abs(betaZ) + two * betaR * abs(alphaZ)

        if denom == zero:
            gR[N - 1, j, 0] = zero; gR[N - 1, j, 1] = zero
            gz[N - 1, j] = zero
        else:
            gR[N - 1, j, 0] = num0 / denom
            gR[N - 1, j, 1] = num1 / denom
            gz[N - 1, j] = (alphaR * betaZ + two * betaR * alphaZ) / denom

    # --- Base tangent vectors gRB ---
    for j in prange(tot_pts + 1):
        A0 = RB[0]; A1 = RB[1]
        B0 = R[0, j, 0]; B1 = R[0, j, 1]
        C0 = R[1, j, 0]; C1 = R[1, j, 1]

        alpha1 = one + dtype(j % 15)
        # avoid division by zero for (Z[1]-Z[0])
        denomZ = Z[1] - Z[0]
        if denomZ == zero:
            beta1 = zero
            beta2 = zero
        else:
            beta1 = (Bh - alpha1 * (Z[0] - Bh)) / denomZ
            beta2 = (Bh - (-alpha1) * (Z[0] - Bh)) / denomZ

        D11 = alpha1 * (B0 - A0) + beta1 * (C0 - B0)
        D12 = alpha1 * (B1 - A1) + beta1 * (C1 - B1)

        D21 = -alpha1 * (B0 - A0) + beta2 * (C0 - B0)
        D22 = -alpha1 * (B1 - A1) + beta2 * (C1 - B1)

        # E = unit(B - A)
        BA_norm = vec_norm_float([B0 - A0, B1 - A1])
        if BA_norm == zero:
            E0 = one; E1 = zero  # default unit vector
        else:
            E0 = (B0 - A0) / BA_norm
            E1 = (B1 - A1) / BA_norm

        # F = unit(D1 - D2)
        F0 = D11 - D21
        F1 = D12 - D22
        F_norm = vec_norm_float([F0, F1])
        if F_norm == zero:
            F0_u = E0; F1_u = E1
        else:
            F0_u = F0 / F_norm
            F1_u = F1 / F_norm

        cross1 = cross2d_float([E0, E1], [F0_u, F1_u])
        dot1 = E0 * F0_u + E1 * F1_u

        if abs(cross1) < tol:
            gRB[j, 0] = E0; gRB[j, 1] = E1
        elif dot1 > zero:
            gRB[j, 0] = F0_u; gRB[j, 1] = F1_u
        else:
            gRB[j, 0] = -F0_u; gRB[j, 1] = -F1_u

    # --- Crown tangent vectors gRC ---
    for j in prange(tot_pts + 1):
        A0 = R[N - 2, j, 0]; A1 = R[N - 2, j, 1]
        B0 = R[N - 1, j, 0]; B1 = R[N - 1, j, 1]
        C0 = RC[0]; C1 = RC[1]

        alpha1 = one + dtype(j % 15)
        denomZ = T - Z[N - 1]
        if denomZ == zero:
            beta1 = zero
            beta2 = zero
        else:
            beta1 = (T - alpha1 * (Z[N - 1] - Z[N - 2])) / denomZ
            beta2 = (T - (-alpha1) * (Z[N - 1] - Z[N - 2])) / denomZ

        D11 = alpha1 * (B0 - A0) + beta1 * (C0 - B0)
        D12 = alpha1 * (B1 - A1) + beta1 * (C1 - B1)

        D21 = -alpha1 * (B0 - A0) + beta2 * (C0 - B0)
        D22 = -alpha1 * (B1 - A1) + beta2 * (C1 - B1)

        # E = unit(C - B)
        CB_norm = vec_norm_float([C0 - B0, C1 - B1])
        if CB_norm == zero:
            E0 = one; E1 = zero
        else:
            E0 = (C0 - B0) / CB_norm
            E1 = (C1 - B1) / CB_norm

        # F = unit(D1 - D2)
        F0 = D11 - D21
        F1 = D12 - D22
        F_norm = vec_norm_float([F0, F1])
        if F_norm == zero:
            F0_u = E0; F1_u = E1
        else:
            F0_u = F0 / F_norm
            F1_u = F1 / F_norm

        cross1 = cross2d_float([E0, E1], [F0_u, F1_u])
        dot1 = E0 * F0_u + E1 * F1_u

        if abs(cross1) < tol:
            gRC[j, 0] = E0; gRC[j, 1] = E1
        elif dot1 > zero:
            gRC[j, 0] = F0_u; gRC[j, 1] = F1_u
        else:
            gRC[j, 0] = -F0_u; gRC[j, 1] = -F1_u

    # --- fb and fc computations ---
    for j in prange(tot_pts + 1):
        A0 = RB[0]; A1 = RB[1]
        B0 = R[0, j, 0]; B1 = R[0, j, 1]
        Cmag = vec_norm_float([B0 - A0, B1 - A1])
        # protect divide by zero
        if Cmag == zero:
            fb[j] = zero
        else:
            # compute norm(gR[0,j] / Cmag)
            gx = gR[0, j, 0] / Cmag
            gy = gR[0, j, 1] / Cmag
            denom = np.sqrt(one + abs(Z[0] - Bh) * np.sqrt(gx*gx + gy*gy))
            if denom == zero:
                fb[j] = zero
            else:
                fb[j] = np.sqrt(two) * Cmag / denom

    for j in prange(tot_pts + 1):
        A0 = RC[0]; A1 = RC[1]
        B0 = R[N - 1, j, 0]; B1 = R[N - 1, j, 1]
        Cmag = vec_norm_float([A0 - B0, A1 - B1])
        if Cmag == zero:
            fc[j] = zero
        else:
            gx = gR[N - 1, j, 0] / Cmag
            gy = gR[N - 1, j, 1] / Cmag
            denom = np.sqrt(one + abs(T - Z[N - 1]) * np.sqrt(gx*gx + gy*gy))
            if denom == zero:
                fc[j] = zero
            else:
                fc[j] = np.sqrt(two) * Cmag / denom

    return gR, gz, gRB, gRC, fb, fc

# ---------------------
# float64 surface tangents
# ---------------------
@njit(parallel=True, fastmath=True)
def surf_tangent_numba_f64(R, N, tot_pts, Z, Null_Hts, RB, RC, Bh, T, base_null='y', crown_null='y'):
    """
    Float32-safe, numba-jittable version of surf_tangent.
    Inputs:
      R         : (N, tot_pts+1, 2) float32 - aligned contours (assumed float32)
      N         : int
      tot_pts   : int
      Z         : (N,) float32
      Null_Hts  : (2,) float32
      RB, RC    : (2,) float32 base and crown 2D points
      Bh, T     : float32 base and crown heights (may be overridden)
      base_null, crown_null: 'y'/'n' flags (single-character strings)
    Returns:
      gR  : (N, tot_pts+1, 2) float32  radial tangents
      gz  : (N, tot_pts+1)      float32  axial tangents
      gRB : (tot_pts+1, 2)      float32  base boundary tangents
      gRC : (tot_pts+1, 2)      float32  crown boundary tangents
      fb  : (tot_pts+1,)        float32  base scalar factors
      fc  : (tot_pts+1,)        float32  crown scalar factors
    """

    dtype = np.float64

    zero = dtype(0.0)
    one = dtype(1.0)
    two = dtype(2.0)
    m_one = dtype(-1.0)
    tol = dtype(1e-6)

    # allocate outputs (ensure float32 dtype)
    gR = np.zeros((N, tot_pts + 1, 2), dtype=dtype)
    gz = np.zeros((N, tot_pts + 1), dtype=dtype)
    gRB = np.zeros((tot_pts + 1, 2), dtype=dtype)
    gRC = np.zeros((tot_pts + 1, 2), dtype=dtype)
    fb = np.zeros(tot_pts + 1, dtype=dtype)
    fc = np.zeros(tot_pts + 1, dtype=dtype)

    # adjust Bh and T if null flags set
    if base_null == 'n' or base_null == 'N':
        Bh = Null_Hts[0]
    if crown_null == 'n' or crown_null == 'N':
        T = Null_Hts[1]

    # --- Interior points ---
    for i in prange(1, N - 1):
        z_i = Z[i]
        z_ip1 = Z[i + 1]
        z_im1 = Z[i - 1]
        for j in range(tot_pts + 1):
            # get 2D points
            A0 = R[i - 1, j, 0]; A1 = R[i - 1, j, 1]
            B0 = R[i, j, 0];     B1 = R[i, j, 1]
            C0 = R[i + 1, j, 0]; C1 = R[i + 1, j, 1]

            # check all close: use component-wise tolerance
            if (abs(A0 - B0) < tol and abs(A1 - B1) < tol and
                abs(A0 - C0) < tol and abs(A1 - C1) < tol):
                gR[i, j, 0] = zero
                gR[i, j, 1] = zero
                # sign of z difference
                diffz = z_ip1 - z_i
                if diffz >= zero:
                    gz[i, j] = one
                else:
                    gz[i, j] = m_one
            else:
                # compute norms (float32)
                alphaR = vec_norm_float([C0 - B0, C1 - B1])
                betaR  = vec_norm_float([B0 - A0, B1 - A1])
                alphaZ = z_ip1 - z_i
                betaZ  = z_i - z_im1

                # numerator = alphaR * (B - A) + betaR * (C - B)
                num0 = alphaR * (B0 - A0) + betaR * (C0 - B0)
                num1 = alphaR * (B1 - A1) + betaR * (C1 - B1)

                denom = alphaR * abs(betaZ) + betaR * abs(alphaZ)
                # safe division
                if denom == zero:
                    gR[i, j, 0] = zero
                    gR[i, j, 1] = zero
                    gz[i, j] = zero
                else:
                    gR[i, j, 0] = num0 / denom
                    gR[i, j, 1] = num1 / denom
                    gz[i, j] = (alphaR * betaZ + betaR * alphaZ) / denom

    # --- Base section (i=0) ---
    z0 = Z[0]; z1 = Z[1]
    for j in prange(tot_pts + 1):
        A0 = RB[0]; A1 = RB[1]
        B0 = R[0, j, 0]; B1 = R[0, j, 1]
        C0 = R[1, j, 0]; C1 = R[1, j, 1]

        alphaR = vec_norm_float([C0 - B0, C1 - B1])
        betaR = vec_norm_float([B0 - A0, B1 - A1])
        alphaZ = z1 - z0
        betaZ = z0 - Bh

        num0 = two * alphaR * (B0 - A0) + betaR * (C0 - B0)
        num1 = two * alphaR * (B1 - A1) + betaR * (C1 - B1)
        denom = two * alphaR * abs(betaZ) + betaR * abs(alphaZ)

        if denom == zero:
            gR[0, j, 0] = zero; gR[0, j, 1] = zero
            gz[0, j] = zero
        else:
            gR[0, j, 0] = num0 / denom
            gR[0, j, 1] = num1 / denom
            gz[0, j] = (two * alphaR * betaZ + betaR * alphaZ) / denom

    # --- Crown section (i=N-1) ---
    zN1 = Z[N - 1]; zN2 = Z[N - 2]
    for j in prange(tot_pts + 1):
        A0 = R[N - 2, j, 0]; A1 = R[N - 2, j, 1]
        B0 = R[N - 1, j, 0]; B1 = R[N - 1, j, 1]
        C0 = RC[0]; C1 = RC[1]

        alphaR = vec_norm_float([C0 - B0, C1 - B1])
        betaR = vec_norm_float([B0 - A0, B1 - A1])
        alphaZ = T - zN1
        betaZ = zN1 - zN2

        num0 = alphaR * (B0 - A0) + two * betaR * (C0 - B0)
        num1 = alphaR * (B1 - A1) + two * betaR * (C1 - B1)
        denom = alphaR * abs(betaZ) + two * betaR * abs(alphaZ)

        if denom == zero:
            gR[N - 1, j, 0] = zero; gR[N - 1, j, 1] = zero
            gz[N - 1, j] = zero
        else:
            gR[N - 1, j, 0] = num0 / denom
            gR[N - 1, j, 1] = num1 / denom
            gz[N - 1, j] = (alphaR * betaZ + two * betaR * alphaZ) / denom

    # --- Base tangent vectors gRB ---
    for j in prange(tot_pts + 1):
        A0 = RB[0]; A1 = RB[1]
        B0 = R[0, j, 0]; B1 = R[0, j, 1]
        C0 = R[1, j, 0]; C1 = R[1, j, 1]

        alpha1 = one + dtype(j % 15)
        # avoid division by zero for (Z[1]-Z[0])
        denomZ = Z[1] - Z[0]
        if denomZ == zero:
            beta1 = zero
            beta2 = zero
        else:
            beta1 = (Bh - alpha1 * (Z[0] - Bh)) / denomZ
            beta2 = (Bh - (-alpha1) * (Z[0] - Bh)) / denomZ

        D11 = alpha1 * (B0 - A0) + beta1 * (C0 - B0)
        D12 = alpha1 * (B1 - A1) + beta1 * (C1 - B1)

        D21 = -alpha1 * (B0 - A0) + beta2 * (C0 - B0)
        D22 = -alpha1 * (B1 - A1) + beta2 * (C1 - B1)

        # E = unit(B - A)
        BA_norm = vec_norm_float([B0 - A0, B1 - A1])
        if BA_norm == zero:
            E0 = one; E1 = zero  # default unit vector
        else:
            E0 = (B0 - A0) / BA_norm
            E1 = (B1 - A1) / BA_norm

        # F = unit(D1 - D2)
        F0 = D11 - D21
        F1 = D12 - D22
        F_norm = vec_norm_float([F0, F1])
        if F_norm == zero:
            F0_u = E0; F1_u = E1
        else:
            F0_u = F0 / F_norm
            F1_u = F1 / F_norm

        cross1 = cross2d_float([E0, E1], [F0_u, F1_u])
        dot1 = E0 * F0_u + E1 * F1_u

        if abs(cross1) < tol:
            gRB[j, 0] = E0; gRB[j, 1] = E1
        elif dot1 > zero:
            gRB[j, 0] = F0_u; gRB[j, 1] = F1_u
        else:
            gRB[j, 0] = -F0_u; gRB[j, 1] = -F1_u

    # --- Crown tangent vectors gRC ---
    for j in prange(tot_pts + 1):
        A0 = R[N - 2, j, 0]; A1 = R[N - 2, j, 1]
        B0 = R[N - 1, j, 0]; B1 = R[N - 1, j, 1]
        C0 = RC[0]; C1 = RC[1]

        alpha1 = one + dtype(j % 15)
        denomZ = T - Z[N - 1]
        if denomZ == zero:
            beta1 = zero
            beta2 = zero
        else:
            beta1 = (T - alpha1 * (Z[N - 1] - Z[N - 2])) / denomZ
            beta2 = (T - (-alpha1) * (Z[N - 1] - Z[N - 2])) / denomZ

        D11 = alpha1 * (B0 - A0) + beta1 * (C0 - B0)
        D12 = alpha1 * (B1 - A1) + beta1 * (C1 - B1)

        D21 = -alpha1 * (B0 - A0) + beta2 * (C0 - B0)
        D22 = -alpha1 * (B1 - A1) + beta2 * (C1 - B1)

        # E = unit(C - B)
        CB_norm = vec_norm_float([C0 - B0, C1 - B1])
        if CB_norm == zero:
            E0 = one; E1 = zero
        else:
            E0 = (C0 - B0) / CB_norm
            E1 = (C1 - B1) / CB_norm

        # F = unit(D1 - D2)
        F0 = D11 - D21
        F1 = D12 - D22
        F_norm = vec_norm_float([F0, F1])
        if F_norm == zero:
            F0_u = E0; F1_u = E1
        else:
            F0_u = F0 / F_norm
            F1_u = F1 / F_norm

        cross1 = cross2d_float([E0, E1], [F0_u, F1_u])
        dot1 = E0 * F0_u + E1 * F1_u

        if abs(cross1) < tol:
            gRC[j, 0] = E0; gRC[j, 1] = E1
        elif dot1 > zero:
            gRC[j, 0] = F0_u; gRC[j, 1] = F1_u
        else:
            gRC[j, 0] = -F0_u; gRC[j, 1] = -F1_u

    # --- fb and fc computations ---
    for j in prange(tot_pts + 1):
        A0 = RB[0]; A1 = RB[1]
        B0 = R[0, j, 0]; B1 = R[0, j, 1]
        Cmag = vec_norm_float([B0 - A0, B1 - A1])
        # protect divide by zero
        if Cmag == zero:
            fb[j] = zero
        else:
            # compute norm(gR[0,j] / Cmag)
            gx = gR[0, j, 0] / Cmag
            gy = gR[0, j, 1] / Cmag
            denom = np.sqrt(one + abs(Z[0] - Bh) * np.sqrt(gx*gx + gy*gy))
            if denom == zero:
                fb[j] = zero
            else:
                fb[j] = np.sqrt(two) * Cmag / denom

    for j in prange(tot_pts + 1):
        A0 = RC[0]; A1 = RC[1]
        B0 = R[N - 1, j, 0]; B1 = R[N - 1, j, 1]
        Cmag = vec_norm_float([A0 - B0, A1 - B1])
        if Cmag == zero:
            fc[j] = zero
        else:
            gx = gR[N - 1, j, 0] / Cmag
            gy = gR[N - 1, j, 1] / Cmag
            denom = np.sqrt(one + abs(T - Z[N - 1]) * np.sqrt(gx*gx + gy*gy))
            if denom == zero:
                fc[j] = zero
            else:
                fc[j] = np.sqrt(two) * Cmag / denom

    return gR, gz, gRB, gRC, fb, fc

# ---------------------
# float32 surface points
# ---------------------
@njit(parallel=True, fastmath=True)
def surf_pts_numba_f32(R, N, tot_pts, Z, RB, RC, Bh, T, gRB, gRC, fb, fc, gR, gz, base_circular, crown_circular, n):

    dtype = np.float32

    zero = dtype(0.0)
    one = dtype(1.0)
    two = dtype(2.0)
    three = dtype(3.0)

    u = np.linspace(zero, one, n + 1)
    u = u.astype(dtype)

    L0 = one - three * u**2 + two * u**3
    L1 = three * u**2 - two * u**3
    H0 = u - two * u**2 + u**3
    H1 = -u**2 + u**3

    FR = np.zeros((N + 1, tot_pts + 1, n + 1, 2), dtype=dtype)
    Fz = np.zeros((N + 1, tot_pts + 1, n + 1), dtype=dtype)

    # --- Interior ---
    for i in range(1, N):
        deltaZ = np.abs(Z[i] - Z[i - 1])
        for j in range(tot_pts + 1):
            FR[i, j, :, 0] = (
                L0 * R[i - 1, j, 0]
                + L1 * R[i, j, 0]
                + deltaZ * (H0 * gR[i - 1, j, 0] + H1 * gR[i, j, 0])
            )
            FR[i, j, :, 1] = (
                L0 * R[i - 1, j, 1]
                + L1 * R[i, j, 1]
                + deltaZ * (H0 * gR[i - 1, j, 1] + H1 * gR[i, j, 1])
            )
            Fz[i, j, :] = (
                Z[i - 1] * L0
                + Z[i] * L1
                + deltaZ * (gz[i - 1, j] * H0 + gz[i, j] * H1)
            )

    # --- Base ---
    for j in range(tot_pts + 1):
        if base_circular in ('y', 'Y'):
            FR[0, j, :, 0] = (
                L0 * RB[0]
                + L1 * R[0, j, 0]
                + fb[j] * H0 * gRB[j, 0]
                + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 0]
            )
            FR[0, j, :, 1] = (
                L0 * RB[1]
                + L1 * R[0, j, 1]
                + fb[j] * H0 * gRB[j, 1]
                + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 1]
            )
            Fz[0, j, :] = (one - u**2) * Bh + u**2 * Z[0]
        else:
            FR[0, j, :, 0] = L0 * RB[0] + L1 * R[0, j, 0] + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 0]
            FR[0, j, :, 1] = L0 * RB[1] + L1 * R[0, j, 1] + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 1]
            Fz[0, j, :] = (
                Bh * L0 + Z[0] * L1
                + (Z[0] - Bh) * H0
                + np.abs(Z[0] - Bh) * gz[0, j] * H1
            )

    # --- Crown ---
    for j in range(tot_pts + 1):
        if crown_circular in ('y', 'Y'):
            FR[N, j, :, 0] = (
                L0 * R[N - 1, j, 0]
                + L1 * RC[0]
                + fc[j] * H1 * gRC[j, 0]
                + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 0]
            )
            FR[N, j, :, 1] = (
                L0 * R[N - 1, j, 1]
                + L1 * RC[1]
                + fc[j] * H1 * gRC[j, 1]
                + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 1]
            )
            Fz[N, j, :] = (one - u)**2 * Z[N - 1] + u * (two - u) * T
        else:
            FR[N, j, :, 0] = L0 * R[N - 1, j, 0] + L1 * RC[0] + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 0]
            FR[N, j, :, 1] = L0 * R[N - 1, j, 1] + L1 * RC[1] + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 1]
            Fz[N, j, :] = (
                Z[N - 1] * L0 + T * L1
                + np.abs(T - Z[N - 1]) * gz[N - 1, j] * H0
                + (T - Z[N - 1]) * H1
            )

    return FR, Fz

# ---------------------
# float64 surface points
# ---------------------
@njit(parallel=True, fastmath=True)
def surf_pts_numba_f64(R, N, tot_pts, Z, RB, RC, Bh, T, gRB, gRC, fb, fc, gR, gz, base_circular, crown_circular, n):

    dtype = np.float64

    zero = dtype(0.0)
    one = dtype(1.0)
    two = dtype(2.0)
    three = dtype(3.0)

    u = np.linspace(zero, one, n + 1)
    u = u.astype(dtype)

    L0 = one - three * u**2 + two * u**3
    L1 = three * u**2 - two * u**3
    H0 = u - two * u**2 + u**3
    H1 = -u**2 + u**3

    FR = np.zeros((N + 1, tot_pts + 1, n + 1, 2), dtype=dtype)
    Fz = np.zeros((N + 1, tot_pts + 1, n + 1), dtype=dtype)

    # --- Interior ---
    for i in range(1, N):
        deltaZ = np.abs(Z[i] - Z[i - 1])
        for j in range(tot_pts + 1):
            FR[i, j, :, 0] = (
                L0 * R[i - 1, j, 0]
                + L1 * R[i, j, 0]
                + deltaZ * (H0 * gR[i - 1, j, 0] + H1 * gR[i, j, 0])
            )
            FR[i, j, :, 1] = (
                L0 * R[i - 1, j, 1]
                + L1 * R[i, j, 1]
                + deltaZ * (H0 * gR[i - 1, j, 1] + H1 * gR[i, j, 1])
            )
            Fz[i, j, :] = (
                Z[i - 1] * L0
                + Z[i] * L1
                + deltaZ * (gz[i - 1, j] * H0 + gz[i, j] * H1)
            )

    # --- Base ---
    for j in range(tot_pts + 1):
        if base_circular in ('y', 'Y'):
            FR[0, j, :, 0] = (
                L0 * RB[0]
                + L1 * R[0, j, 0]
                + fb[j] * H0 * gRB[j, 0]
                + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 0]
            )
            FR[0, j, :, 1] = (
                L0 * RB[1]
                + L1 * R[0, j, 1]
                + fb[j] * H0 * gRB[j, 1]
                + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 1]
            )
            Fz[0, j, :] = (one - u**2) * Bh + u**2 * Z[0]
        else:
            FR[0, j, :, 0] = L0 * RB[0] + L1 * R[0, j, 0] + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 0]
            FR[0, j, :, 1] = L0 * RB[1] + L1 * R[0, j, 1] + two * np.abs(Z[0] - Bh) * H1 * gR[0, j, 1]
            Fz[0, j, :] = (
                Bh * L0 + Z[0] * L1
                + (Z[0] - Bh) * H0
                + np.abs(Z[0] - Bh) * gz[0, j] * H1
            )

    # --- Crown ---
    for j in range(tot_pts + 1):
        if crown_circular in ('y', 'Y'):
            FR[N, j, :, 0] = (
                L0 * R[N - 1, j, 0]
                + L1 * RC[0]
                + fc[j] * H1 * gRC[j, 0]
                + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 0]
            )
            FR[N, j, :, 1] = (
                L0 * R[N - 1, j, 1]
                + L1 * RC[1]
                + fc[j] * H1 * gRC[j, 1]
                + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 1]
            )
            Fz[N, j, :] = (one - u)**2 * Z[N - 1] + u * (two - u) * T
        else:
            FR[N, j, :, 0] = L0 * R[N - 1, j, 0] + L1 * RC[0] + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 0]
            FR[N, j, :, 1] = L0 * R[N - 1, j, 1] + L1 * RC[1] + two * np.abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 1]
            Fz[N, j, :] = (
                Z[N - 1] * L0 + T * L1
                + np.abs(T - Z[N - 1]) * gz[N - 1, j] * H0
                + (T - Z[N - 1]) * H1
            )

    return FR, Fz

@cuda.jit
def surf_pts_gpu(R, Z, RB, RC, Bh, T, gRB, gRC, fb, fc, gR, gz,
                 base_circular, crown_circular, FR, Fz, u, L0, L1, H0, H1):
    i, j, k = cuda.grid(3)
    N, M, n = gR.shape[0], gR.shape[1], len(u)
    if i < N + 1 and j < M and k < n:
        if i > 0 and i < N:
            deltaZ = abs(Z[i] - Z[i-1])
            FR[i, j, k, 0] = L0[k]*R[i-1,j,0] + L1[k]*R[i,j,0] + deltaZ*(H0[k]*gR[i-1,j,0] + H1[k]*gR[i,j,0])
            FR[i, j, k, 1] = L0[k]*R[i-1,j,1] + L1[k]*R[i,j,1] + deltaZ*(H0[k]*gR[i-1,j,1] + H1[k]*gR[i,j,1])
            Fz[i, j, k] = Z[i-1]*L0[k] + Z[i]*L1[k] + deltaZ*(gz[i-1,j]*H0[k] + gz[i,j]*H1[k])

        elif i == 0:  # base
            if base_circular in (ord('y'), ord('Y')):
                FR[i,j,k,0] = L0[k]*RB[0] + L1[k]*R[0,j,0] + fb[j]*H0[k]*gRB[j,0] + 2*abs(Z[0]-Bh)*H1[k]*gR[0,j,0]
                FR[i,j,k,1] = L0[k]*RB[1] + L1[k]*R[0,j,1] + fb[j]*H0[k]*gRB[j,1] + 2*abs(Z[0]-Bh)*H1[k]*gR[0,j,1]
                Fz[i,j,k] = (1-u[k]**2)*Bh + u[k]**2*Z[0]
            else:
                FR[i,j,k,0] = L0[k]*RB[0] + L1[k]*R[0,j,0] + 2*abs(Z[0]-Bh)*H1[k]*gR[0,j,0]
                FR[i,j,k,1] = L0[k]*RB[1] + L1[k]*R[0,j,1] + 2*abs(Z[0]-Bh)*H1[k]*gR[0,j,1]
                Fz[i,j,k] = Bh*L0[k] + Z[0]*L1[k] + (Z[0]-Bh)*H0[k] + abs(Z[0]-Bh)*gz[0,j]*H1[k]

        elif i == N:  # crown
            if crown_circular in (ord('y'), ord('Y')):
                FR[i,j,k,0] = L0[k]*R[N-1,j,0] + L1[k]*RC[0] + fc[j]*H1[k]*gRC[j,0] + 2*abs(T-Z[N-1])*H0[k]*gR[N-1,j,0]
                FR[i,j,k,1] = L0[k]*R[N-1,j,1] + L1[k]*RC[1] + fc[j]*H1[k]*gRC[j,1] + 2*abs(T-Z[N-1])*H0[k]*gR[N-1,j,1]
                Fz[i,j,k] = (1-u[k])**2*Z[N-1] + u[k]*(2-u[k])*T
            else:
                FR[i,j,k,0] = L0[k]*R[N-1,j,0] + L1[k]*RC[0] + 2*abs(T-Z[N-1])*H0[k]*gR[N-1,j,0]
                FR[i,j,k,1] = L0[k]*R[N-1,j,1] + L1[k]*RC[1] + 2*abs(T-Z[N-1])*H0[k]*gR[N-1,j,1]
                Fz[i,j,k] = Z[N-1]*L0[k] + T*L1[k] + abs(T-Z[N-1])*gz[N-1,j]*H0[k] + (T-Z[N-1])*H1[k]
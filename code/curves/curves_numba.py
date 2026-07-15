# curves_numba.py
# ---------------
# Numba-accelerated implementations of Goodman’s 2D curve smoothing method.
# This module provides two versions of the algorithm specialized for
# float32 and float64 precision, compiled with @njit for high-performance
# CPU execution using parallel loops and fast math.
#
# ⚠️ This file uses Numba ONLY (no GPU, no PyTorch).
# It is intended as a performance-optimized counterpart to
# `curves_python_loops.py`.
#
# PyTorch and GPU-enabled variants are provided in separate modules.

import numpy as np
from numba import njit, prange


# --- Helper functions (all float32-safe) ---
@njit(fastmath=True)
def cross2d_float(a, b):
    # a, b are length-2 arrays (float32)
    return a[0]*b[1] - a[1]*b[0]


@njit(fastmath=True)
def vec_norm_float(a):
    # Euclidean norm for a length-2 float32 vector
    return np.sqrt(a[0]*a[0] + a[1]*a[1])


@njit(fastmath=True)
def safe_div_float(num, den, dtype=None):
    # safe division returning float32, handle den == 0
    if den == 0.0:
        if dtype is None:
            return np.float64(0.0)
        else:
            return dtype(0.0)
    else:
        return num / den


# --- Rewritten curve function (float32-safe) ---
# ---------------------
# float32 specialized function
# ---------------------
@njit(parallel=True, fastmath=True)
def curve_goodman_numba_f32(I, seg_pts):
    """
    Goodman-style closed curve smoothing — float32-safe Numba implementation.

    Parameters
    ----------
    I : (no_pts, 2) ndarray (float32)
        Closed polygon points in order (I[0]..I[no_pts-1]), dtype must be np.float32.
    seg_pts : int
        Number of interpolation points per segment (end point excluded).

    Returns
    -------
    r : (no_pts * seg_pts, 2) ndarray (float32)
        Smoothed curve points in float32.
    """
    dtype = np.float32

    no_pts = I.shape[0]

    # constants as dtype
    m = dtype(0.5)
    n = dtype(0.5)
    r_coef = dtype(0.25)
    s_coef = dtype(0.25)
    zero = dtype(0.0)
    one = dtype(1.0)
    two = dtype(2.0)
    eps = dtype(1e-6)

    # build t values in float32 excluding the final endpoint
    t_vals = np.empty(seg_pts, dtype=dtype)
    for j in range(seg_pts):
        # j / seg_pts  in float32
        t_vals[j] = dtype(j) / dtype(seg_pts)

    # allocate arrays (all float32)
    lineleft = np.zeros(no_pts, dtype=np.bool_)
    lineright = np.zeros(no_pts, dtype=np.bool_)
    kleft = np.zeros(no_pts, dtype=dtype)
    kright = np.zeros(no_pts, dtype=dtype)

    # --- curvature flags + curvature values ---
    for i in prange(no_pts):
        im1 = (i - 1) % no_pts
        ip1 = (i + 1) % no_pts
        ip2 = (i + 2) % no_pts
        im2 = (i - 2) % no_pts

        # collinearity checks using cross products (float32)
        lhs1 = (I[i,1] - I[im1,1]) * (I[ip1,0] - I[im1,0]) - (I[ip1,1] - I[im1,1]) * (I[i,0] - I[im1,0])
        if abs(lhs1) < eps:  # tolerance in float32/float64
            lineleft[i] = True
            lineright[i] = True
        else:
            lhs2 = (I[i,1] - I[ip1,1]) * (I[ip2,0] - I[ip1,0]) - (I[ip2,1] - I[ip1,1]) * (I[i,0] - I[ip1,0])
            if abs(lhs2) < eps:
                lineright[i] = True
            else:
                lhs3 = (I[i,1] - I[im1,1]) * (I[im2,0] - I[im1,0]) - (I[im2,1] - I[im1,1]) * (I[i,0] - I[im1,0])
                if abs(lhs3) < eps:
                    lineleft[i] = True

        if not lineleft[i]:
            # compute curvature-like quantity (float32-safe)
            v1 = np.empty(2, dtype=dtype); v2 = np.empty(2, dtype=dtype)
            v1[0] = I[i,0] - I[im1,0]; v1[1] = I[i,1] - I[im1,1]
            v2[0] = I[ip1,0] - I[i,0]; v2[1] = I[ip1,1] - I[i,1]
            num = two * cross2d_float(v1, v2)
            den = vec_norm_float(v1) * vec_norm_float(v2) * vec_norm_float(np.array([I[ip1,0]-I[im1,0], I[ip1,1]-I[im1,1]], dtype=dtype))
            kleft[i] = safe_div_float(num, den, dtype)

        if not lineright[i]:
            v1 = np.empty(2, dtype=dtype); v2 = np.empty(2, dtype=dtype)
            v1[0] = I[i,0] - I[im1,0]; v1[1] = I[i,1] - I[im1,1]
            v2[0] = I[ip1,0] - I[i,0]; v2[1] = I[ip1,1] - I[i,1]
            num = two * cross2d_float(v1, v2)
            den = vec_norm_float(v1) * vec_norm_float(v2) * vec_norm_float(np.array([I[ip1,0]-I[im1,0], I[ip1,1]-I[im1,1]], dtype=dtype))
            kright[i] = safe_div_float(num, den, dtype)

    # --- tangents Tu (unit directions) and T (unnormalized) ---
    T = np.zeros((no_pts, 2), dtype=dtype)
    Tu = np.zeros((no_pts, 2), dtype=dtype)

    for i in prange(no_pts):
        im1 = (i - 1) % no_pts
        ip1 = (i + 1) % no_pts
        if (kleft[i] != 0.0) or (kright[i] != 0.0):
            a = np.abs(kleft[ip1]) * (vec_norm_float(np.array([I[ip1,0]-I[i,0], I[ip1,1]-I[i,1]], dtype=dtype))**2)
            b = np.abs(kright[im1]) * (vec_norm_float(np.array([I[i,0]-I[im1,0], I[i,1]-I[im1,1]], dtype=dtype))**2)
            T_i0 = a * (I[i,0] - I[im1,0]) + b * (I[ip1,0] - I[i,0])
            T_i1 = a * (I[i,1] - I[im1,1]) + b * (I[ip1,1] - I[i,1])
            T[i,0] = T_i0; T[i,1] = T_i1
            normT = vec_norm_float(np.array([T[i,0], T[i,1]], dtype=dtype))
            if normT != zero:
                Tu[i,0] = T[i,0] / normT
                Tu[i,1] = T[i,1] / normT
            else:
                Tu[i,0] = zero; Tu[i,1] = zero

    # --- Bezier control points and lengths (float32) ---
    A = np.zeros((no_pts, 2), dtype=dtype)
    B = np.zeros((no_pts, 2), dtype=dtype)
    C = np.zeros((no_pts, 2), dtype=dtype)
    D = np.zeros((no_pts, 2), dtype=dtype)
    lengthab = np.zeros(no_pts, dtype=dtype)
    lengthcd = np.zeros(no_pts, dtype=dtype)

    for i in prange(no_pts):
        ip1 = (i + 1) % no_pts

        if (kright[i] * kleft[ip1]) > zero:
            # convex case
            T_i = np.array([T[i,0], T[i,1]], dtype=dtype)
            T_ip1 = np.array([T[ip1,0], T[ip1,1]], dtype=dtype)
            L = vec_norm_float(np.array([I[ip1,0] - I[i,0], I[ip1,1] - I[i,1]], dtype=dtype))
            # compute sines safely
            sina = safe_div_float(cross2d_float(T_i, np.array([I[ip1,0]-I[i,0], I[ip1,1]-I[i,1]], dtype=dtype)),
                                vec_norm_float(T_i) * L, dtype)
            sinb = safe_div_float(cross2d_float(np.array([I[ip1,0]-I[i,0], I[ip1,1]-I[i,1]], dtype=dtype), T_ip1),
                                vec_norm_float(T_ip1) * L, dtype)
            sinab = safe_div_float(cross2d_float(T_i, T_ip1), vec_norm_float(T_i) * vec_norm_float(T_ip1), dtype)

            p = safe_div_float(two * np.abs(sinb),
                             two * m * np.abs(sinb) + (one - m) * L * np.abs(kleft[ip1]) + two * np.abs(sinab), dtype)
            q = safe_div_float(two * np.abs(sina),
                             two * n * np.abs(sina) + (one - n) * L * np.abs(kright[i]) + two * np.abs(sinab), dtype)

            lengthab[i] = p * L
            lengthcd[i] = q * L

            A[i,0] = I[i,0]; A[i,1] = I[i,1]
            D[i,0] = I[ip1,0]; D[i,1] = I[ip1,1]
            B[i,0] = A[i,0] + lengthab[i] * Tu[i,0]
            B[i,1] = A[i,1] + lengthab[i] * Tu[i,1]
            C[i,0] = D[i,0] - lengthcd[i] * Tu[ip1,0]
            C[i,1] = D[i,1] - lengthcd[i] * Tu[ip1,1]

        elif (kright[i] * kleft[ip1]) < zero:
            # inflection
            L = vec_norm_float(np.array([I[ip1,0] - I[i,0], I[ip1,1] - I[i,1]], dtype=dtype))
            lengthab[i] = r_coef * L
            lengthcd[i] = s_coef * L

            A[i,0] = I[i,0]; A[i,1] = I[i,1]
            D[i,0] = I[ip1,0]; D[i,1] = I[ip1,1]
            B[i,0] = A[i,0] + lengthab[i] * Tu[i,0]
            B[i,1] = A[i,1] + lengthab[i] * Tu[i,1]
            C[i,0] = D[i,0] - lengthcd[i] * Tu[ip1,0]
            C[i,1] = D[i,1] - lengthcd[i] * Tu[ip1,1]

        else:
            # straight segment
            A[i,0] = I[i,0]; A[i,1] = I[i,1]
            D[i,0] = I[ip1,0]; D[i,1] = I[ip1,1]

    # --- weights alpha and beta (float32) ---
    alpha = np.zeros(no_pts, dtype=dtype)
    beta = np.zeros(no_pts, dtype=dtype)

    for i in prange(no_pts):
        ip1 = (i + 1) % no_pts
        if kright[i] != zero:
            denom1 = two * cross2d_float(B[i] - A[i], C[i] - B[i])
            if denom1 != zero:
                alpha[i] = kright[i] * (lengthab[i]**3) / denom1
            denom2 = two * cross2d_float(C[i] - B[i], D[i] - C[i])
            if denom2 != zero:
                beta[i] = kleft[ip1] * (lengthcd[i]**3) / denom2

    # --- final curve points (float32) ---
    r = np.zeros((no_pts * seg_pts, 2), dtype=dtype)

    for i in prange(no_pts):
        ip1 = (i + 1) % no_pts
        if kright[i] != zero:
            for j in range(seg_pts):
                tj = t_vals[j]
                one_minus = one - tj
                # numerator and denominator computed in float32
                num0 = (A[i,0]*alpha[i]*(one_minus**3) +
                        B[i,0]*tj*(one_minus**2) +
                        C[i,0]*(tj**2)*one_minus +
                        D[i,0]*beta[i]*(tj**3))
                num1 = (A[i,1]*alpha[i]*(one_minus**3) +
                        B[i,1]*tj*(one_minus**2) +
                        C[i,1]*(tj**2)*one_minus +
                        D[i,1]*beta[i]*(tj**3))
                den = (alpha[i]*(one_minus**3) +
                       tj*(one_minus**2) +
                       (tj**2)*one_minus +
                       beta[i]*(tj**3))
                if den != zero:
                    r[i*seg_pts + j, 0] = num0 / den
                    r[i*seg_pts + j, 1] = num1 / den
                else:
                    # fallback to linear
                    r[i*seg_pts + j, 0] = (one - tj)*A[i,0] + tj*D[i,0]
                    r[i*seg_pts + j, 1] = (one - tj)*A[i,1] + tj*D[i,1]
        else:
            for j in range(seg_pts):
                tj = t_vals[j]
                r[i*seg_pts + j, 0] = (one - tj)*A[i,0] + tj*D[i,0]
                r[i*seg_pts + j, 1] = (one - tj)*A[i,1] + tj*D[i,1]

    return r


# ---------------------
# float64 specialized function
# ---------------------
@njit(parallel=True, fastmath=True)
def curve_goodman_numba_f64(I, seg_pts):
    """
    Goodman-style closed curve smoothing — float32-safe Numba implementation.

    Parameters
    ----------
    I : (no_pts, 2) ndarray (float32)
        Closed polygon points in order (I[0]..I[no_pts-1]), dtype must be np.float32.
    seg_pts : int
        Number of interpolation points per segment (end point excluded).

    Returns
    -------
    r : (no_pts * seg_pts, 2) ndarray (float32)
        Smoothed curve points in float32.
    """
    dtype = np.float64

    no_pts = I.shape[0]

    # constants as dtype
    m = dtype(0.5)
    n = dtype(0.5)
    r_coef = dtype(0.25)
    s_coef = dtype(0.25)
    zero = dtype(0.0)
    one = dtype(1.0)
    two = dtype(2.0)
    eps = dtype(1e-6)

    # build t values in float32 excluding the final endpoint
    t_vals = np.empty(seg_pts, dtype=dtype)
    for j in range(seg_pts):
        # j / seg_pts  in float32
        t_vals[j] = dtype(j) / dtype(seg_pts)

    # allocate arrays (all float32)
    lineleft = np.zeros(no_pts, dtype=np.bool_)
    lineright = np.zeros(no_pts, dtype=np.bool_)
    kleft = np.zeros(no_pts, dtype=dtype)
    kright = np.zeros(no_pts, dtype=dtype)

    # --- curvature flags + curvature values ---
    for i in prange(no_pts):
        im1 = (i - 1) % no_pts
        ip1 = (i + 1) % no_pts
        ip2 = (i + 2) % no_pts
        im2 = (i - 2) % no_pts

        # collinearity checks using cross products (float32)
        lhs1 = (I[i,1] - I[im1,1]) * (I[ip1,0] - I[im1,0]) - (I[ip1,1] - I[im1,1]) * (I[i,0] - I[im1,0])
        if abs(lhs1) < eps:  # tolerance in float32/float64
            lineleft[i] = True
            lineright[i] = True
        else:
            lhs2 = (I[i,1] - I[ip1,1]) * (I[ip2,0] - I[ip1,0]) - (I[ip2,1] - I[ip1,1]) * (I[i,0] - I[ip1,0])
            if abs(lhs2) < eps:
                lineright[i] = True
            else:
                lhs3 = (I[i,1] - I[im1,1]) * (I[im2,0] - I[im1,0]) - (I[im2,1] - I[im1,1]) * (I[i,0] - I[im1,0])
                if abs(lhs3) < eps:
                    lineleft[i] = True

        if not lineleft[i]:
            # compute curvature-like quantity (float32-safe)
            v1 = np.empty(2, dtype=dtype); v2 = np.empty(2, dtype=dtype)
            v1[0] = I[i,0] - I[im1,0]; v1[1] = I[i,1] - I[im1,1]
            v2[0] = I[ip1,0] - I[i,0]; v2[1] = I[ip1,1] - I[i,1]
            num = two * cross2d_float(v1, v2)
            den = vec_norm_float(v1) * vec_norm_float(v2) * vec_norm_float(np.array([I[ip1,0]-I[im1,0], I[ip1,1]-I[im1,1]], dtype=dtype))
            kleft[i] = safe_div_float(num, den, dtype)

        if not lineright[i]:
            v1 = np.empty(2, dtype=dtype); v2 = np.empty(2, dtype=dtype)
            v1[0] = I[i,0] - I[im1,0]; v1[1] = I[i,1] - I[im1,1]
            v2[0] = I[ip1,0] - I[i,0]; v2[1] = I[ip1,1] - I[i,1]
            num = two * cross2d_float(v1, v2)
            den = vec_norm_float(v1) * vec_norm_float(v2) * vec_norm_float(np.array([I[ip1,0]-I[im1,0], I[ip1,1]-I[im1,1]], dtype=dtype))
            kright[i] = safe_div_float(num, den, dtype)

    # --- tangents Tu (unit directions) and T (unnormalized) ---
    T = np.zeros((no_pts, 2), dtype=dtype)
    Tu = np.zeros((no_pts, 2), dtype=dtype)

    for i in prange(no_pts):
        im1 = (i - 1) % no_pts
        ip1 = (i + 1) % no_pts
        if (kleft[i] != 0.0) or (kright[i] != 0.0):
            a = np.abs(kleft[ip1]) * (vec_norm_float(np.array([I[ip1,0]-I[i,0], I[ip1,1]-I[i,1]], dtype=dtype))**2)
            b = np.abs(kright[im1]) * (vec_norm_float(np.array([I[i,0]-I[im1,0], I[i,1]-I[im1,1]], dtype=dtype))**2)
            T_i0 = a * (I[i,0] - I[im1,0]) + b * (I[ip1,0] - I[i,0])
            T_i1 = a * (I[i,1] - I[im1,1]) + b * (I[ip1,1] - I[i,1])
            T[i,0] = T_i0; T[i,1] = T_i1
            normT = vec_norm_float(np.array([T[i,0], T[i,1]], dtype=dtype))
            if normT != zero:
                Tu[i,0] = T[i,0] / normT
                Tu[i,1] = T[i,1] / normT
            else:
                Tu[i,0] = zero; Tu[i,1] = zero

    # --- Bezier control points and lengths (float32) ---
    A = np.zeros((no_pts, 2), dtype=dtype)
    B = np.zeros((no_pts, 2), dtype=dtype)
    C = np.zeros((no_pts, 2), dtype=dtype)
    D = np.zeros((no_pts, 2), dtype=dtype)
    lengthab = np.zeros(no_pts, dtype=dtype)
    lengthcd = np.zeros(no_pts, dtype=dtype)

    for i in prange(no_pts):
        ip1 = (i + 1) % no_pts

        if (kright[i] * kleft[ip1]) > zero:
            # convex case
            T_i = np.array([T[i,0], T[i,1]], dtype=dtype)
            T_ip1 = np.array([T[ip1,0], T[ip1,1]], dtype=dtype)
            L = vec_norm_float(np.array([I[ip1,0] - I[i,0], I[ip1,1] - I[i,1]], dtype=dtype))
            # compute sines safely
            sina = safe_div_float(cross2d_float(T_i, np.array([I[ip1,0]-I[i,0], I[ip1,1]-I[i,1]], dtype=dtype)),
                                vec_norm_float(T_i) * L, dtype)
            sinb = safe_div_float(cross2d_float(np.array([I[ip1,0]-I[i,0], I[ip1,1]-I[i,1]], dtype=dtype), T_ip1),
                                vec_norm_float(T_ip1) * L, dtype)
            sinab = safe_div_float(cross2d_float(T_i, T_ip1), vec_norm_float(T_i) * vec_norm_float(T_ip1), dtype)

            p = safe_div_float(two * np.abs(sinb),
                             two * m * np.abs(sinb) + (one - m) * L * np.abs(kleft[ip1]) + two * np.abs(sinab), dtype)
            q = safe_div_float(two * np.abs(sina),
                             two * n * np.abs(sina) + (one - n) * L * np.abs(kright[i]) + two * np.abs(sinab), dtype)

            lengthab[i] = p * L
            lengthcd[i] = q * L

            A[i,0] = I[i,0]; A[i,1] = I[i,1]
            D[i,0] = I[ip1,0]; D[i,1] = I[ip1,1]
            B[i,0] = A[i,0] + lengthab[i] * Tu[i,0]
            B[i,1] = A[i,1] + lengthab[i] * Tu[i,1]
            C[i,0] = D[i,0] - lengthcd[i] * Tu[ip1,0]
            C[i,1] = D[i,1] - lengthcd[i] * Tu[ip1,1]

        elif (kright[i] * kleft[ip1]) < zero:
            # inflection
            L = vec_norm_float(np.array([I[ip1,0] - I[i,0], I[ip1,1] - I[i,1]], dtype=dtype))
            lengthab[i] = r_coef * L
            lengthcd[i] = s_coef * L

            A[i,0] = I[i,0]; A[i,1] = I[i,1]
            D[i,0] = I[ip1,0]; D[i,1] = I[ip1,1]
            B[i,0] = A[i,0] + lengthab[i] * Tu[i,0]
            B[i,1] = A[i,1] + lengthab[i] * Tu[i,1]
            C[i,0] = D[i,0] - lengthcd[i] * Tu[ip1,0]
            C[i,1] = D[i,1] - lengthcd[i] * Tu[ip1,1]

        else:
            # straight segment
            A[i,0] = I[i,0]; A[i,1] = I[i,1]
            D[i,0] = I[ip1,0]; D[i,1] = I[ip1,1]

    # --- weights alpha and beta (float32) ---
    alpha = np.zeros(no_pts, dtype=dtype)
    beta = np.zeros(no_pts, dtype=dtype)

    for i in prange(no_pts):
        ip1 = (i + 1) % no_pts
        if kright[i] != zero:
            denom1 = two * cross2d_float(B[i] - A[i], C[i] - B[i])
            if denom1 != zero:
                alpha[i] = kright[i] * (lengthab[i]**3) / denom1
            denom2 = two * cross2d_float(C[i] - B[i], D[i] - C[i])
            if denom2 != zero:
                beta[i] = kleft[ip1] * (lengthcd[i]**3) / denom2

    # --- final curve points (float32) ---
    r = np.zeros((no_pts * seg_pts, 2), dtype=dtype)

    for i in prange(no_pts):
        ip1 = (i + 1) % no_pts
        if kright[i] != zero:
            for j in range(seg_pts):
                tj = t_vals[j]
                one_minus = one - tj
                # numerator and denominator computed in float32
                num0 = (A[i,0]*alpha[i]*(one_minus**3) +
                        B[i,0]*tj*(one_minus**2) +
                        C[i,0]*(tj**2)*one_minus +
                        D[i,0]*beta[i]*(tj**3))
                num1 = (A[i,1]*alpha[i]*(one_minus**3) +
                        B[i,1]*tj*(one_minus**2) +
                        C[i,1]*(tj**2)*one_minus +
                        D[i,1]*beta[i]*(tj**3))
                den = (alpha[i]*(one_minus**3) +
                       tj*(one_minus**2) +
                       (tj**2)*one_minus +
                       beta[i]*(tj**3))
                if den != zero:
                    r[i*seg_pts + j, 0] = num0 / den
                    r[i*seg_pts + j, 1] = num1 / den
                else:
                    # fallback to linear
                    r[i*seg_pts + j, 0] = (one - tj)*A[i,0] + tj*D[i,0]
                    r[i*seg_pts + j, 1] = (one - tj)*A[i,1] + tj*D[i,1]
        else:
            for j in range(seg_pts):
                tj = t_vals[j]
                r[i*seg_pts + j, 0] = (one - tj)*A[i,0] + tj*D[i,0]
                r[i*seg_pts + j, 1] = (one - tj)*A[i,1] + tj*D[i,1]

    return r
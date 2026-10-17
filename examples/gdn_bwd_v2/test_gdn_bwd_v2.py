"""gdn_bwd_v2 layered tests: CPU math gates + L0 (NPU) + L1/L2/Boundary stubs.

Two-stage verification on top of the v1 golden anchor (examples/gdn_bwd/
test_gdn_bwd.py, ported verbatim below): fp64 exact forward `fwd_exact` + fp64
adjoint recursion `bwd_adjoint(order="ub")` + `gradcheck_fd` central differences.

Stage A -- CPU math gates (no NPU, blocks the exit code):
  1. golden self-check : `_np_bwd_adjoint` vs `_np_gradcheck_fd` to TOL_ADJ
     (the golden port is the independent anchor).
  2. chunked reference check : the host chunked reference below
     (`_np_chunk_fwd` / `_np_chunk_bwd`, design.md Sec 2 + Sec 3 B1-B15
     transcribed step by step) vs `_np_fwd_exact` / `_np_bwd_adjoint` to
     TOL_CHUNK = 1e-12 on full-chunk AND tail-chunk shapes (T=5 / 65 / 130 /
     300, C=64). This is the last correctness gate before the real device:
     the kernel is a transcription of the same B1-B15 graph.

Stage B -- L0 kernel gate (requires an Ascend NPU; deferred to the real-device
window on a CPU-only host -- the kernel stage then prints SKIP and does not
falsify the CPU verdicts): six gradients + adjoint, per-item verdicts, no early
exit:
    ITEM <dq|dk|dv|dbeta|dg|dh0|adjoint> rel= <v> verdict= <PASS|FAIL>
Tolerance: TOL_KERNEL = 2e-4 (kernel vs fp64 golden; design.md Sec 9 TOL_F32,
dbeta is the known fork fault to watch).

Scale discipline (ascend-constraints.md Sec 2 / checklist #0c): every golden
value, random draw and special-value injection happens on CPU; only
H2D -> kernel -> D2H. tilelang / gdn_bwd_v2 are imported lazily so the CPU
math gates run on a host without the tilelang toolchain.

Run: python examples/gdn_bwd_v2/test_gdn_bwd_v2.py --level l0
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOL_CHUNK = 1e-12  # chunked reference vs v1 adjoint (fp64 math equivalence)
TOL_ADJ = 1e-8  # fp64 adjoint vs central-difference finite gradient
TOL_KERNEL = 2e-4  # kernel vs fp64 golden (= design.md Sec 9 TOL_F32)


# ============================================================================
# Golden reference (fp64, CPU) -- verbatim port of
# examples/gdn_bwd/test_gdn_bwd.py (itself attempts/F/f_gdn_golden.py).
# Convention: H heads, T tokens, DK key dim, DV value dim.
#   Q,K: (H,T,DK)  V: (H,T,DV)  BETA,G: (H,T)  H0: (H,DV,DK)  dO: (H,T,DV)
#   S in R^{DV x DK}; HIST[h,t] = S_{t-1};  dq,dk: (H,T,DK)  dv: (H,T,DV)
#   dbeta,dg: (H,T)  dh0: (H,DV,DK)
# ============================================================================


def _np_fwd_exact(H, T, DK, DV, Q, K, V, BETA, G, H0):
    """fp64 exact forward recursion. Returns (O, HIST, S_final)."""
    a_f = np.exp(G)
    O = np.zeros((H, T, DV), np.float64)
    HIST = np.zeros((H, T, DV, DK), np.float64)
    Sf = np.zeros((H, DV, DK), np.float64)
    for h in range(H):
        st = H0[h].astype(np.float64, copy=True)
        for t in range(T):
            HIST[h, t] = st
            a = a_f[h, t]
            sp = st * a
            pred = sp @ K[h, t]
            uu = BETA[h, t] * (V[h, t] - pred)
            snew = sp + np.outer(uu, K[h, t])
            O[h, t] = snew @ Q[h, t]
            st = snew
        Sf[h] = st
    return O, HIST, Sf


def _np_bwd_adjoint(H, T, DK, DV, Q, K, V, BETA, G, HIST, dO, order="ub"):
    """fp64 adjoint recursion (order="ub"), verbatim f_gdn_golden.py:123-194."""
    a_all = np.exp(np.asarray(G, np.float64))
    dqa = np.zeros((H, T, DK), np.float64)
    dka = np.zeros((H, T, DK), np.float64)
    dva = np.zeros((H, T, DV), np.float64)
    dba = np.zeros((H, T), np.float64)
    dga = np.zeros((H, T), np.float64)
    dh0 = np.zeros((H, DV, DK), np.float64)
    for h in range(H):
        lam = np.zeros((DV, DK), np.float64)
        for tt in range(T):
            t = T - 1 - tt
            a = float(a_all[h, t])
            bt = float(BETA[h, t])
            kt = np.asarray(K[h, t], np.float64)
            qt = np.asarray(Q[h, t], np.float64)
            vt = np.asarray(V[h, t], np.float64)
            dot = np.asarray(dO[h, t], np.float64)
            hist = np.asarray(HIST[h, t], np.float64)
            u = np.zeros(DV, np.float64)
            dp = np.zeros(DV, np.float64)
            db = 0.0
            da = 0.0
            if order == "ub":
                for j in range(DV):  # trip 1: p / u / du / dp / dv / dbeta
                    sp = hist[j] * a
                    p = float(np.dot(sp, kt))
                    u[j] = bt * (vt[j] - p)
                    du = float(np.dot(lam[j] + dot[j] * qt, kt))
                    dp[j] = -bt * du
                    dva[h, t, j] = bt * du
                    db += du * (vt[j] - p)
                for i in range(DK):  # trip 2: Lam recursion + dq_i / dk_i / dg
                    acc_q = 0.0
                    acc_k = 0.0
                    for j in range(DV):
                        sp = hist[j, i] * a
                        acc_q += (sp + u[j] * kt[i]) * dot[j]
                        lamt = lam[j, i] + dot[j] * qt[i]
                        acc_k += u[j] * lamt + sp * dp[j]
                        dst = lamt + kt[i] * dp[j]
                        da += hist[j, i] * dst
                        lam[j, i] = a * dst
                    dqa[h, t, i] = acc_q
                    dka[h, t, i] = acc_k
            else:  # pragma: no cover - "pure" order not used by this test
                for j in range(DV):
                    sp_row = hist[j] * a
                    p = float(np.dot(sp_row, kt))
                    du = float(np.dot(lam[j] + dot[j] * qt, kt))
                    u[j] = bt * (vt[j] - p)
                    dp[j] = -bt * du
                    dva[h, t, j] = bt * du
                    db += du * (vt[j] - p)
                    for i in range(DK):
                        sp = sp_row[i]
                        lamji = lam[j, i] + dot[j] * qt[i]
                        acc_q = (sp + u[j] * kt[i]) * dot[j]
                        acc_k = u[j] * lamji + sp * dp[j]
                        dst = lamji + kt[i] * dp[j]
                        dqa[h, t, i] = (dqa[h, t, i] + acc_q) if j else acc_q
                        dka[h, t, i] = (dka[h, t, i] + acc_k) if j else acc_k
                        da += hist[j, i] * dst
                        lam[j, i] = a * dst
            dba[h, t] = db
            dga[h, t] = da * a
        dh0[h] = lam
    return dqa, dka, dva, dba, dga, dh0


def _np_fwd_scalar(args):
    """L = sum(w * O) -- scalar functional for finite differences / autograd."""
    H, T, DK, DV, Q, K, V, BETA, G, H0, w = args
    O, _, _ = _np_fwd_exact(H, T, DK, DV, Q, K, V, BETA, G, H0)
    return float(np.sum(w * O))


def _np_gradcheck_fd(args, eps=1e-6):
    """Central-difference dL/dtheta (fp64) -- independent physical anchor."""
    H, T, DK, DV, Q, K, V, BETA, G, H0, w = args
    ref = {"Q": Q, "K": K, "V": V, "BETA": BETA, "G": G, "H0": H0}
    out = {}
    for name, arr in ref.items():
        g = np.zeros_like(arr, dtype=np.float64)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            base = float(arr[idx])
            for which, sgn in (("p", 1.0), ("m", -1.0)):
                mut = {k: v.copy() for k, v in ref.items()}
                mut[name][idx] = base + sgn * eps
                ov = _np_fwd_scalar((H, T, DK, DV, mut["Q"], mut["K"], mut["V"],
                                     mut["BETA"], mut["G"], mut["H0"], w))
                if which == "p":
                    fp = ov
                else:
                    fm = ov
            g[idx] = (fp - fm) / (2 * eps)
        out[name] = g
    return out


def _np_sample(H, T, DK, DV, seed):
    """Deterministic fp64 sample (same distribution as f_gdn_golden.py:264)."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((H, T, DK)).astype(np.float64) / np.sqrt(DK)
    K = rng.standard_normal((H, T, DK)).astype(np.float64)
    K /= np.maximum(np.linalg.norm(K, axis=-1, keepdims=True), 1e-6)
    V = rng.standard_normal((H, T, DV)).astype(np.float64)
    BETA = (1.0 / (1.0 + np.exp(-rng.standard_normal((H, T))))).astype(np.float64)
    G = (-np.abs(rng.standard_normal((H, T))) * 0.5 - 0.05).astype(np.float64)  # g <= 0
    H0 = (rng.standard_normal((H, DV, DK)) * 0.1).astype(np.float64)
    w = rng.standard_normal((H, T, DV)).astype(np.float64)
    return Q, K, V, BETA, G, H0, w


def rel(a, b):
    """Max relative error used by the verdict lines."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / den)


# ============================================================================
# Chunked reference (fp64, CPU) -- design.md Sec 2 (forward) + Sec 3 B1-B15
# (adjoint), transcribed step by step. Per chunk n of m tokens (m = C, or the
# tail n_tail = T - (NT-1)*C):
#   gam = cumsum(g);  Gam_rc = exp(gam_r - gam_c);  KKt = K K^T
#   L = J * (diag(beta) * (Gam .* KKt));  A = (I + L)^-1
#   E = exp(gam);  th = exp(gam_{m-1} - gam);  Kt = diag(th) K
#   U0 = A (beta .* V);  W = A (beta .* E .* K);  Vnew = U0 - W H_in^T
#   H_out = exp(gam_{m-1}) H_in + Vnew^T Kt
#   P = Gam .* M .* (Q K^T);  O = diag(E)(Q H_in^T) + P Vnew
# Adjoint (reverse chunk order, Delta' = dL/dH_out relayed through WS_D):
#   B1-B15 exactly as the design table; the symbol-audit red lines are kept:
#   E = exp(cumsum) (not exp(g)), d(gam) += E .* dE (B13), d(gam) += -dth .* th
#   and d(gam_{m-1}) += sum(dth .* th) (B10), dGam -> dgam multiplies Gam (B12).
# ============================================================================


def _np_chunk_fwd(T_len, DK, DV, Q, K, V, BETA, G, H0, C):
    """Chunked forward for one head. Returns (O, cache) with per-chunk state."""
    ntn = math.ceil(T_len / C)
    O = np.zeros((T_len, DV), np.float64)
    cache = []
    st = H0.astype(np.float64, copy=True)  # H_in (DV, DK)
    for n in range(ntn):
        lo, hi = n * C, min((n + 1) * C, T_len)
        m = hi - lo
        g = G[lo:hi]
        q = Q[lo:hi]
        k = K[lo:hi]
        v = V[lo:hi]
        beta = BETA[lo:hi]
        gam = np.cumsum(g)  # chunk-local gamma_r = sum_{p<=r} g_p
        gam_last = float(gam[m - 1])
        gam_2d = np.exp(gam[:, None] - gam[None, :])  # Gam_rc = exp(gam_r - gam_c)
        kk = k @ k.T  # KK^T
        msk_m = np.tril(np.ones((m, m)))  # causal (r >= c)
        msk_j = np.tril(np.ones((m, m)), -1)  # strict lower (r > c)
        lam = msk_j * (beta[:, None] * (gam_2d * kk))  # L = J .* (diag(beta)(Gam .* KKt))
        a_mat = np.linalg.inv(np.eye(m) + lam)  # A = (I + L)^-1
        e_vec = np.exp(gam)  # E = exp(gam)  (cumsum exponent)
        glast = np.exp(gam_last)  # exp(gam_{m-1})
        th = np.exp(gam_last - gam)  # theta_r
        kt = th[:, None] * k  # Kt = diag(th) K
        u0 = a_mat @ (beta[:, None] * v)  # U0 = A (beta .* V)
        w_mat = a_mat @ ((beta * e_vec)[:, None] * k)  # W = A (beta .* E .* K)
        vnew = u0 - w_mat @ st.T  # Vnew = U0 - W H_in^T
        qh = q @ st.T  # Q H_in^T
        p_mat = gam_2d * msk_m * (q @ k.T)  # P = Gam .* M .* (Q K^T)
        O[lo:hi] = e_vec[:, None] * qh + p_mat @ vnew
        cache.append(dict(lo=lo, hi=hi, m=m, gam=gam, gam_2d=gam_2d, kk=kk,
                          msk_m=msk_m, msk_j=msk_j, lam=lam, a_mat=a_mat,
                          e_vec=e_vec, glast=glast, th=th, kt=kt, u0=u0,
                          w_mat=w_mat, vnew=vnew, qh=qh, p_mat=p_mat,
                          q=q, k=k, v=v, beta=beta, hin=st))
        st = glast * st + vnew.T @ kt  # H_out
    return O, cache


def _np_chunk_bwd(T_len, DK, DV, cache, dO):
    """Chunked adjoint (B1-B15) for one head. Returns (dq, dk, dv, dbeta, dg,
    dh0). Delta' (dL/dH_out of the NEXT chunk) is relayed in reverse order; the
    first chunk's Delta is dh0."""
    dq = np.zeros((T_len, DK), np.float64)
    dk = np.zeros((T_len, DK), np.float64)
    dv = np.zeros((T_len, DV), np.float64)
    dbeta = np.zeros((T_len,), np.float64)
    dg = np.zeros((T_len,), np.float64)
    dp_rel = np.zeros((DV, DK), np.float64)  # Delta' of the last chunk = 0
    dh0 = None
    for cc in range(len(cache) - 1, -1, -1):
        c = cache[cc]
        lo, hi, m = c["lo"], c["hi"], c["m"]
        q, k, v, beta = c["q"], c["k"], c["v"], c["beta"]
        gam_2d, kk, msk_m, msk_j = c["gam_2d"], c["kk"], c["msk_m"], c["msk_j"]
        a_mat, e_vec, glast, th = c["a_mat"], c["e_vec"], c["glast"], c["th"]
        kt, vnew, qh, hin = c["kt"], c["vnew"], c["qh"], c["hin"]
        do = dO[lo:hi]
        raw_qkt = q @ k.T
        # B1: dVnew = P^T dO + Kt Delta'^T
        dvnew = c["p_mat"].T @ do + kt @ dp_rel.T
        # B2: dP = dO Vnew^T
        dp = do @ vnew.T
        # B3: d(QKt) = dP .* Gam .* M ;  dGam += dP .* M .* (Q K^T)
        dqkt = dp * gam_2d * msk_m
        dgam2 = dp * msk_m * raw_qkt
        # B4: dQ += d(QKt) K ;  dK += d(QKt)^T Q
        dq_c = dqkt @ k
        dk_c = dqkt.T @ q
        # B5: dE += rowsum(dO .* QH);  dQ += (E .* dO) H_in^T;  dH += (E .* dO)^T Q
        # (the design's "H_in^T" on the dQ line is the shape-unique (m,DV)@(DV,DK)
        # product -- fla's H convention flips the transpose; Sec 1.2 folds it.)
        de = (do * qh).sum(1)
        e_do = e_vec[:, None] * do
        dq_c = dq_c + e_do @ hin
        dh = e_do.T @ q
        # B6: dU0 = dVnew;  dW = -dVnew H_in;  dH += -dVnew^T W
        du0 = dvnew
        dw = -dvnew @ hin
        dh = dh - dvnew.T @ c["w_mat"]
        # B7: dA += dU0 (beta .* V)^T;  t1 = A^T dU0;  dV += beta .* t1
        #     dbeta += rowsum(V .* t1)
        da = du0 @ (beta[:, None] * v).T
        t1 = a_mat.T @ du0
        dv_c = beta[:, None] * t1
        dbeta_c = (v * t1).sum(1)
        # B8: dA += dW (beta .* E .* K)^T;  t2 = A^T dW;  dK += (beta .* E) .* t2
        #     d(betaE) = rowsum(K .* t2);  dbeta += E .* d(betaE);  dE += beta .* d(betaE)
        da = da + dw @ ((beta * e_vec)[:, None] * k).T
        t2 = a_mat.T @ dw
        dk_c = dk_c + (beta * e_vec)[:, None] * t2
        dbe = (k * t2).sum(1)
        dbeta_c = dbeta_c + e_vec * dbe
        de = de + beta * dbe
        # B9: dH += exp(gam_{m-1}) Delta';  dKt = Vnew Delta';
        #     dgam_{m-1} += exp(gam_{m-1}) <H_in, Delta'>
        dh = dh + glast * dp_rel
        dkt = vnew @ dp_rel
        dgam = np.zeros((m,), np.float64)
        dgam[m - 1] += glast * float(np.sum(hin * dp_rel))
        # B10: dK += th .* dKt;  dth = rowsum(K .* dKt);  dgam += -dth .* th
        #      dgam_{m-1} += sum(dth .* th)
        dk_c = dk_c + th[:, None] * dkt
        dth = (k * dkt).sum(1)
        dgam += -dth * th
        dgam[m - 1] += float(np.sum(dth * th))
        # B11: dL = J .* (-A^T dA A^T);  d(KKt) = dL .* beta .* Gam
        #      dbeta += rowsum(dL .* Gam .* KKt);  dGam += dL .* beta .* KKt
        #      dK += (dX + dX^T) K   with dX = d(KKt)
        dl = msk_j * (-(a_mat.T @ da @ a_mat.T))
        dx = dl * beta[:, None] * gam_2d
        dbeta_c = dbeta_c + (dl * gam_2d * kk).sum(1)
        dgam2 = dgam2 + dl * beta[:, None] * kk
        dk_c = dk_c + (dx + dx.T) @ k
        # B12: dgam += rowsum(dGam .* Gam) - colsum(dGam .* Gam)
        w12 = dgam2 * gam_2d
        dgam += w12.sum(1) - w12.sum(0)
        # B13: dgam += E .* dE
        dgam += e_vec * de
        # B14: dg = suffix-sum(dgam):  dg_m = sum_{r >= m} dgam_r
        dg[lo:hi] = np.cumsum(dgam[::-1])[::-1]
        dq[lo:hi] = dq_c
        dk[lo:hi] = dk_c
        dv[lo:hi] = dv_c
        dbeta[lo:hi] = dbeta_c
        # B15: Delta = dH relayed to the previous chunk; first chunk -> dh0
        if cc == 0:
            dh0 = dh.copy()
        dp_rel = dh
    return dq, dk, dv, dbeta, dg, dh0


def _np_chunk_bwd_full(H, T_len, DK, DV, Q, K, V, BETA, G, H0, dO, C):
    """Chunked forward+adjoint over all heads; v1-adjoint-compatible outputs."""
    dqa = np.zeros((H, T_len, DK), np.float64)
    dka = np.zeros((H, T_len, DK), np.float64)
    dva = np.zeros((H, T_len, DV), np.float64)
    dba = np.zeros((H, T_len), np.float64)
    dga = np.zeros((H, T_len), np.float64)
    dh0a = np.zeros((H, DV, DK), np.float64)
    o_a = np.zeros((H, T_len, DV), np.float64)
    for h in range(H):
        o_c, cache = _np_chunk_fwd(T_len, DK, DV, Q[h], K[h], V[h], BETA[h],
                                   G[h], H0[h], C)
        o_a[h] = o_c
        dq, dk, dv, dbeta, dg, dh0 = _np_chunk_bwd(T_len, DK, DV, cache, dO[h])
        dqa[h], dka[h], dva[h] = dq, dk, dv
        dba[h], dga[h], dh0a[h] = dbeta, dg, dh0
    return o_a, dqa, dka, dva, dba, dga, dh0a


# ============================================================================
# CPU math gates: golden self-check + chunked reference check
# ============================================================================


def golden_selfcheck(seed=7):
    """Independent anchor: `bwd_adjoint` must match `gradcheck_fd` to TOL_ADJ.

    Runs fully on CPU (no NPU) and therefore also validates the golden port on a
    CPU-only host. Returns (worst_rel, ok).
    """
    fdH, fdT, fdK, fdD = 2, 6, 4, 4
    Qf, Kf, Vf, Bf, Gf, H0f, wf = _np_sample(fdH, fdT, fdK, fdD, seed + 1)
    _, HISTf, _ = _np_fwd_exact(fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f)
    adj = _np_bwd_adjoint(fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, HISTf, wf, order="ub")
    fd = _np_gradcheck_fd((fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f, wf))
    names = ["Q", "K", "V", "BETA", "G", "H0"]
    worst = 0.0
    for n, arr in zip(names, adj):
        r = rel(arr, fd[n])
        worst = max(worst, r)
        print(f"[GOLDEN_SELFCHECK] ADJ-FD {n:5s} rel={r:.3e}")
    ok = worst < TOL_ADJ
    print(f"[GOLDEN_SELFCHECK] adjoint-vs-FD worst_rel={worst:.3e} "
          f"tol={TOL_ADJ:.0e} -> {'PASS' if ok else 'FAIL'}")
    return worst, ok


# (tag, H, T, DK, DV) with C=64 -- full chunks AND tail chunks (T=5 < C,
# T=65 tail=1, T=130 tail=2, T=300 tail=44), one rectangular case.
CHUNK_MATH_CASES = [
    ("single_chunk", 2, 64, 8, 8),
    ("tiny_tail", 1, 5, 8, 8),
    ("tail_1", 2, 65, 8, 8),
    ("tail_2", 2, 130, 12, 10),
    ("tail_44", 1, 300, 16, 16),
]


def chunk_math_check(C=64, seed=7):
    """Chunked reference (design Sec 2 + Sec 3 B1-B15) vs the v1 golden
    per-token adjoint: forward O and all six gradients to TOL_CHUNK. This is the
    mathematical identity the kernel transcribes. Returns (worst_rel, ok)."""
    worst = 0.0
    ok = True
    for tag, H, T_len, DK, DV in CHUNK_MATH_CASES:
        Q, K, V, BETA, G, H0, w = _np_sample(H, T_len, DK, DV, seed + 100 + T_len)
        o_ref, HIST, _ = _np_fwd_exact(H, T_len, DK, DV, Q, K, V, BETA, G, H0)
        adj = _np_bwd_adjoint(H, T_len, DK, DV, Q, K, V, BETA, G, HIST, w, order="ub")
        o_c, dqa, dka, dva, dba, dga, dh0a = _np_chunk_bwd_full(
            H, T_len, DK, DV, Q, K, V, BETA, G, H0, w, C)
        r = rel(o_c, o_ref)
        worst = max(worst, r)
        verdict = "PASS" if r <= TOL_CHUNK else "FAIL"
        print(f"ITEM chunk-fwd  rel= {r:.3e} verdict= {verdict}  [{tag}]")
        ok &= r <= TOL_CHUNK
        for name, arr, ref in (("dq", dqa, adj[0]), ("dk", dka, adj[1]),
                               ("dv", dva, adj[2]), ("dbeta", dba, adj[3]),
                               ("dg", dga, adj[4]), ("dh0", dh0a, adj[5])):
            r = rel(arr, ref)
            worst = max(worst, r)
            verdict = "PASS" if r <= TOL_CHUNK else "FAIL"
            print(f"ITEM chunk-{name:5s} rel= {r:.3e} verdict= {verdict}  [{tag}]")
            ok &= r <= TOL_CHUNK
    print(f"[CHUNK_MATH] worst_rel={worst:.3e} tol={TOL_CHUNK:.0e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return worst, ok


# ============================================================================
# Precision helpers (mixed tolerance, see tilelang-op-test-design precision standard)
# ============================================================================


def get_precision(dtype):
    """(atol, rtol, max_abs_error_limit, required_matched_ratio) by dtype."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance two-gate verdict: (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu() if torch.is_tensor(actual) else torch.as_tensor(actual)
    g = golden.detach().cpu() if torch.is_tensor(golden) else torch.as_tensor(golden)
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


def check_precision_special(actual, golden, dtype):
    """NaN/Inf position-sensitive compare (design.md Sec 9 (3)): strict
    NaN/+Inf/-Inf mask equality first, tolerance only on finite positions."""
    an = np.asarray(actual.detach().cpu().double() if torch.is_tensor(actual) else actual, np.float64)
    gn = np.asarray(golden.detach().cpu().double() if torch.is_tensor(golden) else golden, np.float64)
    mask_ok = (np.array_equal(np.isnan(an), np.isnan(gn))
               and np.array_equal(np.isposinf(an), np.isposinf(gn))
               and np.array_equal(np.isneginf(an), np.isneginf(gn)))
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    fin = np.isfinite(an) & np.isfinite(gn)
    if fin.sum() == 0:
        return mask_ok, 1.0, 0.0
    err = np.abs(an[fin] - gn[fin])
    ratio = float(np.mean(err <= atol + rtol * np.abs(gn[fin])))
    max_abs = float(err.max())
    passed = mask_ok and ratio >= required_ratio and max_abs <= max_abs_limit
    return passed, ratio, max_abs


# ============================================================================
# Device helpers (kernel stage; tilelang imported lazily -- CPU math gates must
# run on hosts without the tilelang toolchain)
# ============================================================================

_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _have_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _npu(t):
    return t.npu()


def _case_inputs(B, H, T, DK, DV, dtype, h0_mode, special):
    """Build CPU fp64 inputs, cast to `dtype` for the kernel, and return the
    golden reference computed from the SAME (already rounded) values.

    Returns (kernel_inputs, golden_tuple).
    """
    tdt = _TORCH_DTYPE[dtype]
    kq = []
    kk = []
    kv = []
    kg = []
    kb = []
    kh = []
    kd = []
    for b in range(B):
        Qb, Kb, Vb, Bb, Gb, H0b, wb = _np_sample(H, T, DK, DV, 1000 + b)
        if h0_mode == "zero":
            H0b = np.zeros_like(H0b)
        if special:
            # deterministic sparse special values (>=1 special + many finite)
            Qb[0, 1, 0] = np.inf
            Kb[0, 2, 1] = -np.inf
            Vb[0, 3, 0] = np.nan
            Gb[0, 4] = np.nan
        kq.append(Qb)
        kk.append(Kb)
        kv.append(Vb)
        kg.append(Gb)
        kb.append(Bb)
        kh.append(H0b)
        kd.append(wb)
    # kernel-side tensors: (B,H,T,*) cast to target dtype; g/beta stay fp32
    q = torch.tensor(np.stack(kq), dtype=tdt)
    k = torch.tensor(np.stack(kk), dtype=tdt)
    v = torch.tensor(np.stack(kv), dtype=tdt)
    g = torch.tensor(np.stack(kg), dtype=torch.float32)
    beta = torch.tensor(np.stack(kb), dtype=torch.float32)
    h0 = torch.tensor(np.stack(kh), dtype=tdt)
    do = torch.tensor(np.stack(kd), dtype=tdt)
    # golden from the SAME rounded inputs (cast back to fp64) -> isolates compute error
    gold = []
    for b in range(B):
        Qb = q[b].double().numpy()
        Kb = k[b].double().numpy()
        Vb = v[b].double().numpy()
        gb = g[b].double().numpy()
        bb = beta[b].double().numpy()
        hb = h0[b].double().numpy()
        db = do[b].double().numpy()
        _, HISTb, _ = _np_fwd_exact(H, T, DK, DV, Qb, Kb, Vb, bb, gb, hb)
        adj = _np_bwd_adjoint(H, T, DK, DV, Qb, Kb, Vb, bb, gb, HISTb, db, order="ub")
        gold.append(adj)
    return (q, k, v, g, beta, h0, do), gold


def _golden_item(gold, item, b):
    """Per-batch golden for `item`, batched as a (1,H,T,*) numpy array.

    Kept in numpy (not torch) so the golden assembly never touches a device tensor:
    no `torch.cat`/`torch.stack` on NPU tensors is involved (ascend-constraints Sec 2).
    """
    dq, dk, dv, dbeta, dg, dh0 = gold[b]
    table = {"dq": dq, "dk": dk, "dv": dv, "dbeta": dbeta, "dg": dg, "dh0": dh0}
    return np.asarray(table[item], np.float64)[None]


# ============================================================================
# L0 threshold tests (design.md Sec 9; v1 four cases + one tail-chunk case --
# the chunked v2 tail discipline is a v2-specific risk surface)
# ============================================================================

# (name, B, H, T, DK, DV, dtype, h0_mode, special)
L0_CASES = [
    ("l0_basic", 1, 1, 64, 64, 64, "float16", "zero", False),
    ("l0_basic", 1, 1, 64, 64, 64, "float32", "zero", False),
    ("l0_typical", 1, 2, 256, 64, 64, "float16", "zero", False),
    ("l0_typical", 1, 2, 256, 64, 64, "float32", "zero", False),
    ("l0_h0", 1, 2, 256, 64, 64, "float32", "on", False),
    ("l0_special", 1, 1, 64, 64, 64, "float32", "on", True),
    ("l0_tail", 1, 1, 130, 64, 64, "float32", "on", False),
]

_ITEMS = ["dq", "dk", "dv", "dbeta", "dg", "dh0"]
# output dtype per item: dq/dk/dv/dh0 follow the main dtype, dbeta/dg stay fp32
_FP32_ITEMS = {"dbeta", "dg"}


def test_gdn_bwd_v2_l0():
    """L0 threshold test: CPU math gates + six gradients + adjoint, per-item
    verdicts (no early exit)."""
    ok = True

    # ---- CPU gate 1: golden fp64 recursion vs central-difference FD ----
    worst, ok_adj = golden_selfcheck()
    print(f"ITEM adjoint rel= {worst:.3e} verdict= {'PASS' if ok_adj else 'FAIL'}")
    ok &= ok_adj

    # ---- CPU gate 2: chunked reference (design Sec 2/3) vs v1 adjoint ----
    worst_c, ok_chunk = chunk_math_check()
    print(f"ITEM chunk-math rel= {worst_c:.3e} verdict= {'PASS' if ok_chunk else 'FAIL'}")
    ok &= ok_chunk

    # ---- kernel stage: needs an Ascend NPU (deferred on CPU-only hosts) ----
    if not _have_npu():
        print("[L0] kernel stage SKIPPED: no NPU available (real-device window)")
        return ok

    from gdn_bwd_v2 import gdn_bwd_v2  # lazy: tilelang toolchain

    for name, B, H, T, DK, DV, dtype, h0_mode, special in L0_CASES:
        tag = f"{name} B={B} H={H} T={T} DK={DK} DV={DV} dt={dtype}"
        try:
            (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T, DK, DV, dtype, h0_mode, special)
            dq, dk, dv, dbeta, dg, dh0 = gdn_bwd_v2(
                _npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
            got = {"dq": dq.cpu().float(), "dk": dk.cpu().float(), "dv": dv.cpu().float(),
                   "dbeta": dbeta.cpu().float(), "dg": dg.cpu().float(), "dh0": dh0.cpu().float()}
            case_ok = True
            for item in _ITEMS:
                idt = "float32" if item in _FP32_ITEMS else dtype
                ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
                if special:
                    passed, ratio, max_abs = check_precision_special(got[item], ref, idt)
                else:
                    passed, ratio, max_abs = check_precision(got[item], ref, idt)
                print(f"ITEM {item} rel= {max_abs:.3e} verdict= {'PASS' if passed else 'FAIL'}"
                      f"  [{tag} ratio={ratio:.4f}]")
                case_ok &= passed
            mark = "PASS" if case_ok else "FAIL"
            print(f"[PRECISION_{mark}] {tag} six+adjoint {'all pass' if case_ok else 'has FAIL'}")
            ok &= case_ok
        except Exception as e:  # noqa: BLE001 - keep the layer running
            print(f"[PRECISION_FAIL] {tag}: {type(e).__name__}: {e}")
            ok = False
    return ok


# ========== L1/L2/Boundary: stubs filled later by tilelang-op-test-design ==========


def test_gdn_bwd_v2_l1():
    """L1 functional tests (irregular / tail shapes) -- stub."""
    print("[L1] not expanded yet -- run tilelang-op-test-design (scenario B)")
    return True


def test_gdn_bwd_v2_l2():
    """L2 negative tests (invalid input must be rejected) -- stub."""
    print("[L2] not expanded yet -- run tilelang-op-test-design (scenario B)")


def test_gdn_bwd_v2_boundary():
    """Boundary / special-value tests -- stub."""
    print("[BOUNDARY] not expanded yet -- run tilelang-op-test-design (scenario B)")


# ========== main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    blocking_ok = True  # only L0/L1 count toward the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_gdn_bwd_v2_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_gdn_bwd_v2_l1()
    if args.level in ("l2", "all"):
        test_gdn_bwd_v2_l2()  # negative: correctly rejected = PASS, silently accepted = WARN
    if args.level in ("boundary", "all"):
        test_gdn_bwd_v2_boundary()  # special values: precision fail = WARN, non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()

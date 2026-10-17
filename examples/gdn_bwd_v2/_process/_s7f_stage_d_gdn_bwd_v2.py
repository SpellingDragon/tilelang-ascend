"""gdn_bwd_v2 -- chunked gated delta rule backward (six gradients) on Ascend NPU.

Chunked adjoint decomposition (design.md Sec 2/Sec 3, B1-B15): the sequence is
cut into chunks of C=64 tokens; per-chunk replay (forward) and reverse-scan
(adjoint) passes exchange only chunk-level state through GM workspace.

Layout convention (design.md Sec 1.2, same as v1 golden): S in R^{DV x DK},
HIST[h,t] = S_{t-1}, h0/dh0: (B,H,DV,DK).

Hard constraints (design.md "three hard constraints", all enforced here):
  H1  every j-contraction is T.gemm_v0 (incl. ones-GEMM row/col sums); the only
      reduce_sum contraction is the solve_tril forward-substitution (verbatim
      fwd-6 idiom gdn_solve_tril.py:53-57, verdict-matrix #5 pure-2D reduce) and
      the two [1] scalar reductions (B9 inner product / B10 sum-r, design Sec 5.1
      explicitly-allowed alternative, verdict #3).
  H2  no per-token scalar coefficients: all coefficients are whole-block tile ops
      (T.tile.mul row-broadcast idiom gdn_wy_fast.py:65); the only chunk-level
      scalar exp(gam_{n-1}) uses a compile-time-constant offset (gdn_chunk_h.py:112
      idiom), with a compile-time branch for the tail chunk.
  H3  pass_configs all-True (auto CV/sync/memory-planning/tail-mask), cross-chunk
      recurrence state ONLY through GM workspace (WS_H / WS_D), zero hand-written
      set_cross_flag / wait_cross_flag.

Kernel split (S7 implementation ruling, Q6 handoff 3-2; S7e ruling splits the
scan into two kernels so each pass keeps a pass-local UB peak -- that also
dissolves the memory-planning KILL-DISPLACED failure, see _s7d_handoff.md Sec 2
and Sec 5):
  1. gdn_bwd_v2_cumsum : G -> WS_G (chunk-local cumsum of g, fwd gdn_chunk_cumsum
     idiom; H-safe grid B*H*chunk_num -- the fwd VEC_NUM grid degenerates to 0
     blocks at H=1 which is the L0 baseline case).
  2. gdn_bwd_v2_solve_a : K, beta, WS_G -> WS_A = (I+L)^{-1}, L = J*(diag(beta) *
     (Gamma .* KK^T)) (merged kkt+solve_tril: keeps L on-chip, zero intermediate
     GM, zero cross flag; fwd kkt/solve_tril arithmetic idioms preserved).
  3. gdn_bwd_v2_scan1  : pass 1 (replay) over chunks -> WS_H (H_in/H_out chain;
     design Sec 2). GEMM-chain cube<->vec handoffs go through explicit
     workspace_* GM buffers (L0C->GM->UB / UB->GM->L1,
     example_gqa_sink_bwd_bhsd.py:170/175/210/213 idiom; S7c: the implicit
     L0C<->UB / UB->L1 direct-copy form broke the CombineCV workspace sync
     audit -- see design.md Sec 7 erratum).
  4. gdn_bwd_v2_scan2  : pass 2 (reverse adjoint, B1-B15; design Sec 3) -> the
     six gradients. Cross-kernel state (WS_G / WS_A / WS_H / WS_D) travels in
     GM only (H3); the same workspace_* CV-handoff idiom as scan1. Vnew/P/QH/A
     are recomputed inside the pass (design Sec 8.1: "not stored, recomputed
     with 2-3 GEMMs").

Tail chunks (n_tail = L - (NT-1)*C < C): zero-fill discipline -- every UB tile is
T.tile.fill(0.0)-first, GM loads are valid-extent clamped by TL_ASCEND_TAIL_MASK,
so the invalid region stays zero and cannot pollute any contraction; chunk-level
scalar placements (exp(gam_{n-1}) source cell, dgam[n-1] accumulation) use a
compile-time `if c == NT - 1` branch with compile-time-constant offsets.

dtype (proto.yaml): main dtype for q/k/v/h0/do -> dq/dk/dv/dh0 (float16|float32),
g/beta/dbeta/dg float32, all accumulation float32. WS_H/WS_A follow the main
dtype (fp16 case matches proto; the fp32 case keeps the A/dH precision chain in
fp32 -- see S7 handoff, deviation note). The ones-GEMM constants (OnesDV/OnesDK/
OnesC and their L1 copies) follow the main dtype too (S7e erratum: gemm_v0 needs
A/B to share T1, tl_templates/ascend/common.h:1204; the values are exactly 1 so
no precision is lost). bf16 not supported (ruling 3);
variable-length seq_lens not supported (ruling 4).

Sequence length L is a compile-time parameter (v1 + fwd-6 precedent; design
Sec 12-4 "T symbolic" is overridden by the fwd-6 precedent per the wave rule
"design pseudocode vs fwd-6 precedent conflict -> precedent wins", reported in
the S7 handoff). NT = ceil(L/C) and n_tail are therefore compile-time constants,
which makes all tail-chunk scalar placements compile-time-constant offsets.
"""

import tilelang
from tilelang import language as T

# Developer + combineCV: design.md Sec 7 -- all five keys True (H3).
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}

# [REORDER-COST-AUDIT] (design.md Sec 8/Sec 11 cost ledger, landed as-is)
#   GM pass      : inputs read once per chunk trip (q/k/v/do C*DK*B bytes each;
#                  g/beta C*4B; h0 once), WS_H written NT times (replay) + read
#                  NT times (reverse), WS_A written once (solve_a) + read NT
#                  times, WS_G written once + read NT times, WS_D NT writes + NT
#                  reads (Delta relay), six gradients written once per chunk.
#   DMA bytes    : per chunk at DK=DV=128 -- q/k/do 16KiB, v 16KiB, WS_H 32KiB,
#                  WS_A 8KiB, WS_D 64KiB; all region copies are chunk-aligned
#                  contiguous blocks (no per-element strided GM access).
#   GM scalar    : 0 -- no per-token scalar GM loads/stores (H2; contrast v1's
#                  2 loads + 2 stores per token trip).
#   div/mod      : 0 runtime div/mod on data paths (block decode is compile-time
#                  integer arithmetic on the task axis).
#   active cores : NB = B*H blocks; per-core serial work = 2*NT chunk trips
#                  (replay NT + reverse NT).
#   workspace    : WS_H B*H*(NT+1)*DV*DK + WS_A B*H*NT*C*C + WS_G B*H*L +
#                  WS_D B*H*DV*DK -- 82.00 MiB at the true-shape profile
#                  (B1 H16 DK128 DV128 L8192), vs 16.0 GiB for v1 (design 8.2).


# ---------------------------------------------------------------------------
# 1) gdn_bwd_v2_cumsum: G -> WS_G, chunk-local cumsum (design Sec 2: gam_r =
#    sum_{m<=r, same chunk} g_m; fwd gdn_chunk_cumsum.py:32-40 idiom verbatim,
#    grid made H-safe: B*H*chunk_num, no VEC_NUM split).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gdn_bwd_v2_cumsum(B, H, L, C):
    chunk_num = T.ceildiv(L, C)

    @T.prim_func
    def main(
        G: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore
    ):
        with T.Kernel(B * H * chunk_num, threads=1, is_npu=True) as (cid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % H
            bz = (cid // chunk_num) // H

            g_ub = T.alloc_ub([C], "float")
            s_ub = T.alloc_ub([C], "float")

            T.tile.fill(s_ub, 0.0)
            T.copy(G[bz, by, bx * C], g_ub)
            # serial chain scan (verdict-matrix #7 PASS; tail cells read as 0
            # via the tail-mask valid extent -> gamma keeps its last value).
            for i in range(C):
                if i > 0:
                    s_ub[i] = s_ub[i - 1]
                tmp = s_ub[i] + g_ub[i]
                s_ub[i] = tmp
            T.copy(s_ub, S[bz, by, bx * C])

    return main


# ---------------------------------------------------------------------------
# 2) gdn_bwd_v2_solve_a: A = (I + L)^{-1}, L = J * (diag(beta) * (Gam .* KK^T))
#    (merged fwd kkt + solve_tril arithmetic, one kernel, L stays on-chip).
#    GM->L1 loads always stage through a zero-filled UB tile so the tail region
#    is pad-zero (garbage-free) before any GEMM sees it.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gdn_bwd_v2_solve_a(B, H, L, DK, C, BK, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    bk_num = T.ceildiv(DK, BK)

    @T.prim_func
    def main(
        K: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        Beta: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore
        MskJ: T.Tensor([C, C], "float"),  # type: ignore
        IDT: T.Tensor([C, C], "float"),  # type: ignore
        A: T.Tensor([B, H, L, C], dtype),  # type: ignore
    ):
        with T.Kernel(B * H * chunk_num, threads=1, is_npu=True) as (cid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % H
            bz = (cid // chunk_num) // H

            k_l1 = T.alloc_L1([C, BK], dtype)
            kk_l0 = T.alloc_L0C([C, C], accum_dtype)

            k_half = T.alloc_ub([C, BK], dtype)
            a_half = T.alloc_ub([C, C], dtype)
            kk_ub = T.alloc_ub([C, C], accum_dtype)  # KK^T -> L (in-place)
            o_ub = T.alloc_ub([C, C], accum_dtype)  # substitution target -> A
            mul_ub = T.alloc_ub([C, C], accum_dtype)
            red_ub = T.alloc_ub([C], accum_dtype)
            i_ub = T.alloc_ub([C, C], accum_dtype)
            gm_ub = T.alloc_ub([C, C], accum_dtype)  # Gamma
            msk_ub = T.alloc_ub([C, C], accum_dtype)  # J (strict lower)
            be_ub = T.alloc_ub([C], accum_dtype)
            g_ub = T.alloc_ub([C], accum_dtype)

            # ---- KK^T = K K^T, BK-blocked GEMM (fwd kkt idiom) ----
            T.tile.fill(k_half, 0.0)
            for i in T.serial(bk_num):
                T.copy(K[bz, by, bx * C, i * BK], k_half)
                T.copy(k_half, k_l1)
                T.gemm_v0(k_l1, k_l1, kk_l0, transpose_B=True, init=(i == 0))
            T.copy(kk_l0, kk_ub)

            # ---- Gam_rc = exp(gam_r - gam_c) (fwd kkt exp idiom) ----
            T.tile.fill(g_ub, 0.0)
            T.tile.fill(be_ub, 0.0)
            T.copy(S[bz, by, bx * C], g_ub)
            T.copy(Beta[bz, by, bx * C], be_ub)
            T.copy(MskJ[0, 0], msk_ub)
            for (i, j) in T.Parallel(C, C):
                gm_ub[i, j] = g_ub[i] - g_ub[j]
            for (i, j) in T.Parallel(C, C):
                gm_ub[i, j] = T.exp(gm_ub[i, j])

            # ---- L = J .* (beta .* (Gam .* KK^T))  [row broadcast = H2 idiom] ----
            for (i, j) in T.Parallel(C, C):
                kk_ub[i, j] = kk_ub[i, j] * gm_ub[i, j]
            for (i, j) in T.Parallel(C, C):
                kk_ub[i, j] = kk_ub[i, j] * be_ub[i]
            for (i, j) in T.Parallel(C, C):
                kk_ub[i, j] = kk_ub[i, j] * msk_ub[i, j]

            # ---- forward substitution, A = I - L*  (gdn_solve_tril.py:51-59
            #      verbatim; reduce_sum(dim=0) is the fwd-6 precedent exemption,
            #      pure-2D reduce, verdict-matrix #5) ----
            T.copy(IDT[0, 0], i_ub)
            T.tile.fill(mul_ub, 0.0)
            T.copy(kk_ub, o_ub)
            for i in range(2, C):
                T.tile.fill(red_ub, 0.0)
                for j, k in T.Parallel(C, C):
                    mul_ub[j, k] = o_ub[j, k] * o_ub[i, j]
                T.reduce_sum(mul_ub, red_ub, dim=0)
                for j in T.Parallel(C):
                    o_ub[i, j] = o_ub[i, j] - red_ub[j]
            for i, j in T.Parallel(C, C):
                o_ub[i, j] = i_ub[i, j] - o_ub[i, j]

            T.copy(o_ub, a_half)
            T.copy(a_half, A[bz, by, bx * C, 0])

    return main


# ---------------------------------------------------------------------------
# 3) gdn_bwd_v2_scan1: pass 1 (replay) over chunks -> WS_H (design Sec 2).
#    S7e split of the former single scan kernel (handoff _s7d_handoff.md Sec 5):
#    one kernel per pass keeps each UB peak pass-local -- which also dissolves
#    the memory-planning KILL-DISPLACED failure (handoff Sec 2; the displaced
#    cross-pass kill intervals no longer exist). Cross-kernel state
#    (WS_G / WS_A / WS_H) travels in GM. GEMM-chain cube<->vec handoffs go
#    through explicit workspace_* GM buffers (L0C->GM->UB / UB->GM->L1,
#    gqa_sink_bwd workspace idiom); large (DV,DK) tiles are processed in
#    BV=64 row blocks (design 8.3 L0C/UB budget ruling).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[6], workspace_idx=[7, 8, 9, 10, 11, 12],
              pass_configs=pass_configs)
def gdn_bwd_v2_scan1(B, H, L, DK, DV, C, BV, BK, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    bv_num = T.ceildiv(DV, BV)
    n_tail = L - (chunk_num - 1) * C  # compile-time constant (L is compile-time)

    @T.prim_func
    def main(
        K: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        V: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        Beta: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore  WS_G (cumsum gam)
        A: T.Tensor([B, H, L, C], dtype),  # type: ignore  WS_A (solve_a)
        H0: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
        WS_H: T.Tensor([B, H, chunk_num + 1, DV, DK], dtype),  # type: ignore  workspace
        workspace_vdk: T.Tensor([B * H, C, DK], dtype),  # type: ignore  workspace
        workspace_ldk: T.Tensor([B * H, C, DK], accum_dtype),  # type: ignore  workspace
        workspace_lcb: T.Tensor([B * H, C, BV], accum_dtype),  # type: ignore  workspace
        workspace_vcb: T.Tensor([B * H, C, BV], dtype),  # type: ignore  workspace
        workspace_lbd: T.Tensor([B * H, BV, DK], accum_dtype),  # type: ignore  workspace
        workspace_vbd: T.Tensor([B * H, BV, DK], dtype),  # type: ignore  workspace
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H

            # ---- L1 / L0C (Cube) ----
            a_l1 = T.alloc_L1([C, C], dtype)
            bek_l1 = T.alloc_L1([C, DK], dtype)
            kt_l1 = T.alloc_L1([C, DK], dtype)
            w_l1 = T.alloc_L1([C, DK], dtype)
            hin_l1 = T.alloc_L1([BV, DK], dtype)
            bv_l1 = T.alloc_L1([C, BV], dtype)  # beta .* V (U0 GEMM input)
            vnew_l1 = T.alloc_L1([C, BV], dtype)  # Vnew (H_out GEMM input)
            dw1_l0 = T.alloc_L0C([C, DK], accum_dtype)  # replay dW
            u01_l0 = T.alloc_L0C([C, BV], accum_dtype)  # replay U0
            wh1_l0 = T.alloc_L0C([C, BV], accum_dtype)  # replay WH
            hout_l0 = T.alloc_L0C([BV, DK], accum_dtype)

            # ---- UB (Vector, fp32 accumulation) ----
            g_ub = T.alloc_ub([C], accum_dtype)  # gam
            e_ub = T.alloc_ub([C], accum_dtype)  # E = exp(gam)
            th_ub = T.alloc_ub([C], accum_dtype)  # theta = exp(gam_last - gam)
            zero_ub = T.alloc_ub([C], accum_dtype)  # negation source (fwd chunk_h :80/:92)
            be_ub = T.alloc_ub([C], accum_dtype)  # beta
            bee_ub = T.alloc_ub([C], accum_dtype)  # beta .* E
            glast_ub = T.alloc_ub([8], accum_dtype)  # exp(gam_{n-1})
            xk1_ub = T.alloc_ub([C, DK], accum_dtype)  # kt/bek/w staging
            bq1_ub = T.alloc_ub([C, BV], accum_dtype)  # do/betaV staging
            wh1_ub = T.alloc_ub([C, BV], accum_dtype)  # Vnew staging
            hin_ub = T.alloc_ub([BV, DK], accum_dtype)  # H_in block -> H_out block
            dh1_ub = T.alloc_ub([BV, DK], accum_dtype)  # H_out lane
            # main-dtype staging for GM<->UB handoffs (dtype-matched copies)
            io_xk1 = T.alloc_ub([C, DK], dtype)
            io_bq1 = T.alloc_ub([C, BV], dtype)
            io_hv1 = T.alloc_ub([BV, DK], dtype)

            # ================= pass 1: replay -> WS_H =================
            # H0 -> WS_H[.., 0] (h0 != 0 natively supported, design Sec 12-6)
            for bv in T.serial(bv_num):
                T.tile.fill(hin_ub, 0.0)
                T.copy(H0[b, h, bv * BV, 0], io_hv1)
                T.copy(io_hv1, hin_ub)
                T.copy(hin_ub, io_hv1)
                T.copy(io_hv1, WS_H[b, h, 0, bv * BV, 0])
            for c in T.serial(chunk_num):
                # gam / beta block (tail cells pad-zero via valid extent)
                T.tile.fill(g_ub, 0.0)
                T.tile.fill(be_ub, 0.0)
                T.tile.fill(zero_ub, 0.0)
                T.copy(S[b, h, c * C], g_ub)
                T.copy(Beta[b, h, c * C], be_ub)
                # E = exp(gam); glast = exp(gam_{n-1}) (H2 compile-time offset)
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                if c == chunk_num - 1:
                    T.copy(g_ub[n_tail - 1:n_tail], glast_ub[0:1])
                else:
                    T.copy(g_ub[C - 1:C], glast_ub[0:1])
                T.tile.exp(glast_ub, glast_ub)
                # theta_r = exp(gam_last - gam_r) (fwd chunk_h :89-94 3-loop idiom:
                # diff -> zero_ub negation -> pure-load exp; the compound form
                # T.exp(0.0 - x) survives scalar and dies at codegen -- S7d erratum)
                for i in T.Parallel(C):
                    th_ub[i] = g_ub[i] - g_ub[C - 1]
                if c == chunk_num - 1:
                    for i in T.Parallel(C):
                        th_ub[i] = g_ub[i] - g_ub[n_tail - 1]
                for i in T.Parallel(C):
                    th_ub[i] = zero_ub[i] - th_ub[i]
                for i in T.Parallel(C):
                    th_ub[i] = T.exp(th_ub[i])
                # bee = beta .* E (row-broadcast whole-block form, wy_fast idiom)
                for i in T.Parallel(C):
                    bee_ub[i] = be_ub[i] * e_ub[i]
                # Kt = diag(theta) K -> kt_l1 ; beK = bee .* K -> bek_l1
                # (K read twice; both derivations are row-broadcast tile ops)
                T.tile.fill(xk1_ub, 0.0)
                T.copy(K[b, h, c * C, 0], io_xk1)
                T.copy(io_xk1, xk1_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk1_ub[i, j] = xk1_ub[i, j] * th_ub[i]
                T.copy(xk1_ub, io_xk1)
                T.copy(io_xk1, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], kt_l1)
                T.tile.fill(xk1_ub, 0.0)
                T.copy(K[b, h, c * C, 0], io_xk1)
                T.copy(io_xk1, xk1_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk1_ub[i, j] = xk1_ub[i, j] * bee_ub[i]
                T.copy(xk1_ub, io_xk1)
                T.copy(io_xk1, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], bek_l1)
                # W = A (beta .* E .* K) -> w_l1
                T.copy(A[b, h, c * C, 0], a_l1)
                T.gemm_v0(a_l1, bek_l1, dw1_l0, init=True)
                T.copy(dw1_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk1_ub)
                T.copy(xk1_ub, io_xk1)
                T.copy(io_xk1, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], w_l1)
                # per BV block: U0 / Vnew / H_out (GEMM chain fully on-chip)
                for bv in T.serial(bv_num):
                    # U0_bv = A (beta .* V)_bv
                    T.tile.fill(bq1_ub, 0.0)
                    T.copy(V[b, h, c * C, bv * BV], io_bq1)
                    T.copy(io_bq1, bq1_ub)
                    for (i, j) in T.Parallel(C, BV):
                        bq1_ub[i, j] = bq1_ub[i, j] * be_ub[i]
                    T.copy(bq1_ub, io_bq1)
                    T.copy(io_bq1, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bv_l1)
                    T.gemm_v0(a_l1, bv_l1, u01_l0, init=True)
                    T.copy(u01_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], bq1_ub)
                    # WH_bv = W @ H_in_bv^T ; Vnew_bv = U0 - WH
                    T.tile.fill(hin_ub, 0.0)
                    T.copy(WS_H[b, h, c, bv * BV, 0], io_hv1)
                    T.copy(io_hv1, hin_ub)
                    T.copy(hin_ub, io_hv1)
                    T.copy(io_hv1, workspace_vbd[cid, :, :])
                    T.copy(workspace_vbd[cid, :, :], hin_l1)
                    T.gemm_v0(w_l1, hin_l1, wh1_l0, transpose_B=True, init=True)
                    T.copy(wh1_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], wh1_ub)
                    for (i, j) in T.Parallel(C, BV):
                        wh1_ub[i, j] = bq1_ub[i, j] - wh1_ub[i, j]
                    # H_out_bv = exp(gam_{n-1}) H_in + Vnew^T Kt
                    T.copy(wh1_ub, io_bq1)
                    T.copy(io_bq1, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], vnew_l1)
                    T.gemm_v0(vnew_l1, kt_l1, hout_l0, transpose_A=True, init=True)
                    for (i, j) in T.Parallel(BV, DK):
                        hin_ub[i, j] = hin_ub[i, j] * glast_ub[0]
                    T.copy(hout_l0, workspace_lbd[cid, :, :])
                    T.copy(workspace_lbd[cid, :, :], dh1_ub)
                    for (i, j) in T.Parallel(BV, DK):
                        hin_ub[i, j] = hin_ub[i, j] + dh1_ub[i, j]
                    T.copy(hin_ub, io_hv1)
                    T.copy(io_hv1, WS_H[b, h, c + 1, bv * BV, 0])

    return main


# ---------------------------------------------------------------------------
# 4) gdn_bwd_v2_scan2: pass 2 (reverse adjoint, B1-B15; design Sec 3) -> the
#    six gradients. S7e split sibling of scan1 (handoff _s7d_handoff.md Sec 5):
#    reads the cross-kernel state WS_G / WS_A / WS_H from GM, relays Delta
#    through WS_D (H3-2); same workspace_* CV-handoff idiom as scan1.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[14, 15, 16, 17, 18, 19],
              workspace_idx=[13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
              pass_configs=pass_configs)
def gdn_bwd_v2_scan2(B, H, L, DK, DV, C, BV, BK, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    bv_num = T.ceildiv(DV, BV)
    n_tail = L - (chunk_num - 1) * C  # compile-time constant (L is compile-time)

    @T.prim_func
    def main(
        Q: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        K: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        V: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        DO: T.Tensor([B, H, L, DV], dtype),  # type: ignore  (S7e: handoff Sec 5 table omission, B1/B2/B5 input)
        Beta: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore  WS_G (cumsum gam)
        A: T.Tensor([B, H, L, C], dtype),  # type: ignore  WS_A (solve_a)
        MskM: T.Tensor([C, C], "float"),  # type: ignore
        MskJ: T.Tensor([C, C], "float"),  # type: ignore
        OnesDV: T.Tensor([DV, 16], dtype),  # type: ignore  ones follow main dtype (S7e)
        OnesDK: T.Tensor([DK, 16], dtype),  # type: ignore
        OnesC: T.Tensor([C, 16], dtype),  # type: ignore
        WS_H: T.Tensor([B, H, chunk_num + 1, DV, DK], dtype),  # type: ignore  scan1 out (GM)
        WS_D: T.Tensor([B, H, DV, DK], "float"),  # type: ignore  workspace
        DQ: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        DKG: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        DVG: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        DBeta: T.Tensor([B, H, L], "float"),  # type: ignore
        DG: T.Tensor([B, H, L], "float"),  # type: ignore
        DH0: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
        workspace_ldk: T.Tensor([B * H, C, DK], accum_dtype),  # type: ignore  workspace
        workspace_lcc: T.Tensor([B * H, C, C], accum_dtype),  # type: ignore  workspace
        workspace_lcb: T.Tensor([B * H, C, BV], accum_dtype),  # type: ignore  workspace
        workspace_lbd: T.Tensor([B * H, BV, DK], accum_dtype),  # type: ignore  workspace
        workspace_lc16: T.Tensor([B * H, C, 16], accum_dtype),  # type: ignore  workspace
        workspace_vdk: T.Tensor([B * H, C, DK], dtype),  # type: ignore  workspace
        workspace_vcc: T.Tensor([B * H, C, C], dtype),  # type: ignore  workspace
        workspace_vcb: T.Tensor([B * H, C, BV], dtype),  # type: ignore  workspace
        workspace_vbd: T.Tensor([B * H, BV, DK], dtype),  # type: ignore  workspace
        WS_DA: T.Tensor([B, H, C, C], accum_dtype),  # type: ignore  workspace (S7f-D: dA GM accumulator)
        WS_DG2: T.Tensor([B, H, C, C], accum_dtype),  # type: ignore  workspace (S7f-D: dGam GM accumulator)
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H

            # ---- L1 / L0C (Cube) ----
            k_l1 = T.alloc_L1([C, DK], dtype)
            q_l1 = T.alloc_L1([C, DK], dtype)
            a_l1 = T.alloc_L1([C, C], dtype)
            bek_l1 = T.alloc_L1([C, DK], dtype)
            kt_l1 = T.alloc_L1([C, DK], dtype)
            w_l1 = T.alloc_L1([C, DK], dtype)
            hin_l1 = T.alloc_L1([BV, DK], dtype)
            dhp_l1 = T.alloc_L1([BV, DK], dtype)
            p_l1 = T.alloc_L1([C, C], dtype)
            da_l1 = T.alloc_L1([C, C], dtype)  # dA / d(QKt) / staged dsum GEMM input
            bq_l1 = T.alloc_L1([C, BV], dtype)  # do / dvnew ... lane
            vnew_l1 = T.alloc_L1([C, BV], dtype)  # Vnew (B2/B9 GEMM input)
            t1_l1 = T.alloc_L1([C, BV], dtype)  # de/dbeta ones-GEMM input
            bv_l1 = T.alloc_L1([C, BV], dtype)  # beta .* V (B7 dA input)
            xk_l1 = T.alloc_L1([C, DK], dtype)  # dW / dKt products / dsum lane
            onesbv_l1 = T.alloc_L1([BV, 16], dtype)  # rowsum over BV cols (S7e: main dtype)
            onesdk_l1 = T.alloc_L1([DK, 16], dtype)
            onesc_l1 = T.alloc_L1([C, 16], dtype)
            kk_l0 = T.alloc_L0C([C, C], accum_dtype)
            u0_l0 = T.alloc_L0C([C, BV], accum_dtype)
            wh_l0 = T.alloc_L0C([C, BV], accum_dtype)
            qh_l0 = T.alloc_L0C([C, BV], accum_dtype)
            dvnew_l0 = T.alloc_L0C([C, BV], accum_dtype)
            dp_l0 = T.alloc_L0C([C, C], accum_dtype)
            dq_l0 = T.alloc_L0C([C, DK], accum_dtype)
            dk_l0 = T.alloc_L0C([C, DK], accum_dtype)
            dw_l0 = T.alloc_L0C([C, DK], accum_dtype)
            dkt_l0 = T.alloc_L0C([C, DK], accum_dtype)
            t1_l0 = T.alloc_L0C([C, BV], accum_dtype)
            da_l0 = T.alloc_L0C([C, C], accum_dtype)
            t2_l0 = T.alloc_L0C([C, DK], accum_dtype)
            dsum_l0 = T.alloc_L0C([C, C], accum_dtype)
            dh_l0 = T.alloc_L0C([BV, DK], accum_dtype)
            dx_l0 = T.alloc_L0C([C, C], accum_dtype)
            rs_l0 = T.alloc_L0C([C, 16], accum_dtype)
            # S7c liveness split: per-segment L0C copies so each buffer's
            # first-gen/last-kill interval stays segment-local (the memory
            # planner models one interval per buffer across the whole func).
            dp1_l0 = T.alloc_L0C([C, C], accum_dtype)  # pass-2 QK^T (pre-dP)
            rs2_l0 = T.alloc_L0C([C, 16], accum_dtype)  # c-level ones-GEMM sink
            dw_w_l0 = T.alloc_L0C([C, DK], accum_dtype)  # pass-2 dW replay staging
            rs3_l0 = T.alloc_L0C([C, 16], accum_dtype)  # B7 dbeta ones-GEMM sink
            rs4_l0 = T.alloc_L0C([C, 16], accum_dtype)  # B8 d(betaE) ones-GEMM sink

            # ---- UB (Vector, fp32 accumulation) ----
            g_ub = T.alloc_ub([C], accum_dtype)  # gam
            e_ub = T.alloc_ub([C], accum_dtype)  # E = exp(gam)
            th_ub = T.alloc_ub([C], accum_dtype)  # theta = exp(gam_last - gam)
            zero_ub = T.alloc_ub([C], accum_dtype)  # negation source (fwd chunk_h :80/:92)
            be_ub = T.alloc_ub([C], accum_dtype)  # beta
            bee_ub = T.alloc_ub([C], accum_dtype)  # beta .* E
            dE_ub = T.alloc_ub([C], accum_dtype)
            dgam_ub = T.alloc_ub([C], accum_dtype)
            dbe_ub = T.alloc_ub([C], accum_dtype)
            tmp1_ub = T.alloc_ub([C], accum_dtype)
            tmp2_ub = T.alloc_ub([C], accum_dtype)
            scal_ub = T.alloc_ub([8], accum_dtype)  # 32B cell (#1307)
            glast_ub = T.alloc_ub([8], accum_dtype)  # exp(gam_{n-1})
            gm_ub = T.alloc_ub([C, C], accum_dtype)  # Gamma
            msk_ub = T.alloc_ub([C, C], accum_dtype)  # M or J (reloaded per use)
            kk_ub = T.alloc_ub([C, C], accum_dtype)  # KK^T
            pk_ub = T.alloc_ub([C, C], accum_dtype)  # QK^T -> P -> dP -> dQKt
            dgam2_ub = T.alloc_ub([C, C], accum_dtype)  # dGam / dGam.*Gam
            da_ub = T.alloc_ub([C, C], accum_dtype)  # dA -> dX -> dL -> dXkk
            dx_ub = T.alloc_ub([C, C], accum_dtype)  # dX^T scratch / dsum lane
            mul_ub = T.alloc_ub([C, C], accum_dtype)  # subst scratch / general
            bq_ub = T.alloc_ub([C, BV], accum_dtype)  # u0 -> do -> e_do -> t1 lane
            xk_ub = T.alloc_ub([C, DK], accum_dtype)  # bek / kt / dw / products lane
            k2_ub = T.alloc_ub([C, DK], accum_dtype)  # K fp32 resident (K .* X products)
            wh_ub = T.alloc_ub([C, BV], accum_dtype)  # wh -> vnew lane
            t1_ub = T.alloc_ub([C, BV], accum_dtype)  # QH / V .* t1 product lane
            dvnew_ub = T.alloc_ub([C, BV], accum_dtype)
            dq_ub = T.alloc_ub([C, DK], accum_dtype)
            dk_ub = T.alloc_ub([C, DK], accum_dtype)
            hin_ub = T.alloc_ub([BV, DK], accum_dtype)  # H_in block
            dhp_ub = T.alloc_ub([BV, DK], accum_dtype)  # Delta' block -> dH block
            dh_ub = T.alloc_ub([BV, DK], accum_dtype)  # dH block accumulator (B5/6/9)
            rs_ub = T.alloc_ub([C, 16], accum_dtype)  # ones-GEMM row/col-sum sink
            red1_ub = T.alloc_ub([BV], accum_dtype)  # 1D reduce sink (B9 inner product)
            rowb_ub = T.alloc_ub([1, BV], accum_dtype)  # B9 second reduce stage
            rowc_ub = T.alloc_ub([1, C], accum_dtype)  # B10 second reduce stage
            # main-dtype staging for GM<->UB handoffs (dtype-matched copies)
            io_bq = T.alloc_ub([C, BV], dtype)
            io_xk = T.alloc_ub([C, DK], dtype)
            io_cc = T.alloc_ub([C, C], dtype)
            io_hv = T.alloc_ub([BV, DK], dtype)
            # S7c liveness split: segment-local copies so each buffer's
            # first-gen/last-kill interval stays segment-local.
            xk2a_ub = T.alloc_ub([C, DK], accum_dtype)  # dW sign fold lane
            xk2b_ub = T.alloc_ub([C, DK], accum_dtype)  # bee .* t2 lane
            xk2c_ub = T.alloc_ub([C, DK], accum_dtype)  # K .* t2 lane
            xk3a_ub = T.alloc_ub([C, DK], accum_dtype)  # K .* dKt lane
            xk3b_ub = T.alloc_ub([C, DK], accum_dtype)  # dKt .* th lane
            xk4_ub = T.alloc_ub([C, DK], accum_dtype)  # dKt replay lane
            k2b_ub = T.alloc_ub([C, DK], accum_dtype)  # K reload (B8 d(betaE))
            k2c_ub = T.alloc_ub([C, DK], accum_dtype)  # K reload (B10 dth)
            io_xk2a = T.alloc_ub([C, DK], dtype)
            io_xk2b = T.alloc_ub([C, DK], dtype)
            io_xk3 = T.alloc_ub([C, DK], dtype)
            io_xk4 = T.alloc_ub([C, DK], dtype)
            io_xk5 = T.alloc_ub([C, DK], dtype)
            t1b_ub = T.alloc_ub([C, BV], accum_dtype)  # B7 t1 replay lane
            wh2_ub = T.alloc_ub([C, BV], accum_dtype)  # B7 V .* t1 lane
            io_bq2 = T.alloc_ub([C, BV], dtype)
            io_hv2 = T.alloc_ub([BV, DK], dtype)
            pk2_ub = T.alloc_ub([C, C], accum_dtype)  # B3 dP lane
            msk2_ub = T.alloc_ub([C, C], accum_dtype)  # B3 mask lane
            msk3_ub = T.alloc_ub([C, C], accum_dtype)  # B11 J mask lane
            da2_ub = T.alloc_ub([C, C], accum_dtype)  # B11 dA replay lane
            da3_ub = T.alloc_ub([C, C], accum_dtype)  # B11 dL.*Gam.*KK lane
            mul2_ub = T.alloc_ub([C, C], accum_dtype)  # B12 dX lane
            mul3_ub = T.alloc_ub([C, C], accum_dtype)  # B12 transpose lane
            io_cc1 = T.alloc_ub([C, C], dtype)

            # ones-GEMM rowsum sources: GM ones -> L1 (once per block)
            T.copy(OnesDV[0, 0], onesbv_l1)
            T.copy(OnesDK[0, 0], onesdk_l1)
            T.copy(OnesC[0, 0], onesc_l1)

            # ================= pass 2: reverse adjoint (B1-B15) =================
            for c in T.serial(chunk_num):
                cc = chunk_num - 1 - c  # reversed chunk index
                # ---- replay state (B5/B8/B9/B10/B13 sources) ----
                T.tile.fill(g_ub, 0.0)
                T.tile.fill(be_ub, 0.0)
                T.tile.fill(zero_ub, 0.0)
                T.copy(S[b, h, cc * C], g_ub)
                T.copy(Beta[b, h, cc * C], be_ub)
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                if cc == chunk_num - 1:
                    T.copy(g_ub[n_tail - 1:n_tail], glast_ub[0:1])
                else:
                    T.copy(g_ub[C - 1:C], glast_ub[0:1])
                T.tile.exp(glast_ub, glast_ub)
                for i in T.Parallel(C):
                    th_ub[i] = g_ub[i] - g_ub[C - 1]
                if cc == chunk_num - 1:
                    for i in T.Parallel(C):
                        th_ub[i] = g_ub[i] - g_ub[n_tail - 1]
                for i in T.Parallel(C):
                    th_ub[i] = zero_ub[i] - th_ub[i]
                for i in T.Parallel(C):
                    th_ub[i] = T.exp(th_ub[i])
                for i in T.Parallel(C):
                    bee_ub[i] = be_ub[i] * e_ub[i]
                # Gamma_rc = exp(gam_r - gam_c)
                for (i, j) in T.Parallel(C, C):
                    gm_ub[i, j] = g_ub[i] - g_ub[j]
                for (i, j) in T.Parallel(C, C):
                    gm_ub[i, j] = T.exp(gm_ub[i, j])
                # ---- K / Kt / beK / W / Q / KK^T / QK^T / P (recompute) ----
                T.copy(K[b, h, cc * C, 0], io_xk)
                T.copy(io_xk, k2_ub)  # K fp32 resident (K .* X products)
                T.copy(k2_ub, io_xk)
                T.copy(io_xk, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], k_l1)
                for (i, j) in T.Parallel(C, DK):
                    xk_ub[i, j] = k2_ub[i, j] * th_ub[i]
                T.copy(xk_ub, io_xk)
                T.copy(io_xk, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], kt_l1)
                for (i, j) in T.Parallel(C, DK):
                    xk_ub[i, j] = k2_ub[i, j] * bee_ub[i]
                T.copy(xk_ub, io_xk)
                T.copy(io_xk, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], bek_l1)
                T.copy(A[b, h, cc * C, 0], a_l1)
                T.gemm_v0(a_l1, bek_l1, dw_w_l0, init=True)
                T.copy(dw_w_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk_ub)
                for (i, j) in T.Parallel(C, DK):  # pass 2 keeps -W (B6 sign fold)
                    xk_ub[i, j] = 0.0 - xk_ub[i, j]
                T.copy(xk_ub, io_xk)
                T.copy(io_xk, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], w_l1)
                T.copy(Q[b, h, cc * C, 0], q_l1)
                T.gemm_v0(k_l1, k_l1, kk_l0, transpose_B=True, init=True)
                T.copy(kk_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], kk_ub)
                T.gemm_v0(q_l1, k_l1, dp1_l0, transpose_B=True, init=True)
                T.copy(dp1_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], pk_ub)  # QK^T
                T.copy(dp1_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], mul_ub)  # QK^T resident copy (B3 needs it)
                T.copy(MskM[0, 0], msk_ub)
                for (i, j) in T.Parallel(C, C):
                    pk_ub[i, j] = pk_ub[i, j] * gm_ub[i, j]
                for (i, j) in T.Parallel(C, C):
                    pk_ub[i, j] = pk_ub[i, j] * msk_ub[i, j]
                T.copy(pk_ub, io_cc1)  # pk_ub = P
                T.copy(io_cc1, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                # ---- 1D accumulators ----
                T.tile.fill(dE_ub, 0.0)
                T.tile.fill(dgam_ub, 0.0)
                T.tile.fill(tmp1_ub, 0.0)  # dbeta accumulation
                T.tile.fill(dbe_ub, 0.0)
                # S7c: dead fills dropped -- dq_ub reloaded at write-back,
                # dk_ub first-assigned (=) at its first accumulation.
                # ---- BV blocks: B1/B2/B5/B6/B7/B9 (+ dA/B7 accum) ----
                for bv in T.serial(bv_num):
                    T.tile.fill(bq_ub, 0.0)
                    T.copy(DO[b, h, cc * C, bv * BV], io_bq)
                    T.copy(io_bq, bq_ub)  # dO block
                    T.tile.fill(hin_ub, 0.0)
                    T.copy(WS_H[b, h, cc, bv * BV, 0], io_hv)
                    T.copy(io_hv, hin_ub)  # H_in block
                    T.copy(hin_ub, io_hv)
                    T.copy(io_hv, workspace_vbd[cid, :, :])
                    T.copy(workspace_vbd[cid, :, :], hin_l1)
                    T.tile.fill(dhp_ub, 0.0)
                    # S7e: WS_D is fp32 GM -- copy straight into fp32 dhp_ub
                    # (gm->ub copies must be dtype-matched; io_hv is main dtype)
                    T.copy(WS_D[b, h, bv * BV, 0], dhp_ub)  # Delta' block
                    T.copy(dhp_ub, io_hv)
                    T.copy(io_hv, workspace_vbd[cid, :, :])
                    T.copy(workspace_vbd[cid, :, :], dhp_l1)
                    # Vnew_bv recomputed = U0 - W H_in^T
                    T.tile.fill(wh_ub, 0.0)
                    T.copy(V[b, h, cc * C, bv * BV], io_bq)
                    T.copy(io_bq, wh_ub)
                    for (i, j) in T.Parallel(C, BV):
                        wh_ub[i, j] = wh_ub[i, j] * be_ub[i]
                    T.copy(wh_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bv_l1)  # beta .* V (B7 dA input)
                    T.gemm_v0(a_l1, bv_l1, u0_l0, init=True)
                    T.copy(u0_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], wh_ub)
                    T.tile.fill(t1_ub, 0.0)
                    T.gemm_v0(w_l1, hin_l1, wh_l0, transpose_B=True, init=True)
                    T.copy(wh_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1_ub)  # W H_in^T
                    for (i, j) in T.Parallel(C, BV):
                        wh_ub[i, j] = wh_ub[i, j] + t1_ub[i, j]  # Vnew (w_l1=-W: U0 + (-W)H^T)
                    T.copy(wh_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], vnew_l1)
                    # QH = Q H_in^T -> B5a dE rowsum
                    T.gemm_v0(q_l1, hin_l1, qh_l0, transpose_B=True, init=True)
                    T.copy(qh_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1_ub)
                    for (i, j) in T.Parallel(C, BV):
                        t1_ub[i, j] = bq_ub[i, j] * t1_ub[i, j]  # dO .* QH
                    T.copy(t1_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], t1_l1)
                    T.gemm_v0(t1_l1, onesbv_l1, rs_l0, init=True)
                    T.copy(rs_l0, workspace_lc16[cid, :, :])
                    T.copy(workspace_lc16[cid, :, :], rs_ub)
                    for i in T.serial(C):
                        dE_ub[i] = dE_ub[i] + rs_ub[i, 0]
                    # B1: dVnew = P^T dO + Kt Delta'^T
                    T.copy(bq_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bq_l1)  # dO L1
                    T.gemm_v0(p_l1, bq_l1, dvnew_l0, transpose_A=True, init=True)
                    T.gemm_v0(kt_l1, dhp_l1, dvnew_l0, transpose_B=True, init=False)
                    T.copy(dvnew_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], dvnew_ub)
                    # B2: dP = dO Vnew^T (accumulate across BV blocks)
                    T.gemm_v0(bq_l1, vnew_l1, dp_l0, transpose_B=True, init=(bv == 0))
                    # B5b/c: dQ += (E .* dO) H_in^T ; dH += (E .* dO)^T Q
                    for (i, j) in T.Parallel(C, BV):
                        bq_ub[i, j] = bq_ub[i, j] * e_ub[i]  # E .* dO
                    T.copy(bq_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], t1_l1)
                    T.gemm_v0(t1_l1, hin_l1, dq_l0, init=(bv == 0))  # (E.*dO) H_in (over DV)
                    T.gemm_v0(t1_l1, q_l1, dh_l0, transpose_A=True, init=True)
                    # B6: dW += dVnew H_in (sign folded once at full width);
                    #     dH += -dVnew^T W  (w_l1 holds -W -> +dVnew^T(-W))
                    T.copy(dvnew_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bq_l1)  # dVnew L1 (B6/B7)
                    T.gemm_v0(bq_l1, hin_l1, dw_l0, init=(bv == 0))  # dVnew H_in (over DV)
                    T.gemm_v0(vnew_l1, w_l1, dh_l0, transpose_A=True, init=False)
                    # B9: dH += exp(gam_{n-1}) Delta'
                    T.copy(dh_l0, workspace_lbd[cid, :, :])
                    T.copy(workspace_lbd[cid, :, :], dh_ub)
                    for (i, j) in T.Parallel(BV, DK):
                        dh_ub[i, j] = dh_ub[i, j] + dhp_ub[i, j] * glast_ub[0]
                    T.gemm_v0(vnew_l1, dhp_l1, dkt_l0, init=(bv == 0))  # Vnew Delta' (over DV)
                    # B9 scalar: dgam_{n-1} += exp(gam_{n-1}) <H_in, Delta'>
                    for (i, j) in T.Parallel(BV, DK):
                        hin_ub[i, j] = hin_ub[i, j] * dhp_ub[i, j]
                    T.reduce_sum(hin_ub, red1_ub, dim=1)
                    for j in T.Parallel(BV):
                        rowb_ub[0, j] = red1_ub[j]
                    T.reduce_sum(rowb_ub, scal_ub[0:1], dim=1)
                    if cc == chunk_num - 1:
                        dgam_ub[n_tail - 1] = dgam_ub[n_tail - 1] + scal_ub[0] * glast_ub[0]
                    else:
                        dgam_ub[C - 1] = dgam_ub[C - 1] + scal_ub[0] * glast_ub[0]
                    # B15: dH block -> WS_D relay (dh0 for the first chunk)
                    # S7e: WS_D is float GM -- fp32 ub direct copy (ub->gm must
                    # be dtype-matched); DH0 keeps the main-dtype io_hv2 bridge
                    T.copy(dh_ub, WS_D[b, h, bv * BV, 0])
                    if cc == 0:
                        T.copy(dh_ub, io_hv2)
                        T.copy(io_hv2, DH0[b, h, bv * BV, 0])
                    # B7: t1 = A^T dU0(=dVnew) ; dV = beta .* t1 ;
                    #     dbeta += rowsum(V .* t1) ; dA += dU0 (beta .* V)^T
                    T.gemm_v0(a_l1, bq_l1, t1_l0, transpose_A=True, init=True)
                    T.copy(t1_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1b_ub)  # t1 block
                    T.copy(V[b, h, cc * C, bv * BV], io_bq2)
                    T.copy(io_bq2, wh2_ub)  # V block (Vnew dead after B9)
                    for (i, j) in T.Parallel(C, BV):
                        wh2_ub[i, j] = wh2_ub[i, j] * t1b_ub[i, j]  # V .* t1
                    T.copy(wh2_ub, io_bq2)
                    T.copy(io_bq2, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], t1_l1)
                    T.gemm_v0(t1_l1, onesbv_l1, rs3_l0, init=True)
                    T.copy(rs3_l0, workspace_lc16[cid, :, :])
                    T.copy(workspace_lc16[cid, :, :], rs_ub)
                    for i in T.serial(C):
                        tmp1_ub[i] = tmp1_ub[i] + rs_ub[i, 0]  # dbeta7 accumulate
                    for (i, j) in T.Parallel(C, BV):
                        t1b_ub[i, j] = t1b_ub[i, j] * be_ub[i]  # dV = beta .* t1
                    T.copy(t1b_ub, io_bq2)
                    T.copy(io_bq2, DVG[b, h, cc * C, bv * BV])
                    T.gemm_v0(bq_l1, bv_l1, da_l0, transpose_B=True, init=(bv == 0))

                # S7f-D: dA accumulation record -> GM workspace (H3; the
                # B7-over-BV sum is the Cube/L0C intrinsic GEMM accumulation,
                # the accumulated box's durable copy lives in GM)
                T.copy(da_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], da2_ub)
                T.copy(da2_ub, WS_DA[b, h, 0, 0])

                # ---- B8 (full width): dA += dW (beK)^T ; t2 = A^T dW ----
                T.copy(dw_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk2a_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk2a_ub[i, j] = 0.0 - xk2a_ub[i, j]  # dW = -sum dVnew H_in
                T.copy(xk2a_ub, io_xk2a)
                T.copy(io_xk2a, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], xk_l1)
                T.gemm_v0(xk_l1, bek_l1, da_l0, transpose_B=True, init=False)
                T.copy(da_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], da2_ub)
                T.copy(da2_ub, WS_DA[b, h, 0, 0])  # S7f-D: dA record update (B7+B8)
                T.gemm_v0(a_l1, xk_l1, t2_l0, transpose_A=True, init=True)  # t2
                T.copy(t2_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk2b_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk2b_ub[i, j] = xk2b_ub[i, j] * bee_ub[i]  # (beta .* E) .* t2
                for (i, j) in T.Parallel(C, DK):
                    dk_ub[i, j] = xk2b_ub[i, j]  # dk first-assigned (S7c)
                # d(betaE) = rowsum(K .* t2)
                # S7e: K (main dtype GM) reloads through the dtype bridge
                # (gm->ub copies must be dtype-matched; k2b_ub is fp32)
                T.copy(K[b, h, cc * C, 0], io_xk2b)  # K reload (S7c liveness split)
                T.copy(io_xk2b, k2b_ub)
                T.copy(t2_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk2c_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk2c_ub[i, j] = k2b_ub[i, j] * xk2c_ub[i, j]
                T.copy(xk2c_ub, io_xk2b)
                T.copy(io_xk2b, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], xk_l1)
                T.gemm_v0(xk_l1, onesdk_l1, rs4_l0, init=True)
                T.copy(rs4_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    dbe_ub[i] = rs_ub[i, 0]
                # dE += beta .* d(betaE) ; dbeta += E .* d(betaE)
                for i in T.Parallel(C):
                    dE_ub[i] = dE_ub[i] + be_ub[i] * dbe_ub[i]
                for i in T.Parallel(C):
                    tmp1_ub[i] = tmp1_ub[i] + e_ub[i] * dbe_ub[i]
                # ---- B3: d(QKt) = dP .* Gam .* M ; dGam += dP .* M .* QK^T ----
                T.copy(dp_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], pk2_ub)  # dP from B2 accum (P itself lives in p_l1)
                T.copy(MskM[0, 0], msk2_ub)
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = pk2_ub[i, j] * gm_ub[i, j]
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = da_ub[i, j] * msk2_ub[i, j]  # d(QKt)
                for (i, j) in T.Parallel(C, C):
                    dgam2_ub[i, j] = pk2_ub[i, j] * msk2_ub[i, j]
                for (i, j) in T.Parallel(C, C):
                    dgam2_ub[i, j] = dgam2_ub[i, j] * mul_ub[i, j]  # dGam (init)
                # S7f-D: dGam accumulation record -> GM workspace (H3
                # "cross-chunk accumulation state only via GM workspace";
                # WS_D :665/:750 fp32 direct-copy discipline)
                T.copy(dgam2_ub, WS_DG2[b, h, 0, 0])
                # B4: dQ += d(QKt) K ; dK += d(QKt)^T Q
                T.copy(da_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                T.gemm_v0(da_l1, k_l1, dq_l0, init=False)
                T.gemm_v0(da_l1, q_l1, dk_l0, transpose_A=True, init=True)
                # ---- B11: dL = J .* (-(A^T dA) A^T) ----
                # S7f-D: dA comes back from the GM accumulation record
                T.copy(WS_DA[b, h, 0, 0], da2_ub)
                T.copy(da2_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                T.gemm_v0(a_l1, da_l1, dsum_l0, transpose_A=True, init=True)
                # S7e erratum (错C second site, handoff Sec 3 "c"): gemm_v0 takes
                # A/B from L1 via copy_l1_to_l0a/l0b and both must share T1, so
                # the L0C accumulator dsum_l0 cannot be the A source (fp16 T1
                # mismatch; fp32 was an illegal L0C->L0A move that only looked
                # green because the template does not encode the scope). Stage
                # dsum into the main-dtype L1 lane instead (da_l1 dies at the
                # line above; its next use restages it first) -- keeps the
                # kernel-wide A/B=main-dtype-L1, C=fp32-L0C discipline.
                T.copy(dsum_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], da2_ub)
                T.copy(da2_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                T.gemm_v0(da_l1, a_l1, dx_l0, transpose_B=True, init=True)  # (A^T dA) A^T
                T.copy(dx_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], dx_ub)
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = 0.0 - dx_ub[i, j]
                T.copy(MskJ[0, 0], msk3_ub)
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = dx_ub[i, j] * msk3_ub[i, j]  # dL
                # dbeta += rowsum(dL .* Gam .* KK^T)
                for (i, j) in T.Parallel(C, C):
                    da3_ub[i, j] = dx_ub[i, j] * gm_ub[i, j]
                for (i, j) in T.Parallel(C, C):
                    da3_ub[i, j] = da3_ub[i, j] * kk_ub[i, j]
                T.copy(da3_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                T.gemm_v0(da_l1, onesc_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    tmp1_ub[i] = tmp1_ub[i] + rs_ub[i, 0]
                # dGam += dL .* beta .* KK^T ; dX = dL .* beta .* Gam
                for (i, j) in T.Parallel(C, C):
                    mul2_ub[i, j] = dx_ub[i, j] * be_ub[i]
                for (i, j) in T.Parallel(C, C):
                    dgam2_ub[i, j] = dgam2_ub[i, j] + mul2_ub[i, j] * kk_ub[i, j]
                T.copy(dgam2_ub, WS_DG2[b, h, 0, 0])  # S7f-D: dGam record update (B3+B11)
                for (i, j) in T.Parallel(C, C):
                    mul2_ub[i, j] = mul2_ub[i, j] * gm_ub[i, j]  # dX
                # dK += (dX + dX^T) K
                T.tile.transpose(dx_ub, mul2_ub)
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = dx_ub[i, j] + mul2_ub[i, j]
                T.copy(dx_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                T.gemm_v0(p_l1, k_l1, dk_l0, init=False)
                # ---- B12: dgam += rowsum(dGam.*Gam) - colsum(dGam.*Gam) ----
                for (i, j) in T.Parallel(C, C):
                    dgam2_ub[i, j] = dgam2_ub[i, j] * gm_ub[i, j]
                T.copy(dgam2_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                T.gemm_v0(p_l1, onesc_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    dgam_ub[i] = dgam_ub[i] + rs_ub[i, 0]
                T.tile.transpose(mul3_ub, dgam2_ub)
                T.copy(mul3_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                T.gemm_v0(p_l1, onesc_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    dgam_ub[i] = dgam_ub[i] - rs_ub[i, 0]
                # ---- B10: dK += theta .* dKt ; dth = rowsum(K .* dKt) ----
                # S7e: K reload through the dtype bridge (gm->ub dtype-matched)
                T.copy(K[b, h, cc * C, 0], io_xk3)  # K reload (S7c liveness split)
                T.copy(io_xk3, k2c_ub)
                T.copy(dkt_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk3a_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk3a_ub[i, j] = k2c_ub[i, j] * xk3a_ub[i, j]  # K .* dKt
                T.copy(xk3a_ub, io_xk3)
                T.copy(io_xk3, workspace_vdk[cid, :, :])
                T.copy(workspace_vdk[cid, :, :], xk_l1)
                T.gemm_v0(xk_l1, onesdk_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    tmp2_ub[i] = rs_ub[i, 0]  # dth
                T.copy(dkt_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk3b_ub)
                for (i, j) in T.Parallel(C, DK):
                    xk3b_ub[i, j] = xk3b_ub[i, j] * th_ub[i]
                for (i, j) in T.Parallel(C, DK):
                    dk_ub[i, j] = dk_ub[i, j] + xk3b_ub[i, j]
                # dgam += -dth .* theta ; dgam_{n-1} += sum(dth .* theta)
                for i in T.Parallel(C):
                    tmp2_ub[i] = tmp2_ub[i] * th_ub[i]
                for i in T.Parallel(C):
                    dgam_ub[i] = dgam_ub[i] - tmp2_ub[i]
                for j in T.Parallel(C):
                    rowc_ub[0, j] = tmp2_ub[j]
                T.reduce_sum(rowc_ub, scal_ub[0:1], dim=1)
                if cc == chunk_num - 1:
                    dgam_ub[n_tail - 1] = dgam_ub[n_tail - 1] + scal_ub[0]
                else:
                    dgam_ub[C - 1] = dgam_ub[C - 1] + scal_ub[0]
                # ---- B13: dgam += E .* dE ----
                for i in T.Parallel(C):
                    dgam_ub[i] = dgam_ub[i] + e_ub[i] * dE_ub[i]
                # ---- B14: dg = suffix-sum(dgam) (cumsum-mirror serial chain) ----
                T.tile.fill(tmp2_ub, 0.0)
                for i in T.serial(C):
                    if i > 0:
                        tmp2_ub[C - 1 - i] = tmp2_ub[C - i]
                    tmp2_ub[C - 1 - i] = tmp2_ub[C - 1 - i] + dgam_ub[C - 1 - i]
                # ---- gradient write-back ----
                T.copy(dq_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], dq_ub)
                T.copy(dq_ub, io_xk4)
                T.copy(io_xk4, DQ[b, h, cc * C, 0])
                T.copy(dk_l0, workspace_ldk[cid, :, :])
                T.copy(workspace_ldk[cid, :, :], xk4_ub)
                for (i, j) in T.Parallel(C, DK):
                    dk_ub[i, j] = dk_ub[i, j] + xk4_ub[i, j]
                T.copy(dk_ub, io_xk5)
                T.copy(io_xk5, DKG[b, h, cc * C, 0])
                # S7e: DBeta/DG are float GM and tmp1/tmp2 are fp32 ub --
                # same-dtype direct copy (ub->gm must be dtype-matched)
                T.copy(tmp1_ub, DBeta[b, h, cc * C])
                T.copy(tmp2_ub, DG[b, h, cc * C])

    return main


def gdn_bwd_v2(q, k, v, g, beta, h0, do, chunk_size=64, NT_MAX=None):
    """Host wrapper (proto.yaml schema). Device follows `q`.

    Four-kernel pipeline: cumsum (g -> WS_G) -> solve_a (A = (I+L)^-1 ->
    WS_A) -> scan1 (replay -> WS_H) -> scan2 (reverse B1-B15 -> six
    gradients). Cross-kernel state (WS_G / WS_A / WS_H / WS_D) travels in GM;
    the workspace_* CV-handoff buffers are per-kernel workspaces
    (workspace_idx, auto-allocated by the runtime).
    """
    import torch

    B, H, T_len, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    nt = (T_len + C - 1) // C
    if NT_MAX is None:
        NT_MAX = nt
    if nt > NT_MAX:
        raise ValueError(f"NT={nt} exceeds NT_MAX={NT_MAX}")
    BV, BK = 64, 64
    dev, dt = q.device, q.dtype
    dtype = "float16" if dt == torch.float16 else "float32"
    msk_m = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
    idt = torch.eye(C, dtype=torch.float32, device=dev)
    # ones-GEMM constants follow the main dtype (S7e: gemm_v0 A/B share T1)
    ones_dv = torch.ones(DV, 16, dtype=dt, device=dev)
    ones_dk = torch.ones(DK, 16, dtype=dt, device=dev)
    ones_c = torch.ones(C, 16, dtype=dt, device=dev)

    ws_g = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, BK, dtype=dtype)(
        k, beta, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        k, v, beta, ws_g, ws_a, h0)
    outs = gdn_bwd_v2_scan2(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        q, k, v, do, beta, ws_g, ws_a, msk_m, msk_j,
        ones_dv, ones_dk, ones_c, ws_h)
    return outs


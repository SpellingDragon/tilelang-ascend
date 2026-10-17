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
  H3  pass_configs (auto CV/sync/memory-planning/tail-mask all True; vector
      mask reuse forced OFF -- see the pass_configs block for the 507015
      mechanism), cross-chunk recurrence state ONLY through GM workspace
      (WS_H / WS_D), zero hand-written set_cross_flag / wait_cross_flag.

Kernel split (S7 implementation ruling, Q6 handoff 3-2; S7e splits the scan
into two kernels so each pass keeps a pass-local UB peak; S7f-E splits the
reverse adjoint again (B1-B6 window / B7-B15 window, _s7f_ub_budget.py E-ruling)
so every kernel's UB static sum stays under the 192 KiB UB budget -- which also
keeps the memory-planning KILL-DISPLACED failure dissolved, see
_s7d_handoff.md Sec 2/Sec 5):
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
  4. gdn_bwd_v2_scan2a : pass 2 front window (reverse adjoint B1-B6 + B3/B4 +
     B9 + B15 relay; design Sec 3) -> DQ, DH0 and the seven cross-split
     gradient boundary boxes WS_DV / WS_DW / WS_DKT / WS_DE / WS_DGAM2 /
     WS_DGAM / WS_DK ([B,H,L,*] region-copy form, same idiom as DQ). The
     DK-wide UB/L1 tiles are processed in BK=64 column blocks (DK-half
     blocking: the DK-contracting GEMMs become two init=(hh==0) partial
     products).
  5. gdn_bwd_v2_scan2b : pass 2 back window (B7/B8 + B10-B15) -> the remaining
     four gradients DKG / DVG / DBeta / DG, consuming the boundary boxes.
     Cross-kernel state (WS_G / WS_A / WS_H / WS_D + the S7f boundary boxes)
     travels in GM only (H3); the same workspace_* CV-handoff idiom as scan1.
     Vnew/P/QH/A are recomputed inside the passes (design Sec 8.1: "not
     stored, recomputed with 2-3 GEMMs").

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

import os

import tilelang
from tilelang import language as T

# Debug switch (compile-time constant; default off => production bytecode
# unchanged). With GDN_S2A_DBG=1, scan2a emits dqkt (da_ub) through the
# WS_DKT box instead of dKt, so one on-card run tells "dqkt construction
# wrong" (dqkt != golden) from "GEMM/L1 path wrong" (dqkt == golden but
# DQ != golden dqkt@K). Only meaningful for single-chunk runs, where the
# DKT box is zero anyway; never set for acceptance runs.
_S2A_DBG = os.environ.get("GDN_S2A_DBG", "0") == "1"

# Developer + combineCV: design.md Sec 7 -- all five keys True (H3).
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
    # Vector-mask reuse OFF (was default-on): the reuse state machine declares
    # cross-dtype copy_ub_to_ub a NormalFull post-state, but its runtime body
    # is AscendC::Cast(dst, src, mode, calCount) -- a count-form op that leaves
    # the counter mask live. A later norm-form Sub (C*BV, repeat=64) then
    # inherits counter/count=4096 and dies 507015 'illegal DFE configuration'
    # (on-card scan1 verdict, kernel.cu L268-286; solve_a has the same cast
    # pair but a mask-resetting op in between, which is why it passes). With
    # reuse off the legalizer clears mask facts per terminal and re-emits
    # every required SetMask* -- conservative, correctness-first.
    tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE: False,
}

# [REORDER-COST-AUDIT] (design.md Sec 8/Sec 11 cost ledger, landed as-is)
#   GM pass      : inputs read once per chunk trip (q/k/v/do C*DK*B bytes each;
#                  g/beta C*4B; h0 once), WS_H written NT times (replay) + read
#                  NT times (reverse), WS_A written once (solve_a) + read NT
#                  times, WS_G written once + read NT times, WS_D NT writes + NT
#                  reads (Delta relay), six gradients written once per chunk.
#                  S7f-E: the seven cross-split boundary boxes add one write
#                  (scan2a) + one read (scan2b) per chunk each; the DK-half
#                  blocking adds extra half-block GM trips for K/Q/dO staging.
#   DMA bytes    : per chunk at DK=DV=128 -- q/k/do 16KiB, v 16KiB, WS_H 32KiB,
#                  WS_A 8KiB, WS_D 64KiB; all region copies are chunk-aligned
#                  contiguous blocks (no per-element strided GM access).
#   GM scalar    : 0 -- no per-token scalar GM loads/stores (H2; contrast v1's
#                  2 loads + 2 stores per token trip).
#   div/mod      : 0 runtime div/mod on data paths (block decode is compile-time
#                  integer arithmetic on the task axis).
#   active cores : NB = B*H blocks; per-core serial work = 3*NT chunk trips
#                  (replay NT + reverse 2NT across scan2a/scan2b).
#   workspace    : WS_H B*H*(NT+1)*DV*DK + WS_A B*H*NT*C*C + WS_G B*H*L +
#                  WS_D B*H*DV*DK -- 82.00 MiB at the true-shape profile
#                  (B1 H16 DK128 DV128 L8192), vs 16.0 GiB for v1 (design 8.2).
#                  S7f-D: + WS_DA/WS_DG2 B*H*C*C*4B (0.50 MiB) dA/dGam GM
#                  accumulation records. S7f-E boundary boxes (design 5.4):
#                  + WS_DV/WS_DW 32+32 MiB (main dtype), WS_DKT/WS_DK 64+64
#                  MiB (fp32), WS_DE/WS_DGAM 0.5+0.5 MiB, WS_DGAM2 32 MiB --
#                  +225.0 MiB, total workspace 307.5 MiB at the true-shape
#                  profile (still GM-only state, H3 intact).


# ---------------------------------------------------------------------------
# 1) gdn_bwd_v2_cumsum: G -> WS_G, chunk-local cumsum (design Sec 2: gam_r =
#    sum_{m<=r, same chunk} g_m; fwd gdn_chunk_cumsum.py:32-40 idiom verbatim,
#    grid made H-safe: B*H*chunk_num, no VEC_NUM split).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)  # explicit index, see solve_a note
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
# out_idx MUST be the declared index of A (=5), not -1: the CythonAdapter
# legalizes -1 to len(params)-1 AFTER the CV pass has appended the auto_gm
# buffer params, so -1 lands on the trailing agm slot and A stays in the
# consumed-input range (on-card diag: nparams=8, result=[7], agm=[6,7] ->
# forward() consumes inputs[5] -> IndexError). Explicit index is immune;
# scan1/scan2a/scan2b already use explicit indices for the same reason.
@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
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
                # glast must be a scalar read, NOT `T.copy(..., glast_ub[0:1])`:
                # the 1-element UB->UB DataCopy raises 507015 "VEC supports
                # illegal configurations in commands" on card (w11 micro-probes:
                # P3 FAIL vs P4/P6 PASS; fixp0 0x60000df; granularity hypothesis
                # -- 4B vs 32B copy unit -- not directly proven). Same form in
                # scan2a/scan2b below.
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                if c == chunk_num - 1:
                    glast_ub[0] = g_ub[n_tail - 1]
                else:
                    glast_ub[0] = g_ub[C - 1]
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
# 4) gdn_bwd_v2_scan2a: pass 2 front window (reverse adjoint B1-B6 + B3/B4 +
#    B9 scalar + B15 relay; design Sec 3) -> DQ, DH0 + seven cross-split
#    gradient boundary boxes. S7f-E ruling (after S7f-D): scan2's UB static
#    sum stayed at 983.6 KiB over the 192 KiB budget (_s7f_ub_budget.py), so
#    the reverse pass splits into two kernels -- same GM-boundary idiom as the
#    S7e scan1/scan2 split (handoff _s7d_handoff.md Sec 5). DK-wide tiles run
#    in BK=64 column halves: the DK-contracting GEMMs (B5a QH, B1 Kt Dhp^T,
#    B6b Vnew^T W) become init=(hh==0) half products, and the DK-wide outputs
#    (dq/dh/dw/dkt) accumulate into per-half L0C boxes streamed back half by
#    half. Cross-split boundary boxes, [B,H,L,*] region copies:
#      WS_DV    dVnew  (B1, main dtype -> scan2b B7)
#      WS_DW    -dW    (B6 sign fold, main dtype -> scan2b B8)
#      WS_DKT   dKt    (B9, fp32 -> scan2b B10)
#      WS_DE    dE     (B5a, fp32 -> scan2b B8 merge / B13)
#      WS_DGAM2 dGam record (B3 init, fp32 -> scan2b B11 merge / B12; the
#                S7f-D dGam GM-accumulation record now lives in this box)
#      WS_DGAM  B9 dgam scalar row (fp32 -> scan2b B10-B13 merge)
#      WS_DK    B4 dK part (fp32 -> scan2b dK merge)
#    The raw QK^T stash rides workspace_lcc (read-before-clobber at B3); the
#    S7f-D dA record WS_DA moves to scan2b (its B7/B8 accumulation and B11
#    consumer now share one kernel).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[11, 12, 13, 14, 15, 16, 17, 18, 19],
              workspace_idx=[10, 20, 21, 22, 23, 24, 25, 26, 27, 28],
              pass_configs=pass_configs)
def gdn_bwd_v2_scan2a(B, H, L, DK, DV, C, BV, BK, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    bv_num = T.ceildiv(DV, BV)
    bk_num = T.ceildiv(DK, BK)
    n_tail = L - (chunk_num - 1) * C  # compile-time constant (L is compile-time)

    @T.prim_func
    def main(
        Q: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        K: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        V: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        DO: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        Beta: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore  WS_G (cumsum gam)
        A: T.Tensor([B, H, L, C], dtype),  # type: ignore  WS_A (solve_a)
        MskM: T.Tensor([C, C], "float"),  # type: ignore
        OnesDV: T.Tensor([DV, 16], dtype),  # type: ignore  ones follow main dtype (S7e)
        WS_H: T.Tensor([B, H, chunk_num + 1, DV, DK], dtype),  # type: ignore  scan1 out (GM)
        WS_D: T.Tensor([B, H, DV, DK], "float"),  # type: ignore  workspace (Delta' relay)
        DQ: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        DH0: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
        WS_DV: T.Tensor([B, H, L, DV], dtype),  # type: ignore  boundary out (S7f-E)
        WS_DW: T.Tensor([B, H, L, DK], dtype),  # type: ignore  boundary out (S7f-E)
        WS_DKT: T.Tensor([B, H, L, DK], "float"),  # type: ignore  boundary out (S7f-E)
        WS_DE: T.Tensor([B, H, L], "float"),  # type: ignore  boundary out (S7f-E)
        WS_DGAM2: T.Tensor([B, H, L, C], "float"),  # type: ignore  boundary out (S7f-E/D record)
        WS_DGAM: T.Tensor([B, H, L], "float"),  # type: ignore  boundary out (S7f-E)
        WS_DK: T.Tensor([B, H, L, DK], "float"),  # type: ignore  boundary out (S7f-E)
        workspace_ldk: T.Tensor([B * H, C, DK], accum_dtype),  # type: ignore  workspace
        workspace_lcc: T.Tensor([B * H, C, C], accum_dtype),  # type: ignore  workspace
        workspace_lcb: T.Tensor([B * H, C, BV], accum_dtype),  # type: ignore  workspace
        workspace_lbd: T.Tensor([B * H, BV, DK], accum_dtype),  # type: ignore  workspace (dH half dump)
        workspace_lc16: T.Tensor([B * H, C, 16], accum_dtype),  # type: ignore  workspace
        workspace_vdk: T.Tensor([B * H, C, DK], dtype),  # type: ignore  workspace (Kt halves)
        workspace_w: T.Tensor([B * H, C, DK], dtype),  # type: ignore  workspace (-W halves)
        workspace_vcc: T.Tensor([B * H, C, C], dtype),  # type: ignore  workspace
        workspace_vcb: T.Tensor([B * H, C, BV], dtype),  # type: ignore  workspace
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H

            # ---- L1 (Cube; DK/BK-half lanes) ----
            a_l1 = T.alloc_L1([C, C], dtype)
            p_l1 = T.alloc_L1([C, C], dtype)  # P (B1/B2 GEMM input)
            da_l1 = T.alloc_L1([C, C], dtype)  # d(QKt) (B4 GEMM input)
            k_l1 = T.alloc_L1([C, BK], dtype)  # K half (B4 dK GEMM input)
            q_l1 = T.alloc_L1([C, BK], dtype)  # Q half (B5a/B5c/B4)
            kt_l1 = T.alloc_L1([C, BK], dtype)  # th .* K half (B1)
            bek_l1 = T.alloc_L1([C, BK], dtype)  # bee .* K half (W derivation)
            w_full_l1 = T.alloc_L1([bk_num, C, BK], dtype)  # -W halves (S7g keep-alive box)
            hin_l1 = T.alloc_L1([BV, BK], dtype)  # H_in half (B5a/B5b/B6)
            dhp_l1 = T.alloc_L1([BV, BK], dtype)  # Delta' half (B1)
            bq_l1 = T.alloc_L1([C, BV], dtype)  # dO (B1/B2)
            vnew_l1 = T.alloc_L1([C, BV], dtype)  # Vnew (B2/B6b)
            t1_l1 = T.alloc_L1([C, BV], dtype)  # (E .* dO) / rowsum source
            bv_l1 = T.alloc_L1([C, BV], dtype)  # beta .* V (U0 GEMM input)
            onesbv_l1 = T.alloc_L1([BV, 16], dtype)  # rowsum over BV cols
            # ---- L0C (Cube) ----
            dp_l0 = T.alloc_L0C([C, C], accum_dtype)  # raw QK^T -> dP (reused box)
            u0_l0 = T.alloc_L0C([C, BV], accum_dtype)  # U0 -> W H^T half sum (reused)
            qh_l0 = T.alloc_L0C([C, BV], accum_dtype)  # QH half sum
            dvnew_l0 = T.alloc_L0C([C, BV], accum_dtype)  # dVnew
            rs_l0 = T.alloc_L0C([C, 16], accum_dtype)  # ones-GEMM rowsum sink
            dq0_l0 = T.alloc_L0C([C, BK], accum_dtype)  # dQ half accum (B5b -> B4)
            dq1_l0 = T.alloc_L0C([C, BK], accum_dtype)
            dh0_l0 = T.alloc_L0C([BV, BK], accum_dtype)  # dH half accum (B5c/B6b)
            dh1_l0 = T.alloc_L0C([BV, BK], accum_dtype)
            dw0_l0 = T.alloc_L0C([C, BK], accum_dtype)  # dW half accum (B6 -> B4 dK part)
            dw1_l0 = T.alloc_L0C([C, BK], accum_dtype)
            dkt0_l0 = T.alloc_L0C([C, BK], accum_dtype)  # dKt half accum (B9)
            dkt1_l0 = T.alloc_L0C([C, BK], accum_dtype)

            # ---- UB (Vector, fp32 accumulation) ----
            g_ub = T.alloc_ub([C], accum_dtype)  # gam
            e_ub = T.alloc_ub([C], accum_dtype)  # E = exp(gam)
            th_ub = T.alloc_ub([C], accum_dtype)  # theta = exp(gam_last - gam)
            zero_ub = T.alloc_ub([C], accum_dtype)  # negation source (fwd chunk_h idiom)
            be_ub = T.alloc_ub([C], accum_dtype)  # beta
            bee_ub = T.alloc_ub([C], accum_dtype)  # beta .* E
            dE_ub = T.alloc_ub([C], accum_dtype)  # dE accum (B5a)
            dgam_ub = T.alloc_ub([C], accum_dtype)  # B9 dgam scalar row
            glast_ub = T.alloc_ub([8], accum_dtype)  # exp(gam_{n-1})
            scal_ub = T.alloc_ub([8], accum_dtype)  # 32B cell (#1307)
            red1_ub = T.alloc_ub([BV], accum_dtype)  # B9 inner-product accum (over BV halves)
            redh_ub = T.alloc_ub([BV], accum_dtype)  # inner-product half stage
            rowb_ub = T.alloc_ub([1, BV], accum_dtype)  # B9 second reduce stage
            pk2_ub = T.alloc_ub([C, C], accum_dtype)  # dP (B2 accum)
            aux_ub = T.alloc_ub([C, C], accum_dtype)  # raw QK^T / M / Gam lanes (B3)
            da_ub = T.alloc_ub([C, C], accum_dtype)  # P -> dGam -> d(QKt)
            xk_ub = T.alloc_ub([C, BK], accum_dtype)  # K half / product lane
            bq_ub = T.alloc_ub([C, BV], accum_dtype)  # dO -> E .* dO lane
            wh_ub = T.alloc_ub([C, BV], accum_dtype)  # betaV -> U0 -> Vnew lane
            t1_ub = T.alloc_ub([C, BV], accum_dtype)  # dVnew / W H^T / QH lane
            x_ub = T.alloc_ub([BV, BK], accum_dtype)  # H_in half -> dH half (merged lane)
            dhp_ub = T.alloc_ub([BV, BK], accum_dtype)  # Delta' half
            rs_ub = T.alloc_ub([C, 16], accum_dtype)  # ones-GEMM rowsum sink
            # main-dtype staging for GM<->UB handoffs (dtype-matched copies).
            # S7g: all four boxes are pure transits (strict adjacent in/out
            # pairs) and C==BV==BK==64 makes every shape [64,64], so they share
            # ONE box -- the fp32-profile UB capacity fix (loud trace failure
            # if C/BV/BK ever diverge).
            io_cc = T.alloc_ub([C, C], dtype)
            io_xk = io_cc
            io_bq = io_cc
            io_hv = io_cc

            # ones-GEMM rowsum source: GM ones -> L1 (once per block)
            T.copy(OnesDV[0, 0], onesbv_l1)

            # ============ pass 2 front: reverse adjoint B1-B6/B3/B4/B9/B15 ====
            for c in T.serial(chunk_num):
                cc = chunk_num - 1 - c  # reversed chunk index
                # ---- replay state (B1/B5/B6/B9 sources) ----
                T.tile.fill(g_ub, 0.0)
                T.tile.fill(be_ub, 0.0)
                T.tile.fill(zero_ub, 0.0)
                T.copy(S[b, h, cc * C], g_ub)
                T.copy(Beta[b, h, cc * C], be_ub)
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                if cc == chunk_num - 1:
                    glast_ub[0] = g_ub[n_tail - 1]  # scalar, see scan1 glast note
                else:
                    glast_ub[0] = g_ub[C - 1]
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
                # ---- Kt / beK / -W derivations, BK-half blocked ----
                # Kt_hh = th .* K_hh -> workspace_vdk ; beK_hh = bee .* K_hh ->
                # bek_l1 -> W_hh = A beK_hh -> -W_hh -> workspace_w (B6 sign
                # fold kept at the derivation site, S7e :625 form)
                T.copy(A[b, h, cc * C, 0], a_l1)
                for hh in T.serial(bk_num):
                    T.copy(K[b, h, cc * C, hh * BK], io_xk)
                    T.copy(io_xk, xk_ub)
                    for (i, j) in T.Parallel(C, BK):
                        xk_ub[i, j] = xk_ub[i, j] * th_ub[i]
                    T.copy(xk_ub, io_xk)
                    T.copy(io_xk, workspace_vdk[cid, :, hh * BK])
                    T.copy(K[b, h, cc * C, hh * BK], io_xk)  # K reload (beK)
                    T.copy(io_xk, xk_ub)
                    for (i, j) in T.Parallel(C, BK):
                        xk_ub[i, j] = xk_ub[i, j] * bee_ub[i]
                    T.copy(xk_ub, io_xk)
                    T.copy(io_xk, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bek_l1)
                    T.gemm_v0(a_l1, bek_l1, dw0_l0, init=True)  # W half (temp)
                    T.copy(dw0_l0, workspace_ldk[cid, :, hh * BK])
                    T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    for (i, j) in T.Parallel(C, BK):
                        xk_ub[i, j] = 0.0 - xk_ub[i, j]  # -W half
                    T.copy(xk_ub, io_xk)
                    T.copy(io_xk, workspace_w[cid, :, hh * BK])
                    # S7g: single read-back keeps every half alive in L1 for the
                    # two later GEMM consumers (workspace_w stays strict 1W1R)
                    T.copy(workspace_w[cid, :, hh * BK], w_full_l1[hh, :, :])
                # ---- raw QK^T -> workspace_lcc stash (B3 read-before-clobber)
                #      P = (raw .* Gam) .* M -> p_l1 (S7e :640-646 order) ----
                for hh in T.serial(bk_num):
                    T.copy(Q[b, h, cc * C, hh * BK], q_l1)  # GM->L1 direct (S7e :630 idiom)
                    # K goes GM->L1 directly (scan2b :1043 idiom). Routing it
                    # through io_xk/xk_ub spilled io_xk to a GM workspace
                    # (workspace_1) under C-segment liveness, and the spill
                    # readback copy was emitted as
                    #   copy_gm_to_l1(k_l1, workspace_1, 64, 0, 0, true)
                    # (stride=0, rows=0 -- the only zero-arg copy in the
                    # 833-line codegen) -> on-card 507015 unaligned.
                    T.copy(K[b, h, cc * C, hh * BK], k_l1)
                    T.gemm_v0(q_l1, k_l1, dp_l0, transpose_B=True, init=(hh == 0))
                T.copy(dp_l0, workspace_lcc[cid, :, :])  # stash raw QK^T
                # Gamma (aux lane, 3-loop exp idiom) -> P assembly
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = g_ub[i] - g_ub[j]
                # Gamma = exp(gam_r - gam_c) per design L49 / CPU golden
                # (test_gdn_bwd_v2.py:237). The neg that used to sit here
                # flipped it to exp(gam_c - gam_r): the lower triangle then
                # reached ~1e13, P overflowed fp16 to inf and the mask mul
                # turned it into nan (w11 diag: s2a out0/3/4/8 nan). The old
                # `zero_ub[i] - aux_ub[i, j]` spelling (which carried that
                # neg) also lowered to a 4B-stepped, non-32B-aligned VEC
                # address in a 2D Parallel -> on-card 507015 (fixp 0xd8);
                # both problems disappear with the direct exp below.
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = T.exp(aux_ub[i, j])
                T.copy(workspace_lcc[cid, :, :], da_ub)  # raw
                # UB-side raw keep-alive (pk2 is dead until the dP dump below;
                # keeps workspace_lcc at strict W-R-W-R sync pairing, S7f fix-2)
                T.copy(da_ub, pk2_ub)
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = da_ub[i, j] * aux_ub[i, j]  # raw .* Gam
                T.copy(MskM[0, 0], aux_ub)  # M reload (Gam dead here)
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = da_ub[i, j] * aux_ub[i, j]  # P
                T.copy(da_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                # ---- 1D accumulators ----
                T.tile.fill(dE_ub, 0.0)
                T.tile.fill(dgam_ub, 0.0)
                # ---- BV blocks: B1/B2/B5/B6/B9 (+ boundary writes) ----
                for bv in T.serial(bv_num):
                    # dO block -> bq_l1
                    T.tile.fill(bq_ub, 0.0)
                    T.copy(DO[b, h, cc * C, bv * BV], io_bq)
                    T.copy(io_bq, bq_ub)
                    T.copy(bq_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bq_l1)
                    # beta .* V -> bv_l1 ; U0 = A (beta .* V) -> wh_ub
                    T.tile.fill(wh_ub, 0.0)
                    T.copy(V[b, h, cc * C, bv * BV], io_bq)
                    T.copy(io_bq, wh_ub)
                    for (i, j) in T.Parallel(C, BV):
                        wh_ub[i, j] = wh_ub[i, j] * be_ub[i]
                    T.copy(wh_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bv_l1)
                    T.gemm_v0(a_l1, bv_l1, u0_l0, init=True)
                    T.copy(u0_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], wh_ub)  # wh = U0
                    # ---- B1 half-loop: dVnew = P^T dO + sum_hh Kt_hh Dhp_hh^T,
                    #      with the B9 inner product <H_in, Delta'> folded into
                    #      the same half pass (x_ub dies here; the B5/B6
                    #      half-loops reload H_in from WS_H via GM->L1 direct
                    #      copies, Q->q_l1 idiom -- S7g: workspace_vbd gone) ----
                    T.gemm_v0(p_l1, bq_l1, dvnew_l0, transpose_A=True, init=True)
                    T.tile.fill(red1_ub, 0.0)
                    for hh in T.serial(bk_num):
                        T.tile.fill(x_ub, 0.0)
                        T.copy(WS_H[b, h, cc, bv * BV, hh * BK], io_hv)
                        T.copy(io_hv, x_ub)  # H_in half
                        T.tile.fill(dhp_ub, 0.0)
                        T.copy(WS_D[b, h, bv * BV, hh * BK], dhp_ub)  # Delta' half (fp32 GM)
                        for (i, j) in T.Parallel(BV, BK):
                            x_ub[i, j] = x_ub[i, j] * dhp_ub[i, j]
                        T.reduce_sum(x_ub, redh_ub, dim=1)
                        for j in T.Parallel(BV):
                            red1_ub[j] = red1_ub[j] + redh_ub[j]
                        T.copy(dhp_ub, io_hv)
                        T.copy(io_hv, workspace_vcc[cid, :, :])
                        T.copy(workspace_vcc[cid, :, :], dhp_l1)
                        T.copy(workspace_vdk[cid, :, hh * BK], kt_l1)
                        T.gemm_v0(kt_l1, dhp_l1, dvnew_l0, transpose_B=True, init=False)
                    # B9 scalar: dgam_{n-1} += exp(gam_{n-1}) <H_in, Delta'>
                    for j in T.Parallel(BV):
                        rowb_ub[0, j] = red1_ub[j]
                    T.reduce_sum(rowb_ub, scal_ub[0:1], dim=1)
                    if cc == chunk_num - 1:
                        dgam_ub[n_tail - 1] = dgam_ub[n_tail - 1] + scal_ub[0] * glast_ub[0]
                    else:
                        dgam_ub[C - 1] = dgam_ub[C - 1] + scal_ub[0] * glast_ub[0]
                    # dVnew -> WS_DV boundary (main dtype)
                    T.copy(dvnew_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1_ub)
                    T.copy(t1_ub, io_bq)
                    T.copy(io_bq, WS_DV[b, h, cc * C, bv * BV])
                    # ---- W H^T half sum -> Vnew = U0 + (-W) H^T (S7e :685-689);
                    #      L0C box clears via init=(hh==0) only (fill on L0C is
                    #      illegal: tl.ascend_fill mixes Cube/Vector, S7f erratum)
                    for hh in T.serial(bk_num):
                        T.copy(WS_H[b, h, cc, bv * BV, hh * BK], hin_l1)  # H_in reload (GM->L1 direct)
                        T.gemm_v0(w_full_l1[hh, :, :], hin_l1, u0_l0, transpose_B=True, init=(hh == 0))
                    T.copy(u0_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1_ub)  # W H^T
                    for (i, j) in T.Parallel(C, BV):
                        wh_ub[i, j] = wh_ub[i, j] + t1_ub[i, j]  # Vnew
                    T.copy(wh_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], vnew_l1)
                    # ---- B5a half-loop: QH = sum_hh Q_hh H_in_hh^T (init=(hh==0) clear) ----
                    for hh in T.serial(bk_num):
                        T.copy(Q[b, h, cc * C, hh * BK], q_l1)
                        T.copy(WS_H[b, h, cc, bv * BV, hh * BK], hin_l1)  # H_in reload (GM->L1 direct)
                        T.gemm_v0(q_l1, hin_l1, qh_l0, transpose_B=True, init=(hh == 0))
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
                    # (E .* dO) -> t1_l1 (B5b/B5c source)
                    for (i, j) in T.Parallel(C, BV):
                        bq_ub[i, j] = bq_ub[i, j] * e_ub[i]
                    T.copy(bq_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], t1_l1)
                    # ---- B2: dP += dO Vnew^T (full-BV contraction) ----
                    T.gemm_v0(bq_l1, vnew_l1, dp_l0, transpose_B=True, init=(bv == 0))
                    # ---- B5b/B5c/B6 half-loop ----
                    T.copy(dvnew_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1_ub)
                    T.copy(t1_ub, io_bq)
                    T.copy(io_bq, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bq_l1)  # dVnew L1 (B6)
                    for hh in T.serial(bk_num):
                        T.copy(WS_H[b, h, cc, bv * BV, hh * BK], hin_l1)  # H_in reload (GM->L1 direct)
                        T.copy(Q[b, h, cc * C, hh * BK], q_l1)
                        # B5b: dQ_hh += (E .* dO) H_in_hh
                        if hh == 0:
                            T.gemm_v0(t1_l1, hin_l1, dq0_l0, init=(bv == 0))
                        else:
                            T.gemm_v0(t1_l1, hin_l1, dq1_l0, init=(bv == 0))
                        # B5c: dH_hh += (E .* dO)^T Q_hh
                        if hh == 0:
                            T.gemm_v0(t1_l1, q_l1, dh0_l0, transpose_A=True, init=True)
                        else:
                            T.gemm_v0(t1_l1, q_l1, dh1_l0, transpose_A=True, init=True)
                        # B6: dW_hh += dVnew H_in_hh
                        if hh == 0:
                            T.gemm_v0(bq_l1, hin_l1, dw0_l0, init=(bv == 0))
                        else:
                            T.gemm_v0(bq_l1, hin_l1, dw1_l0, init=(bv == 0))
                    # ---- B6b half-loop: dH_hh += Vnew^T (-W_hh) ----
                    for hh in T.serial(bk_num):
                        if hh == 0:
                            T.gemm_v0(vnew_l1, w_full_l1[hh, :, :], dh0_l0, transpose_A=True, init=False)
                        else:
                            T.gemm_v0(vnew_l1, w_full_l1[hh, :, :], dh1_l0, transpose_A=True, init=False)
                    # ---- B15: dH half -> WS_D relay (dh0 for the first chunk);
                    #      B9 vec part: dH += exp(gam_{n-1}) Delta' ----
                    for hh in T.serial(bk_num):
                        T.tile.fill(dhp_ub, 0.0)
                        T.copy(WS_D[b, h, bv * BV, hh * BK], dhp_ub)  # Delta' half reload
                        T.tile.fill(x_ub, 0.0)
                        if hh == 0:
                            T.copy(dh0_l0, workspace_lbd[cid, :, hh * BK])
                            T.copy(workspace_lbd[cid, :, hh * BK], x_ub)
                        else:
                            T.copy(dh1_l0, workspace_lbd[cid, :, hh * BK])
                            T.copy(workspace_lbd[cid, :, hh * BK], x_ub)
                        for (i, j) in T.Parallel(BV, BK):
                            x_ub[i, j] = x_ub[i, j] + dhp_ub[i, j] * glast_ub[0]
                        # S7f: WS_D is fp32 GM -- fp32 ub direct copy (:750 form)
                        T.copy(x_ub, WS_D[b, h, bv * BV, hh * BK])
                        if cc == 0:
                            T.copy(x_ub, io_hv)
                            T.copy(io_hv, DH0[b, h, bv * BV, hh * BK])
                # ---- dW -> WS_DW boundary (sign fold, S7e :789-793; the dw
                #      half boxes become the B4 dK-part accumulators after this
                #      write-back) ----
                for hh in T.serial(bk_num):
                    if hh == 0:
                        T.copy(dw0_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    else:
                        T.copy(dw1_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    for (i, j) in T.Parallel(C, BK):
                        xk_ub[i, j] = 0.0 - xk_ub[i, j]  # dW = -sum dVnew H_in
                    T.copy(xk_ub, io_xk)
                    T.copy(io_xk, WS_DW[b, h, cc * C, hh * BK])
                # ---- B3: d(QKt) = dP .* Gam .* M ; dGam = (dP .* M) .* raw
                #      (aux lane takes raw QK^T from the UB-side keep-alive
                #      box; exact S7e multiply order kept) ----
                T.copy(pk2_ub, aux_ub)  # raw QK^T (UB-side keep-alive)
                T.copy(dp_l0, workspace_lcc[cid, :, :])  # dP dump clobbers stash
                T.copy(workspace_lcc[cid, :, :], pk2_ub)  # dP from B2 accum
                T.copy(MskM[0, 0], da_ub)  # M reload
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = pk2_ub[i, j] * da_ub[i, j]  # dP .* M
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = da_ub[i, j] * aux_ub[i, j]  # (dP .* M) .* raw = dGam
                # S7f-D/E: dGam accumulation record -> GM boundary (H3; the
                # B11 merge in scan2b continues this record)
                T.copy(da_ub, WS_DGAM2[b, h, cc * C, 0])
                # dA = (dP .* Gam) .* M (aux recomputes Gamma, 3-loop idiom;
                # same sign fix as the P assembly above)
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = g_ub[i] - g_ub[j]
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = T.exp(aux_ub[i, j])
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = pk2_ub[i, j] * aux_ub[i, j]  # dP .* Gam
                T.copy(MskM[0, 0], aux_ub)  # M reload (Gam dead)
                for (i, j) in T.Parallel(C, C):
                    da_ub[i, j] = da_ub[i, j] * aux_ub[i, j]  # d(QKt)
                # ---- B4: dQ_hh += d(QKt) K_hh ; dK part_hh = d(QKt)^T Q_hh
                #      (the B4 dK part reuses the dw half boxes: dW is written
                #      out to its boundary below first) ----
                T.copy(da_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                for hh in T.serial(bk_num):
                    T.copy(K[b, h, cc * C, hh * BK], k_l1)
                    T.copy(Q[b, h, cc * C, hh * BK], q_l1)
                    if hh == 0:
                        T.gemm_v0(da_l1, k_l1, dq0_l0, init=False)
                        T.gemm_v0(da_l1, q_l1, dw0_l0, transpose_A=True, init=True)
                        T.copy(dw0_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    else:
                        T.gemm_v0(da_l1, k_l1, dq1_l0, init=False)
                        T.gemm_v0(da_l1, q_l1, dw1_l0, transpose_A=True, init=True)
                        T.copy(dw1_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    T.copy(xk_ub, WS_DK[b, h, cc * C, hh * BK])  # fp32 boundary
                # ---- dW boundary (sign fold, S7e :789-790) + dKt boundary ----
                for hh in T.serial(bk_num):
                    if hh == 0:
                        T.copy(dkt0_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    else:
                        T.copy(dkt1_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    if _S2A_DBG:
                        # debug only: expose dqkt instead of dKt (see _S2A_DBG)
                        T.copy(da_ub, WS_DKT[b, h, cc * C, hh * BK])
                    else:
                        T.copy(xk_ub, WS_DKT[b, h, cc * C, hh * BK])  # fp32 boundary
                # ---- dE / dgam scalar boundaries + DQ write-back ----
                T.copy(dE_ub, WS_DE[b, h, cc * C])
                T.copy(dgam_ub, WS_DGAM[b, h, cc * C])
                for hh in T.serial(bk_num):
                    if hh == 0:
                        T.copy(dq0_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    else:
                        T.copy(dq1_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    T.copy(xk_ub, io_xk)
                    T.copy(io_xk, DQ[b, h, cc * C, hh * BK])

    return main


# ---------------------------------------------------------------------------
# 5) gdn_bwd_v2_scan2b: pass 2 back window (B7/B8 + B10-B15; design Sec 3) ->
#    DKG / DVG / DBeta / DG. Consumes the seven S7f-E boundary boxes written
#    by scan2a (GM only cross-kernel state, H3). The dA gradient accumulates
#    in-kernel (B7 BV sum + B8 half products) and keeps the S7f-D WS_DA GM
#    record discipline (dump at :B8 end, read back at B11). DK-wide outputs
#    (dk) accumulate into per-half L0C/UB boxes, same BK=64 half blocking as
#    scan2a. Gam (Gamma) is recomputed on demand (aux/dx lanes, 3-loop exp
#    idiom) instead of kept resident -- keeps the UB static sum at 187.2 KiB.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[16, 17, 18, 19], workspace_idx=[20, 21, 22, 23, 24, 25, 26, 27],
              pass_configs=pass_configs)
def gdn_bwd_v2_scan2b(B, H, L, DK, DV, C, BV, BK, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    bv_num = T.ceildiv(DV, BV)
    bk_num = T.ceildiv(DK, BK)
    n_tail = L - (chunk_num - 1) * C  # compile-time constant (L is compile-time)

    @T.prim_func
    def main(
        K: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        V: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        Beta: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore  WS_G (cumsum gam)
        A: T.Tensor([B, H, L, C], dtype),  # type: ignore  WS_A (solve_a)
        MskJ: T.Tensor([C, C], "float"),  # type: ignore
        OnesDV: T.Tensor([DV, 16], dtype),  # type: ignore  ones follow main dtype (S7e)
        OnesDK: T.Tensor([DK, 16], dtype),  # type: ignore
        OnesC: T.Tensor([C, 16], dtype),  # type: ignore
        WS_DV: T.Tensor([B, H, L, DV], dtype),  # type: ignore  boundary in (S7f-E)
        WS_DW: T.Tensor([B, H, L, DK], dtype),  # type: ignore  boundary in (S7f-E)
        WS_DKT: T.Tensor([B, H, L, DK], "float"),  # type: ignore  boundary in (S7f-E)
        WS_DE: T.Tensor([B, H, L], "float"),  # type: ignore  boundary in (S7f-E)
        WS_DGAM2: T.Tensor([B, H, L, C], "float"),  # type: ignore  boundary in/out (S7f-E/D record)
        WS_DGAM: T.Tensor([B, H, L], "float"),  # type: ignore  boundary in (S7f-E)
        WS_DK: T.Tensor([B, H, L, DK], "float"),  # type: ignore  boundary in (S7f-E)
        DKG: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        DVG: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        DBeta: T.Tensor([B, H, L], "float"),  # type: ignore
        DG: T.Tensor([B, H, L], "float"),  # type: ignore
        WS_DA: T.Tensor([B, H, C, C], "float"),  # type: ignore  workspace (S7f-D dA record)
        workspace_ldk: T.Tensor([B * H, C, DK], accum_dtype),  # type: ignore  workspace
        workspace_lcc: T.Tensor([B * H, C, C], accum_dtype),  # type: ignore  workspace
        workspace_lcb: T.Tensor([B * H, C, BV], accum_dtype),  # type: ignore  workspace
        workspace_lc16: T.Tensor([B * H, C, 16], accum_dtype),  # type: ignore  workspace
        workspace_vdk: T.Tensor([B * H, C, DK], dtype),  # type: ignore  workspace
        workspace_vcc: T.Tensor([B * H, C, C], dtype),  # type: ignore  workspace
        workspace_vcb: T.Tensor([B * H, C, BV], dtype),  # type: ignore  workspace
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H

            # ---- L1 (Cube) ----
            a_l1 = T.alloc_L1([C, C], dtype)  # A (B7/B8/B11 GEMM input)
            da_l1 = T.alloc_L1([C, C], dtype)  # dA / dsum staging lane
            p_l1 = T.alloc_L1([C, C], dtype)  # dX+dX^T / rowsum source
            k_l1 = T.alloc_L1([C, BK], dtype)  # K half (B11 dK GEMM input)
            xk_l1 = T.alloc_L1([C, BK], dtype)  # dW half / K.*X product lane
            bek_l1 = T.alloc_L1([C, BK], dtype)  # bee .* K half (B8)
            bq_l1 = T.alloc_L1([C, BV], dtype)  # dVnew (B7 dA GEMM input)
            bv_l1 = T.alloc_L1([C, BV], dtype)  # beta .* V (B7 dA GEMM input)
            t1_l1 = T.alloc_L1([C, BV], dtype)  # V .* t1 rowsum source (B7)
            onesdv_l1 = T.alloc_L1([BV, 16], dtype)
            onesdk_l1 = T.alloc_L1([BK, 16], dtype)  # S7g: half-width ones (BK-blocked rowsums)
            onesc_l1 = T.alloc_L1([C, 16], dtype)
            # ---- L0C (Cube) ----
            kk_l0 = T.alloc_L0C([C, C], accum_dtype)  # KK^T
            da_l0 = T.alloc_L0C([C, C], accum_dtype)  # dA accum (B7 BV + B8 halves)
            dsum_l0 = T.alloc_L0C([C, C], accum_dtype)  # A^T dA (B11)
            dx_l0 = T.alloc_L0C([C, C], accum_dtype)  # (A^T dA) A^T (B11)
            t1_l0 = T.alloc_L0C([C, BV], accum_dtype)  # A^T dVnew (B7)
            t20_l0 = T.alloc_L0C([C, BK], accum_dtype)  # t2 half (B8)
            t21_l0 = T.alloc_L0C([C, BK], accum_dtype)
            dkk0_l0 = T.alloc_L0C([C, BK], accum_dtype)  # (dX+dX^T) K half (B11)
            dkk1_l0 = T.alloc_L0C([C, BK], accum_dtype)
            rs2_l0 = T.alloc_L0C([C, 16], accum_dtype)  # ones-GEMM sinks
            rs3_l0 = T.alloc_L0C([C, 16], accum_dtype)
            rs4_l0 = T.alloc_L0C([C, 16], accum_dtype)

            # ---- UB (Vector, fp32 accumulation) ----
            g_ub = T.alloc_ub([C], accum_dtype)  # gam
            e_ub = T.alloc_ub([C], accum_dtype)  # E = exp(gam)
            th_ub = T.alloc_ub([C], accum_dtype)  # theta = exp(gam_last - gam)
            zero_ub = T.alloc_ub([C], accum_dtype)  # negation source (fwd chunk_h idiom)
            be_ub = T.alloc_ub([C], accum_dtype)  # beta
            bee_ub = T.alloc_ub([C], accum_dtype)  # beta .* E
            dE_ub = T.alloc_ub([C], accum_dtype)  # dE (WS_DE in + B8 merge)
            dbe_ub = T.alloc_ub([C], accum_dtype)  # d(betaE) (B8)
            tmp1_ub = T.alloc_ub([C], accum_dtype)  # dbeta accumulation
            tmp2_ub = T.alloc_ub([C], accum_dtype)  # dth / dg (B10/B14)
            dgam_ub = T.alloc_ub([C], accum_dtype)  # dgam accumulation
            glast_ub = T.alloc_ub([8], accum_dtype)  # exp(gam_{n-1}) (B10 tail cell)
            scal_ub = T.alloc_ub([8], accum_dtype)  # 32B cell (#1307)
            rowc_ub = T.alloc_ub([1, C], accum_dtype)  # B10 second reduce stage
            kk_ub = T.alloc_ub([C, C], accum_dtype)  # KK^T (resident)
            dgam2_ub = T.alloc_ub([C, C], accum_dtype)  # dGam record (B3 part in + B11 merge)
            dx_ub = T.alloc_ub([C, C], accum_dtype)  # dL / dX^T lane
            aux_ub = T.alloc_ub([C, C], accum_dtype)  # Gam / mul2 / dX lanes
            xk_ub = T.alloc_ub([C, BK], accum_dtype)  # dW / dKt / t2 product lane
            k2_ub = T.alloc_ub([C, BK], accum_dtype)  # K half fp32 reload lane
            dk0_ub = T.alloc_ub([C, BK], accum_dtype)  # dK half accum
            dk1_ub = T.alloc_ub([C, BK], accum_dtype)
            wh2_ub = T.alloc_ub([C, BV], accum_dtype)  # V .* t1 lane
            t1b_ub = T.alloc_ub([C, BV], accum_dtype)  # t1 replay lane
            rs_ub = T.alloc_ub([C, 16], accum_dtype)  # ones-GEMM row/col-sum sink
            # main-dtype staging for GM<->UB handoffs (dtype-matched copies).
            # S7g: pure transits share ONE box (fp32-profile UB capacity fix;
            # C==BV==BK==64 identical [64,64] shapes, loud failure if diverge).
            io_cc = T.alloc_ub([C, C], dtype)
            io_xk = io_cc
            io_bq2 = io_cc

            # ones-GEMM row/col-sum sources: GM ones -> L1 (once per block)
            T.copy(OnesDV[0, 0], onesdv_l1)
            T.copy(OnesDK[0, 0], onesdk_l1)
            T.copy(OnesC[0, 0], onesc_l1)

            # ============ pass 2 back: reverse adjoint B7/B8/B10-B15 =========
            for c in T.serial(chunk_num):
                cc = chunk_num - 1 - c  # reversed chunk index
                # ---- replay state ----
                T.tile.fill(g_ub, 0.0)
                T.tile.fill(be_ub, 0.0)
                T.tile.fill(zero_ub, 0.0)
                T.copy(S[b, h, cc * C], g_ub)
                T.copy(Beta[b, h, cc * C], be_ub)
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                if cc == chunk_num - 1:
                    glast_ub[0] = g_ub[n_tail - 1]  # scalar, see scan1 glast note
                else:
                    glast_ub[0] = g_ub[C - 1]
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
                # ---- KK^T = sum_hh K_hh K_hh^T (half-blocked) ----
                for hh in T.serial(bk_num):
                    T.copy(K[b, h, cc * C, hh * BK], k_l1)  # GM->L1 direct
                    T.gemm_v0(k_l1, k_l1, kk_l0, transpose_B=True, init=(hh == 0))
                T.copy(kk_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], kk_ub)
                # ---- boundary in: dE / dgam / dGam(B3 part) / dK(B4 part) ----
                T.copy(WS_DE[b, h, cc * C], dE_ub)
                T.copy(WS_DGAM[b, h, cc * C], dgam_ub)
                T.copy(WS_DGAM2[b, h, cc * C, 0], dgam2_ub)
                # ---- 1D accumulators ----
                T.tile.fill(tmp1_ub, 0.0)  # dbeta accumulation
                # ---- BV blocks: B7 ----
                for bv in T.serial(bv_num):
                    # dVnew block from the boundary (GM->L1 direct, main dtype)
                    T.copy(WS_DV[b, h, cc * C, bv * BV], bq_l1)
                    # beta .* V -> bv_l1
                    T.tile.fill(wh2_ub, 0.0)
                    T.copy(V[b, h, cc * C, bv * BV], io_bq2)
                    T.copy(io_bq2, wh2_ub)
                    # S7g: serial broadcast-free form -- T.Parallel cross-dim
                    # refs (be_ub[i]) materialize [C,BV] broadcast temps
                    # (ascend_lower_parallel_to_vector CreateBroadcastBuffer),
                    # which overflow the fp32 TSC UB budget; serial loops fall
                    # back to scalar looping (element-wise, semantics equal).
                    for i in T.serial(C):
                        for j in T.serial(C):
                            wh2_ub[i, j] = wh2_ub[i, j] * be_ub[i]
                    T.copy(wh2_ub, io_bq2)
                    T.copy(io_bq2, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], bv_l1)
                    # t1 = A^T dVnew
                    T.gemm_v0(a_l1, bq_l1, t1_l0, transpose_A=True, init=True)
                    T.copy(t1_l0, workspace_lcb[cid, :, :])
                    T.copy(workspace_lcb[cid, :, :], t1b_ub)  # t1 block
                    # dbeta += rowsum(V .* t1) (S7e :761-772)
                    T.copy(V[b, h, cc * C, bv * BV], io_bq2)
                    T.copy(io_bq2, wh2_ub)  # V block
                    for (i, j) in T.Parallel(C, BV):
                        wh2_ub[i, j] = wh2_ub[i, j] * t1b_ub[i, j]  # V .* t1
                    T.copy(wh2_ub, io_bq2)
                    T.copy(io_bq2, workspace_vcb[cid, :, :])
                    T.copy(workspace_vcb[cid, :, :], t1_l1)
                    T.gemm_v0(t1_l1, onesdv_l1, rs3_l0, init=True)
                    T.copy(rs3_l0, workspace_lc16[cid, :, :])
                    T.copy(workspace_lc16[cid, :, :], rs_ub)
                    for i in T.serial(C):
                        tmp1_ub[i] = tmp1_ub[i] + rs_ub[i, 0]  # dbeta7 accumulate
                    # dV = beta .* t1 -> DVG
                    for i in T.serial(C):  # S7g: serial (no broadcast temp)
                        for j in T.serial(C):
                            t1b_ub[i, j] = t1b_ub[i, j] * be_ub[i]
                    T.copy(t1b_ub, io_bq2)
                    T.copy(io_bq2, DVG[b, h, cc * C, bv * BV])
                    # dA += dVnew (beta .* V)^T (BV accum)
                    T.gemm_v0(bq_l1, bv_l1, da_l0, transpose_B=True, init=(bv == 0))
                # ---- B8 half-loop: dA += -dW beK^T ; t2 = A^T (-dW) ----
                for hh in T.serial(bk_num):
                    T.copy(WS_DW[b, h, cc * C, hh * BK], xk_l1)  # GM->L1 direct
                    # beK_hh = bee .* K_hh -> bek_l1 (k2 lane)
                    T.tile.fill(k2_ub, 0.0)
                    T.copy(K[b, h, cc * C, hh * BK], io_xk)
                    T.copy(io_xk, k2_ub)
                    for i in T.serial(C):  # S7g: serial (no broadcast temp)
                        for j in T.serial(C):
                            k2_ub[i, j] = k2_ub[i, j] * bee_ub[i]
                    T.copy(k2_ub, io_xk)
                    T.copy(io_xk, workspace_vdk[cid, :, :])
                    T.copy(workspace_vdk[cid, :, :], bek_l1)
                    T.gemm_v0(xk_l1, bek_l1, da_l0, transpose_B=True, init=False)
                    if hh == 0:
                        T.gemm_v0(a_l1, xk_l1, t20_l0, transpose_A=True, init=True)
                    else:
                        T.gemm_v0(a_l1, xk_l1, t21_l0, transpose_A=True, init=True)
                    # d(betaE) = rowsum(K .* t2) (k2 lane reloads K, S7e :805-821)
                    if hh == 0:
                        T.copy(t20_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)  # xk = t2 half
                    else:
                        T.copy(t21_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)  # xk = t2 half
                    T.tile.fill(k2_ub, 0.0)
                    T.copy(K[b, h, cc * C, hh * BK], io_xk)
                    T.copy(io_xk, k2_ub)
                    for (i, j) in T.Parallel(C, BK):
                        k2_ub[i, j] = k2_ub[i, j] * xk_ub[i, j]  # K .* t2
                    T.copy(k2_ub, io_xk)
                    T.copy(io_xk, workspace_vdk[cid, :, :])
                    T.copy(workspace_vdk[cid, :, :], xk_l1)
                    T.gemm_v0(xk_l1, onesdk_l1, rs4_l0, init=True)
                    T.copy(rs4_l0, workspace_lc16[cid, :, :])
                    T.copy(workspace_lc16[cid, :, :], rs_ub)
                    for i in T.serial(C):
                        dbe_ub[i] = rs_ub[i, 0]
                    # dk_hh = bee .* t2 (first-assigned, S7e :803-804)
                    if hh == 0:
                        for i in T.serial(C):  # S7g: serial (no broadcast temp)
                            for j in T.serial(C):
                                dk0_ub[i, j] = xk_ub[i, j] * bee_ub[i]
                    else:
                        for i in T.serial(C):  # S7g: serial (no broadcast temp)
                            for j in T.serial(C):
                                dk1_ub[i, j] = xk_ub[i, j] * bee_ub[i]
                    # dE += beta .* d(betaE) ; dbeta += E .* d(betaE)
                    for i in T.Parallel(C):
                        dE_ub[i] = dE_ub[i] + be_ub[i] * dbe_ub[i]
                    for i in T.Parallel(C):
                        tmp1_ub[i] = tmp1_ub[i] + e_ub[i] * dbe_ub[i]
                # ---- dA GM record (S7f-D discipline: dump + read-back at B11)
                #      da_l0 dies into the record; the dw half boxes are free
                #      after the WS_DW write in scan2a's sibling window ----
                T.copy(da_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], aux_ub)
                T.copy(aux_ub, WS_DA[b, h, 0, 0])
                # ---- B11: dL = J .* (-(A^T dA) A^T) (S7e :849-905 order) ----
                T.copy(WS_DA[b, h, 0, 0], aux_ub)  # S7f-D: dA from the GM record
                T.copy(aux_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                T.gemm_v0(a_l1, da_l1, dsum_l0, transpose_A=True, init=True)
                # dsum staged into the main-dtype L1 lane (S7e erratum form)
                T.copy(dsum_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], aux_ub)
                T.copy(aux_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], da_l1)
                T.gemm_v0(da_l1, a_l1, dx_l0, transpose_B=True, init=True)  # (A^T dA) A^T
                T.copy(dx_l0, workspace_lcc[cid, :, :])
                T.copy(workspace_lcc[cid, :, :], dx_ub)
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = 0.0 - dx_ub[i, j]
                T.copy(MskJ[0, 0], aux_ub)  # J reload (aux lane)
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = dx_ub[i, j] * aux_ub[i, j]  # dL
                # dbeta += rowsum(dL .* Gam .* KK^T) -- Gamma recomputed into
                # the aux lane (3-loop exp idiom), then aux = dL .* Gam
                # (no neg -- see scan2a Gamma note)
                for i in T.serial(C):  # S7g: serial (no broadcast temp)
                    for j in T.serial(C):
                        aux_ub[i, j] = g_ub[i] - g_ub[j]
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = T.exp(aux_ub[i, j])
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = dx_ub[i, j] * aux_ub[i, j]  # dL .* Gam
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = aux_ub[i, j] * kk_ub[i, j]  # .* KK^T
                T.copy(aux_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                T.gemm_v0(p_l1, onesc_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    tmp1_ub[i] = tmp1_ub[i] + rs_ub[i, 0]
                # dGam += dL .* beta .* KK^T ; dX = dL .* beta .* Gam
                for i in T.serial(C):  # S7g: serial (no broadcast temp)
                    for j in T.serial(C):
                        aux_ub[i, j] = dx_ub[i, j] * be_ub[i]  # dL .* beta (mul2)
                for (i, j) in T.Parallel(C, C):
                    dgam2_ub[i, j] = dgam2_ub[i, j] + aux_ub[i, j] * kk_ub[i, j]
                # S7f-D/E: dGam record update (B3 part + B11 merge)
                T.copy(dgam2_ub, WS_DGAM2[b, h, cc * C, 0])
                # dX = mul2 .* Gam (Gamma recomputed into the dx lane; dL is
                # dead after the two products above; no neg -- see scan2a note)
                for i in T.serial(C):  # S7g: serial (no broadcast temp)
                    for j in T.serial(C):
                        dx_ub[i, j] = g_ub[i] - g_ub[j]
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = T.exp(dx_ub[i, j])
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = aux_ub[i, j] * dx_ub[i, j]  # dX
                # dK += (dX + dX^T) K (half-blocked)
                T.tile.transpose(dx_ub, aux_ub)  # dx = dX^T (dst, src)
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = dx_ub[i, j] + aux_ub[i, j]
                T.copy(dx_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                for hh in T.serial(bk_num):
                    T.copy(K[b, h, cc * C, hh * BK], k_l1)
                    if hh == 0:
                        T.gemm_v0(p_l1, k_l1, dkk0_l0, init=True)
                        T.copy(dkk0_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    else:
                        T.gemm_v0(p_l1, k_l1, dkk1_l0, init=True)
                        T.copy(dkk1_l0, workspace_ldk[cid, :, hh * BK])
                        T.copy(workspace_ldk[cid, :, hh * BK], xk_ub)
                    if hh == 0:
                        for (i, j) in T.Parallel(C, BK):
                            dk0_ub[i, j] = dk0_ub[i, j] + xk_ub[i, j]
                    else:
                        for (i, j) in T.Parallel(C, BK):
                            dk1_ub[i, j] = dk1_ub[i, j] + xk_ub[i, j]
                # ---- B12: dgam += rowsum(dGam.*Gam) - colsum(dGam.*Gam)
                #      (matrix-form handoff preserved verbatim, S7e :906-925;
                #      Gamma recomputed into the dx lane) ----
                for i in T.serial(C):  # S7g: serial (no broadcast temp)
                    for j in T.serial(C):
                        dx_ub[i, j] = g_ub[i] - g_ub[j]
                for (i, j) in T.Parallel(C, C):
                    dx_ub[i, j] = T.exp(dx_ub[i, j])  # B12 Gam, no neg (scan2a note)
                for (i, j) in T.Parallel(C, C):
                    dgam2_ub[i, j] = dgam2_ub[i, j] * dx_ub[i, j]
                T.copy(dgam2_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                T.gemm_v0(p_l1, onesc_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    dgam_ub[i] = dgam_ub[i] + rs_ub[i, 0]
                T.tile.transpose(aux_ub, dgam2_ub)  # aux = dGam^T (dst, src)
                T.copy(aux_ub, io_cc)
                T.copy(io_cc, workspace_vcc[cid, :, :])
                T.copy(workspace_vcc[cid, :, :], p_l1)
                T.gemm_v0(p_l1, onesc_l1, rs2_l0, init=True)
                T.copy(rs2_l0, workspace_lc16[cid, :, :])
                T.copy(workspace_lc16[cid, :, :], rs_ub)
                for i in T.serial(C):
                    dgam_ub[i] = dgam_ub[i] - rs_ub[i, 0]
                # ---- B10: dK += theta .* dKt ; dth = rowsum(K .* dKt)
                #      (K/dKt half reloads, S7e :926-959 order) ----
                for hh in T.serial(bk_num):
                    T.copy(WS_DKT[b, h, cc * C, hh * BK], xk_ub)  # dKt half (fp32 GM)
                    # dth half = rowsum(K .* dKt) (k2 lane)
                    T.tile.fill(k2_ub, 0.0)
                    T.copy(K[b, h, cc * C, hh * BK], io_xk)
                    T.copy(io_xk, k2_ub)
                    for (i, j) in T.Parallel(C, BK):
                        k2_ub[i, j] = k2_ub[i, j] * xk_ub[i, j]  # K .* dKt
                    T.copy(k2_ub, io_xk)
                    T.copy(io_xk, workspace_vdk[cid, :, :])
                    T.copy(workspace_vdk[cid, :, :], xk_l1)
                    T.gemm_v0(xk_l1, onesdk_l1, rs2_l0, init=True)
                    T.copy(rs2_l0, workspace_lc16[cid, :, :])
                    T.copy(workspace_lc16[cid, :, :], rs_ub)
                    if hh == 0:
                        for i in T.serial(C):
                            tmp2_ub[i] = rs_ub[i, 0]  # dth
                    else:
                        for i in T.serial(C):
                            tmp2_ub[i] = tmp2_ub[i] + rs_ub[i, 0]
                    # dK += theta .* dKt (xk lane still holds dKt)
                    for i in T.serial(C):  # S7g: serial (no broadcast temp)
                        for j in T.serial(C):
                            xk_ub[i, j] = xk_ub[i, j] * th_ub[i]
                    if hh == 0:
                        for (i, j) in T.Parallel(C, BK):
                            dk0_ub[i, j] = dk0_ub[i, j] + xk_ub[i, j]
                    else:
                        for (i, j) in T.Parallel(C, BK):
                            dk1_ub[i, j] = dk1_ub[i, j] + xk_ub[i, j]
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
                for hh in T.serial(bk_num):
                    if hh == 0:
                        T.copy(dk0_ub, io_xk)
                    else:
                        T.copy(dk1_ub, io_xk)
                    T.copy(io_xk, DKG[b, h, cc * C, hh * BK])
                # S7e: DBeta/DG are float GM and tmp1/tmp2 are fp32 ub --
                # same-dtype direct copy (ub->gm must be dtype-matched)
                T.copy(tmp1_ub, DBeta[b, h, cc * C])
                T.copy(tmp2_ub, DG[b, h, cc * C])

    return main


def gdn_bwd_v2(q, k, v, g, beta, h0, do, chunk_size=64, NT_MAX=None):
    """Host wrapper (proto.yaml schema). Device follows `q`.

    Five-kernel pipeline: cumsum (g -> WS_G) -> solve_a (A = (I+L)^-1 ->
    WS_A) -> scan1 (replay -> WS_H) -> scan2a (reverse B1-B6/B3/B4/B9/B15 ->
    DQ, DH0 + seven boundary boxes) -> scan2b (B7/B8/B10-B15 -> DKG/DVG/
    DBeta/DG). Cross-kernel state (WS_G / WS_A / WS_H / WS_D + the S7f-E
    boundary boxes) travels in GM; the workspace_* CV-handoff buffers are
    per-kernel workspaces (workspace_idx, auto-allocated by the runtime).
    Returns the proto 6-tuple (dq, dk, dv, dbeta, dg, dh0) unchanged.
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
    outs_a = gdn_bwd_v2_scan2a(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        q, k, v, do, beta, ws_g, ws_a, msk_m, ones_dv, ws_h)
    # outs_a = (dq, dh0, ws_dv, ws_dw, ws_dkt, ws_de, ws_dgam2, ws_dgam, ws_dk)
    outs_b = gdn_bwd_v2_scan2b(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        k, v, beta, ws_g, ws_a, msk_j, ones_dv, ones_dk, ones_c,
        outs_a[2], outs_a[3], outs_a[4], outs_a[5], outs_a[6], outs_a[7],
        outs_a[8])
    # outs_b = (dkg, dvg, dbeta, dg)
    return outs_a[0], outs_b[0], outs_b[1], outs_b[2], outs_b[3], outs_a[1]

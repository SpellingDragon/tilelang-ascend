# w11 scan1 stage-cut probe (process file; docker cp to container /tmp for the
# parse check, then ship to the card for the on-card bisect).
#
# cut=1: S1 only  -- H0 -> WS_H[.., 0]
# cut=2: S1+S2     -- per-chunk gam/beta/E/glast/th/bee
# cut=3: +S3       -- Kt/bek row-scales + W GEMM + workspace round-trips
# cut=4: +S4       -- per-BV U0/WH/Vnew/H_out chain + WS_H[c+1] write
#
# The cut gate is a Python bool (factory arg) inside the prim_func body; if the
# tvm-script parser rejects python-level `if` here, fall back to verbatim
# pruned copies (window-9 ladder style). Truncation changes the W-R shape, so
# the CombineCV pass re-derives sync for each cut -- a cut may fail to *compile*
# for sync reasons; that is still a locating datum, note it and move on.
import sys

sys.path.insert(0, "/tmp")
import torch  # noqa: E402

try:
    import torch_npu  # noqa: E402,F401
except ImportError:  # device-less container: shim npu so the factory can compile
    _orig_device = torch.device

    def _device(s=None, *a, **k):
        if isinstance(s, str) and s.startswith("npu"):
            return _orig_device("cpu")
        return _orig_device(s, *a, **k)

    torch.device = _device

    class _NpuShim:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def current_stream():
            class _S:
                npu_stream = 0

            return _S()

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def set_device(*a, **k):
            pass

    torch.npu = _NpuShim

import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402

from gdn_bwd_v2 import pass_configs  # noqa: E402  (production set incl. mask-reuse OFF)


@tilelang.jit(out_idx=[6], workspace_idx=[7, 8, 9, 10, 11, 12],
              pass_configs=pass_configs)
def scan1_cut(B, H, L, DK, DV, C, BV, BK, cut, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    bv_num = T.ceildiv(DV, BV)
    n_tail = L - (chunk_num - 1) * C

    @T.prim_func
    def main(
        K: T.Tensor([B, H, L, DK], dtype),  # type: ignore
        V: T.Tensor([B, H, L, DV], dtype),  # type: ignore
        Beta: T.Tensor([B, H, L], "float"),  # type: ignore
        S: T.Tensor([B, H, L], "float"),  # type: ignore
        A: T.Tensor([B, H, L, C], dtype),  # type: ignore
        H0: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
        WS_H: T.Tensor([B, H, chunk_num + 1, DV, DK], dtype),  # type: ignore
        workspace_vdk: T.Tensor([B * H, C, DK], dtype),  # type: ignore
        workspace_ldk: T.Tensor([B * H, C, DK], accum_dtype),  # type: ignore
        workspace_lcb: T.Tensor([B * H, C, BV], accum_dtype),  # type: ignore
        workspace_vcb: T.Tensor([B * H, C, BV], dtype),  # type: ignore
        workspace_lbd: T.Tensor([B * H, BV, DK], accum_dtype),  # type: ignore
        workspace_vbd: T.Tensor([B * H, BV, DK], dtype),  # type: ignore
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H

            a_l1 = T.alloc_L1([C, C], dtype)
            bek_l1 = T.alloc_L1([C, DK], dtype)
            kt_l1 = T.alloc_L1([C, DK], dtype)
            w_l1 = T.alloc_L1([C, DK], dtype)
            hin_l1 = T.alloc_L1([BV, DK], dtype)
            bv_l1 = T.alloc_L1([C, BV], dtype)
            vnew_l1 = T.alloc_L1([C, BV], dtype)
            dw1_l0 = T.alloc_L0C([C, DK], accum_dtype)
            u01_l0 = T.alloc_L0C([C, BV], accum_dtype)
            wh1_l0 = T.alloc_L0C([C, BV], accum_dtype)
            hout_l0 = T.alloc_L0C([BV, DK], accum_dtype)

            g_ub = T.alloc_ub([C], accum_dtype)
            e_ub = T.alloc_ub([C], accum_dtype)
            th_ub = T.alloc_ub([C], accum_dtype)
            zero_ub = T.alloc_ub([C], accum_dtype)
            be_ub = T.alloc_ub([C], accum_dtype)
            bee_ub = T.alloc_ub([C], accum_dtype)
            glast_ub = T.alloc_ub([8], accum_dtype)
            xk1_ub = T.alloc_ub([C, DK], accum_dtype)
            bq1_ub = T.alloc_ub([C, BV], accum_dtype)
            wh1_ub = T.alloc_ub([C, BV], accum_dtype)
            hin_ub = T.alloc_ub([BV, DK], accum_dtype)
            dh1_ub = T.alloc_ub([BV, DK], accum_dtype)
            io_xk1 = T.alloc_ub([C, DK], dtype)
            io_bq1 = T.alloc_ub([C, BV], dtype)
            io_hv1 = T.alloc_ub([BV, DK], dtype)

            # ---- S1: H0 -> WS_H[.., 0] ----
            for bv in T.serial(bv_num):
                T.tile.fill(hin_ub, 0.0)
                T.copy(H0[b, h, bv * BV, 0], io_hv1)
                T.copy(io_hv1, hin_ub)
                T.copy(hin_ub, io_hv1)
                T.copy(io_hv1, WS_H[b, h, 0, bv * BV, 0])

            for c in T.serial(chunk_num):
                # ---- S2: gam / beta / E / glast / th / bee ----
                T.tile.fill(g_ub, 0.0)
                T.tile.fill(be_ub, 0.0)
                T.tile.fill(zero_ub, 0.0)
                T.copy(S[b, h, c * C], g_ub)
                T.copy(Beta[b, h, c * C], be_ub)
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                if c == chunk_num - 1:
                    T.copy(g_ub[n_tail - 1:n_tail], glast_ub[0:1])
                else:
                    T.copy(g_ub[C - 1:C], glast_ub[0:1])
                T.tile.exp(glast_ub, glast_ub)
                for i in T.Parallel(C):
                    th_ub[i] = g_ub[i] - g_ub[C - 1]
                if c == chunk_num - 1:
                    for i in T.Parallel(C):
                        th_ub[i] = g_ub[i] - g_ub[n_tail - 1]
                for i in T.Parallel(C):
                    th_ub[i] = zero_ub[i] - th_ub[i]
                for i in T.Parallel(C):
                    th_ub[i] = T.exp(th_ub[i])
                for i in T.Parallel(C):
                    bee_ub[i] = be_ub[i] * e_ub[i]

                if cut == 2:
                    # DCE anchor: with no S3/S4, S2's chains have no observable
                    # consumer and the whole stage is dead-code-eliminated
                    # (CUT1 src == CUT2 src == 7532B on the first probe run).
                    # Park the three chains on GM so cut=2 really executes S2 --
                    # including the glast chain (g_ub[63] -> glast_ub narrow copy
                    # and the 8-lane exp).
                    T.copy(g_ub, S[b, h, c * C])
                    T.copy(th_ub, S[b, h, c * C])
                    T.copy(glast_ub[0:1], g_ub[0:1])

                if cut >= 3:
                    # ---- S3: Kt / bek row-scales + W GEMM + ws round-trips ----
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
                    T.copy(A[b, h, c * C, 0], a_l1)
                    T.gemm_v0(a_l1, bek_l1, dw1_l0, init=True)
                    T.copy(dw1_l0, workspace_ldk[cid, :, :])
                    T.copy(workspace_ldk[cid, :, :], xk1_ub)
                    T.copy(xk1_ub, io_xk1)
                    T.copy(io_xk1, workspace_vdk[cid, :, :])
                    T.copy(workspace_vdk[cid, :, :], w_l1)

                if cut >= 4:
                    # ---- S4: per-BV U0 / WH / Vnew / H_out + WS_H write ----
                    for bv in T.serial(bv_num):
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


# ---------------- on-card driver ----------------
if __name__ == "__main__":
    from gdn_bwd_v2 import gdn_bwd_v2_cumsum, gdn_bwd_v2_solve_a

    B, H, T_len, DK, DV = 1, 1, 64, 64, 64
    C, BV, BK = 64, 64, 64
    dt = torch.float16
    torch.manual_seed(11)
    k = torch.randn(B, H, T_len, DK)
    k = (k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(dt).npu()
    v = torch.randn(B, H, T_len, DV).to(dt).npu()
    g = (-(torch.randn(B, H, T_len).abs() * 0.5 + 0.05)).npu()
    beta = torch.sigmoid(torch.randn(B, H, T_len)).npu()
    h0 = (torch.randn(B, H, DV, DK) * 0.1).to(dt).npu()
    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device="npu"), -1)
    idt = torch.eye(C, dtype=torch.float32, device="npu")

    ws_g = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, BK, dtype="float16")(
        k, beta, ws_g, msk_j, idt)
    torch.npu.synchronize()

    for cut in (1, 2, 3, 4):
        try:
            ker = scan1_cut(B, H, T_len, DK, DV, C, BV, BK, cut, dtype="float16")
            out = ker(k, v, beta, ws_g, ws_a, h0)
            torch.npu.synchronize()
            print(f"CUT{cut}-PASS")
        except Exception as e:  # noqa: BLE001
            print(f"CUT{cut}-FAIL {type(e).__name__}: {str(e)[:110]}")
    print("CUT-DONE")

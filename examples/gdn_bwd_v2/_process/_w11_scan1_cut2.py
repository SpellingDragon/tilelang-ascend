# w11 scan1 stage-cut probe v2 (process file; docker cp to container /tmp for the
# parse check, then ship to the card).
#
# v1 (w11cut) verdict recap: CUT1 (= S1+S2 verbatim, 0 unique lines vs the
# production scan1 codegen) FAILs 507015 on-card; S1 forms all have PASS twins
# in solve_a (count-mode fill / cross-dtype cast / same-dtype DataCopy), so the
# remaining shape difference is the S2 glast chain:
#   - narrow copy DataCopy(glast_ub[0], g_ub[63], 1) with src offset 252B
#     (NOT 32B-aligned -- window-9 alignment family shape)
#   - 8-lane exp on glast_ub
#
# This probe isolates them with ONE kernel per variant, one process per variant
# (a first-kernel aicore exception poisons the stream; per-process legs give
# independent verdicts):
#   variant=1: S1+S2 verbatim (v1 CUT1 baseline replay)
#   variant=2: S1+S2, glast narrow copy -> scalar (`glast_ub[0] = g_ub[C-1]`)
#   variant=3: S1+S2, glast narrow copy -> aligned 8-lane (g_ub[C-8:C] @224B)
#   variant=4: S1 only (S2 excluded wholesale)
#
# driver: python3 _w11_scan1_cut2.py <variant>; the bash loop runs 1,4,2,3.
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
def scan1_cut2(B, H, L, DK, DV, C, BV, BK, variant, dtype="float16", accum_dtype="float"):
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

            g_ub = T.alloc_ub([C], accum_dtype)
            e_ub = T.alloc_ub([C], accum_dtype)
            th_ub = T.alloc_ub([C], accum_dtype)
            zero_ub = T.alloc_ub([C], accum_dtype)
            be_ub = T.alloc_ub([C], accum_dtype)
            bee_ub = T.alloc_ub([C], accum_dtype)
            glast_ub = T.alloc_ub([8], accum_dtype)
            hin_ub = T.alloc_ub([BV, DK], accum_dtype)
            io_hv1 = T.alloc_ub([BV, DK], dtype)

            # ---- S1: H0 -> WS_H[.., 0] ----
            for bv in T.serial(bv_num):
                T.tile.fill(hin_ub, 0.0)
                T.copy(H0[b, h, bv * BV, 0], io_hv1)
                T.copy(io_hv1, hin_ub)
                T.copy(hin_ub, io_hv1)
                T.copy(io_hv1, WS_H[b, h, 0, bv * BV, 0])

            if variant <= 3:
                for c in T.serial(chunk_num):
                    # ---- S2: gam / beta / E / glast / th / bee ----
                    T.tile.fill(g_ub, 0.0)
                    T.tile.fill(be_ub, 0.0)
                    T.tile.fill(zero_ub, 0.0)
                    T.copy(S[b, h, c * C], g_ub)
                    T.copy(Beta[b, h, c * C], be_ub)
                    for i in T.Parallel(C):
                        e_ub[i] = T.exp(g_ub[i])
                    if variant >= 3:
                        T.copy(g_ub[C - 8:C], glast_ub[0:8])
                    else:
                        if variant == 2:
                            glast_ub[0] = g_ub[C - 1]
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
                    # GM anchors: park the chains so S2 survives DCE even when
                    # downstream S3/S4 are absent.
                    T.copy(g_ub, S[b, h, c * C])
                    T.copy(th_ub, S[b, h, c * C])
                    T.copy(glast_ub[0:1], g_ub[0:1])

    return main


# ---------------- on-card driver (one variant per process) ----------------
if __name__ == "__main__":
    from gdn_bwd_v2 import gdn_bwd_v2_cumsum, gdn_bwd_v2_solve_a

    variant = int(sys.argv[1])
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

    try:
        ws_g = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
        ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, BK, dtype="float16")(
            k, beta, ws_g, msk_j, idt)
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"PRE-V{variant}-FAIL {type(e).__name__}: {str(e)[:150]}")
        raise SystemExit(3)

    try:
        ker = scan1_cut2(B, H, T_len, DK, DV, C, BV, BK, variant, dtype="float16")
        out = ker(k, v, beta, ws_g, ws_a, h0)
        torch.npu.synchronize()
        print(f"CUT2-V{variant}-PASS")
    except Exception as e:  # noqa: BLE001
        print(f"CUT2-V{variant}-FAIL {type(e).__name__}: {str(e)[:150]}")
    print("CUT2-DONE")

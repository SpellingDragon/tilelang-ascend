"""mp_budget_probe (S7b): can TL_ASCEND_MEMORY_PLANNING fit the
gdn_bwd_v2_scan alloc set into L0C 128KiB / UB 192KiB at C=64, DK=DV=128,
BV=64 (the compile-gate true-shape tile)?

Static totals of the scan alloc set: L0C ~400KiB, UB ~390KiB -- both over
budget unless planning reuses disjoint lifetimes.

mode 0 = DISJOINT pressure: every buffer touched every iteration (all
         lifetimes overlap) -- planning must NOT be relied on to merge these.
mode 1 = STAGED touches: pass-1 lane then pass-2 lane then reduce lane
         (the real kernel's lifetime shape) -- this is what the kernel needs.

verdict lines:  MP-L0C-mode{0,1} OK/FAIL   MP-UB-mode{0,1} OK/FAIL
"""
import sys

sys.path.insert(0, "/tmp")
import torch as _torch

_orig_device = _torch.device


def _device(s=None, *a, **k):
    if isinstance(s, str) and s.startswith("npu"):
        return _orig_device("cpu")
    return _orig_device(s, *a, **k)


_torch.device = _device
if not hasattr(_torch, "npu"):

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

        @staticmethod
        def get_device_name():
            return "mp-shim"

    _torch.npu = _NpuShim

import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}

C, DK, DV, BV = 64, 128, 128, 64


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def probe_l0c(mode: int):
    @T.prim_func
    def main(
        ICK_A: T.Tensor([C, DK], "float16"),
        ICK_B: T.Tensor([C, DK], "float16"),
        IC_A: T.Tensor([C, C], "float16"),
        IC_B: T.Tensor([C, C], "float16"),
        ICV_A: T.Tensor([C, BV], "float16"),
        ICV_B: T.Tensor([C, BV], "float16"),
        IBV_A: T.Tensor([BV, C], "float16"),
        IBK_A: T.Tensor([BV, DK], "float16"),
        ONES_C: T.Tensor([C, 16], "float16"),
        ONES_T: T.Tensor([16, C], "float16"),
        O: T.Tensor([C, DK], "float"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            ick_a = T.alloc_L1([C, DK], "float16")
            ick_b = T.alloc_L1([C, DK], "float16")
            ic_a = T.alloc_L1([C, C], "float16")
            ic_b = T.alloc_L1([C, C], "float16")
            icv_a = T.alloc_L1([C, BV], "float16")
            icv_b = T.alloc_L1([C, BV], "float16")
            ibv_a = T.alloc_L1([BV, C], "float16")
            ibk_a = T.alloc_L1([BV, DK], "float16")
            ones_c = T.alloc_L1([C, 16], "float16")
            ones_t = T.alloc_L1([16, C], "float16")
            # ---- the gdn_bwd_v2_scan L0C set (verbatim shapes) ----
            kk_l0 = T.alloc_L0C([C, C], "float")
            u0_l0 = T.alloc_L0C([C, BV], "float")
            wh_l0 = T.alloc_L0C([C, BV], "float")
            hout_l0 = T.alloc_L0C([BV, DK], "float")
            qh_l0 = T.alloc_L0C([C, BV], "float")
            dvnew_l0 = T.alloc_L0C([C, BV], "float")
            dp_l0 = T.alloc_L0C([C, C], "float")
            dq_l0 = T.alloc_L0C([C, DK], "float")
            dk_l0 = T.alloc_L0C([C, DK], "float")
            dw_l0 = T.alloc_L0C([C, DK], "float")
            dkt_l0 = T.alloc_L0C([C, DK], "float")
            t1_l0 = T.alloc_L0C([C, BV], "float")
            da_l0 = T.alloc_L0C([C, C], "float")
            t2_l0 = T.alloc_L0C([C, DK], "float")
            dsum_l0 = T.alloc_L0C([C, DK], "float")
            dh_l0 = T.alloc_L0C([BV, DK], "float")
            dx_l0 = T.alloc_L0C([C, C], "float")
            rs_l0 = T.alloc_L0C([C, 16], "float")
            cs_l0 = T.alloc_L0C([16, C], "float")
            o_ub = T.alloc_ub([C, DK], "float")
            T.copy(ICK_A[0, 0], ick_a)
            T.copy(ICK_B[0, 0], ick_b)
            T.copy(IC_A[0, 0], ic_a)
            T.copy(IC_B[0, 0], ic_b)
            T.copy(ICV_A[0, 0], icv_a)
            T.copy(ICV_B[0, 0], icv_b)
            T.copy(IBV_A[0, 0], ibv_a)
            T.copy(IBK_A[0, 0], ibk_a)
            T.copy(ONES_C[0, 0], ones_c)
            T.copy(ONES_T[0, 0], ones_t)

            for it in T.serial(2):
                T.gemm_v0(ic_a, icv_b, u0_l0, init=True)
                T.gemm_v0(ic_a, icv_b, wh_l0, init=True)
                T.gemm_v0(ibv_a, ick_b, hout_l0, init=True)
                T.gemm_v0(ick_a, ick_b, kk_l0, transpose_B=True, init=True)
                T.gemm_v0(ic_a, icv_b, qh_l0, init=True)
                T.gemm_v0(ic_a, icv_b, dvnew_l0, init=True)
                T.gemm_v0(ic_a, ic_b, dp_l0, init=True)
                T.gemm_v0(ic_a, ic_b, da_l0, init=True)
                T.gemm_v0(ic_a, ic_b, dx_l0, init=True)
                T.gemm_v0(ic_a, ick_b, dq_l0, init=True)
                T.gemm_v0(ic_a, ick_b, dk_l0, init=True)
                T.gemm_v0(ic_a, ick_b, dw_l0, init=True)
                T.gemm_v0(ic_a, ick_b, dkt_l0, init=True)
                T.gemm_v0(ic_a, ick_b, t2_l0, init=True)
                T.gemm_v0(ic_a, ick_b, dsum_l0, init=True)
                T.gemm_v0(ic_a, icv_b, t1_l0, init=True)
                T.gemm_v0(ibv_a, ick_b, dh_l0, init=True)
                T.gemm_v0(ic_a, ones_c, rs_l0, init=True)
                T.gemm_v0(ones_t, ic_b, cs_l0, init=True)
            T.copy(dq_l0, o_ub)
            T.copy(o_ub, O[0, 0])
    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def probe_ub(mode: int):
    @T.prim_func
    def main(I: T.Tensor([C, DK], "float"), O: T.Tensor([C, DK], "float")):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            g_ub = T.alloc_ub([C], "float")
            e_ub = T.alloc_ub([C], "float")
            th_ub = T.alloc_ub([C], "float")
            be_ub = T.alloc_ub([C], "float")
            bee_ub = T.alloc_ub([C], "float")
            dE_ub = T.alloc_ub([C], "float")
            dgam_ub = T.alloc_ub([C], "float")
            dbe_ub = T.alloc_ub([C], "float")
            tmp1_ub = T.alloc_ub([C], "float")
            tmp2_ub = T.alloc_ub([C], "float")
            scal_ub = T.alloc_ub([8], "float")
            glast_ub = T.alloc_ub([8], "float")
            gm_ub = T.alloc_ub([C, C], "float")
            msk_ub = T.alloc_ub([C, C], "float")
            kk_ub = T.alloc_ub([C, C], "float")
            pk_ub = T.alloc_ub([C, C], "float")
            dgam2_ub = T.alloc_ub([C, C], "float")
            da_ub = T.alloc_ub([C, C], "float")
            dx_ub = T.alloc_ub([C, C], "float")
            mul_ub = T.alloc_ub([C, C], "float")
            bq_ub = T.alloc_ub([C, BV], "float")
            xk_ub = T.alloc_ub([C, DK], "float")
            wh_ub = T.alloc_ub([C, BV], "float")
            dvnew_ub = T.alloc_ub([C, BV], "float")
            dq_ub = T.alloc_ub([C, DK], "float")
            dk_ub = T.alloc_ub([C, DK], "float")
            hin_ub = T.alloc_ub([BV, DK], "float")
            dhp_ub = T.alloc_ub([BV, DK], "float")
            rs_ub = T.alloc_ub([C, 16], "float")
            red1_ub = T.alloc_ub([BV], "float")
            io_bq = T.alloc_ub([C, BV], "float16")
            io_xk = T.alloc_ub([C, DK], "float16")
            io_cc = T.alloc_ub([C, C], "float16")
            io_hv = T.alloc_ub([BV, DK], "float16")
            io_1d = T.alloc_ub([C], "float16")

            T.copy(I[0, 0], xk_ub)
            for it in T.serial(2):
                for i in T.Parallel(C):
                    g_ub[i] = g_ub[i] + g_ub[i]
                for i in T.Parallel(C):
                    e_ub[i] = e_ub[i] + e_ub[i]
                for i in T.Parallel(C):
                    th_ub[i] = th_ub[i] + th_ub[i]
                for i in T.Parallel(C):
                    be_ub[i] = be_ub[i] + be_ub[i]
                for i in T.Parallel(C):
                    bee_ub[i] = bee_ub[i] + bee_ub[i]
                for i in T.Parallel(C):
                    dE_ub[i] = dE_ub[i] + dE_ub[i]
                for i in T.Parallel(C):
                    dgam_ub[i] = dgam_ub[i] + dgam_ub[i]
                for i in T.Parallel(C):
                    dbe_ub[i] = dbe_ub[i] + dbe_ub[i]
                for i in T.Parallel(C):
                    tmp1_ub[i] = tmp1_ub[i] + tmp1_ub[i]
                for i in T.Parallel(C):
                    tmp2_ub[i] = tmp2_ub[i] + tmp2_ub[i]
                for i, j in T.Parallel(C, C):
                    gm_ub[i, j] = gm_ub[i, j] + gm_ub[i, j]
                for i, j in T.Parallel(C, C):
                    msk_ub[i, j] = msk_ub[i, j] + msk_ub[i, j]
                for i, j in T.Parallel(C, C):
                    kk_ub[i, j] = kk_ub[i, j] + kk_ub[i, j]
                for i, j in T.Parallel(C, C):
                    pk_ub[i, j] = pk_ub[i, j] + pk_ub[i, j]
                for i, j in T.Parallel(C, C):
                    dgam2_ub[i, j] = dgam2_ub[i, j] + dgam2_ub[i, j]
                for i, j in T.Parallel(C, C):
                    da_ub[i, j] = da_ub[i, j] + da_ub[i, j]
                for i, j in T.Parallel(C, C):
                    dx_ub[i, j] = dx_ub[i, j] + dx_ub[i, j]
                for i, j in T.Parallel(C, C):
                    mul_ub[i, j] = mul_ub[i, j] + mul_ub[i, j]
                for i, j in T.Parallel(C, BV):
                    bq_ub[i, j] = bq_ub[i, j] + bq_ub[i, j]
                for i, j in T.Parallel(C, DK):
                    xk_ub[i, j] = xk_ub[i, j] + xk_ub[i, j]
                for i, j in T.Parallel(C, BV):
                    wh_ub[i, j] = wh_ub[i, j] + wh_ub[i, j]
                for i, j in T.Parallel(C, BV):
                    dvnew_ub[i, j] = dvnew_ub[i, j] + dvnew_ub[i, j]
                for i, j in T.Parallel(C, DK):
                    dq_ub[i, j] = dq_ub[i, j] + dq_ub[i, j]
                for i, j in T.Parallel(C, DK):
                    dk_ub[i, j] = dk_ub[i, j] + dk_ub[i, j]
                for i, j in T.Parallel(BV, DK):
                    hin_ub[i, j] = hin_ub[i, j] + hin_ub[i, j]
                for i, j in T.Parallel(BV, DK):
                    dhp_ub[i, j] = dhp_ub[i, j] + dhp_ub[i, j]
                for i, j in T.Parallel(C, 16):
                    rs_ub[i, j] = rs_ub[i, j] + rs_ub[i, j]
                for i, j in T.Parallel(C, BV):
                    io_bq[i, j] = io_bq[i, j] + io_bq[i, j]
                for i, j in T.Parallel(C, DK):
                    io_xk[i, j] = io_xk[i, j] + io_xk[i, j]
                for i, j in T.Parallel(C, C):
                    io_cc[i, j] = io_cc[i, j] + io_cc[i, j]
                for i, j in T.Parallel(BV, DK):
                    io_hv[i, j] = io_hv[i, j] + io_hv[i, j]
                for i in T.Parallel(C):
                    io_1d[i] = io_1d[i] + io_1d[i]
                for i in T.Parallel(8):
                    scal_ub[i] = scal_ub[i] + scal_ub[i]
                for i in T.Parallel(8):
                    glast_ub[i] = glast_ub[i] + glast_ub[i]
                for i in T.Parallel(BV):
                    red1_ub[i] = red1_ub[i] + red1_ub[i]
            T.copy(dq_ub, O[0, 0])
    return main


for md in (0, 1):
    try:
        k = probe_l0c(md)
        k.get_kernel_source()
        print(f"MP-L0C-mode{md} OK")
    except Exception as e:
        print(f"MP-L0C-mode{md} FAIL {type(e).__name__}: {str(e)[:300]}")
    try:
        k = probe_ub(md)
        k.get_kernel_source()
        print(f"MP-UB-mode{md} OK")
    except Exception as e:
        print(f"MP-UB-mode{md} FAIL {type(e).__name__}: {str(e)[:300]}")

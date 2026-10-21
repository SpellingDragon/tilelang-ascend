# _w11_cube_probe.py — A 路第二块砖：纯 cube kernel（L1 + L0C，无 UB 运算）
#
# 拆 kernel 路线：vector 段与 cube 段各自成为独立 kernel，每个 launch 单侧，
# cv-sync 的配对天然成立（notes [83]）。本探针验证 cube 侧：
#   U0 = A @ BV,  W = A @ BEK   （BV = beta.*V, BEK = beta.*E.*K，由 vec 侧/宿主给）
# 判读：CUBEP PASS ⇒ cube 段可以独立成 kernel，且 L0C→GM 直拷可用。
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

torch.manual_seed(7)

_pass = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[3, 4], pass_configs=_pass)
def cube_uw(B, H, C, DV, DK, dtype="float16", accum="float"):
    @T.prim_func
    def main(
        A: T.Tensor([B, H, C, C], dtype),      # type: ignore
        BV: T.Tensor([B, H, C, DV], dtype),    # type: ignore  beta .* V
        BEK: T.Tensor([B, H, C, DK], dtype),   # type: ignore  beta .* E .* K
        U0: T.Tensor([B, H, C, DV], dtype),    # type: ignore
        W: T.Tensor([B, H, C, DK], dtype),     # type: ignore
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)
            bv_l1 = T.alloc_L1([C, DV], dtype)
            bek_l1 = T.alloc_L1([C, DK], dtype)
            u0_l0 = T.alloc_L0C([C, DV], accum)
            w_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(BV[b, h, 0, 0], bv_l1)
            T.copy(BEK[b, h, 0, 0], bek_l1)
            T.gemm_v0(a_l1, bv_l1, u0_l0, init=True)
            T.gemm_v0(a_l1, bek_l1, w_l0, init=True)
            T.copy(u0_l0, U0[b, h, 0, 0])   # L0C -> GM direct
            T.copy(w_l0, W[b, h, 0, 0])

    return main


def main():
    B, H, C, DV, DK = 1, 2, 64, 64, 64
    a = torch.randn(B, H, C, C, dtype=torch.float16) * 0.1
    bv = torch.randn(B, H, C, DV, dtype=torch.float16)
    bek = torch.randn(B, H, C, DK, dtype=torch.float16)
    u0, w = cube_uw(B, H, C, DV, DK)(a.npu(), bv.npu(), bek.npu())
    torch.npu.synchronize()
    g_u0 = u0.float().cpu().numpy()
    g_w = w.float().cpu().numpy()
    r_u0 = a.double().numpy() @ bv.double().numpy()
    r_w = a.double().numpy() @ bek.double().numpy()
    rel_u0 = float(np.abs(g_u0 - r_u0).max() / (np.abs(r_u0).max() + 1e-30))
    rel_w = float(np.abs(g_w - r_w).max() / (np.abs(r_w).max() + 1e-30))
    print(f"CUBEP rel_u0={rel_u0:.3e} rel_w={rel_w:.3e} "
          f"finite={int(np.isfinite(g_u0).sum())}/{g_u0.size},{int(np.isfinite(g_w).sum())}/{g_w.size}")
    print("CUBEP verdict=%s" % ("PASS" if max(rel_u0, rel_w) < 5e-3 else "FAIL"))
    print("CUBEP-DONE")


if __name__ == "__main__":
    main()

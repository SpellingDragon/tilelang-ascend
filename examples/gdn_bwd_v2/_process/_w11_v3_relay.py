# _w11_v3_relay.py — P4 的四个 relay kernel（全为已验证的 tA/tB 形态）
#
#   dvnew = P^T dO + Kt (Delta')^T      # B1
#   dw    = -dvnew H_in                  # B6（-1 在 vector 段乘）
#   Delta'= e_do^T Q - dvnew^T W         # B5+B6
#   dh0   = Delta'_0                     # 反向扫描的终点就是初值梯度
#
# 每个 kernel 一个 GEMM；两项相加一律在 host。单 chunk 时 Delta'=0，
# 本脚本只做"形状与数值"自检（与 numpy 的同一公式比），不依赖 golden。
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


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_ta(B, H, C, M, N, dtype="float16", accum="float"):
    """O = A^T B,  A:[C,M], B:[C,N] -> [M,N]"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, M], dtype), Bm: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, M, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, M], dtype)
            b_l1 = T.alloc_L1([C, N], dtype)
            o_l0 = T.alloc_L0C([M, N], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(Bm[b, h, 0, 0], b_l1)
            T.gemm_v0(a_l1, b_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_tb(B, H, C, M, N, dtype="float16", accum="float"):
    """O = A B^T,  A:[C,N], B:[M,N] -> [C,M]"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, N], dtype), Bm: T.Tensor([B, H, M, N], dtype),  # type: ignore
             O: T.Tensor([B, H, C, M], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, N], dtype)
            b_l1 = T.alloc_L1([M, N], dtype)
            o_l0 = T.alloc_L0C([C, M], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(Bm[b, h, 0, 0], b_l1)
            T.gemm_v0(a_l1, b_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


def main():
    B, H, C, DV, DK = 1, 2, 64, 64, 64
    def f(t): return torch.from_numpy(t.astype(np.float16)).to("npu")

    P = np.random.randn(B, H, C, C).astype(np.float16) * 0.05
    DO = np.random.randn(B, H, C, DV).astype(np.float16)
    Kt = np.random.randn(B, H, C, DK).astype(np.float16)
    DL = np.random.randn(B, H, DV, DK).astype(np.float16)   # Delta' [DV,DK]
    W = np.random.randn(B, H, C, DK).astype(np.float16)
    EDO = np.random.randn(B, H, C, DV).astype(np.float16)

    kta, ktb = k_ta(B, H, C, C, DV), k_tb(B, H, C, DV, DK)
    ktbk = k_tb(B, H, C, C, DK)

    # dvnew = P^T dO + Kt Delta'^T
    t_a = kta(f(P), f(DO)).float().cpu().numpy().astype(np.float64)
    t_b = ktbk(f(Kt), f(DL)).float().cpu().numpy().astype(np.float64)
    dvnew = t_a + t_b
    ref_a = P.astype(np.float64).transpose(0, 1, 3, 2) @ DO.astype(np.float64)
    ref_b = Kt.astype(np.float64) @ DL.astype(np.float64).transpose(0, 1, 3, 2)
    rel_a = float(np.abs(t_a - ref_a).max() / np.abs(ref_a).max())
    rel_b = float(np.abs(t_b - ref_b).max() / np.abs(ref_b).max())

    # Delta' = e_do^T Q - dvnew^T W  (Q 用 Kt 占位以校验两个 tA 形态)
    q_like = Kt
    d1 = kta(f(EDO), f(q_like)).float().cpu().numpy().astype(np.float64)
    d2 = kta(f(dvnew.astype(np.float16)), f(W)).float().cpu().numpy().astype(np.float64)
    rel_d1 = float(np.abs(d1 - EDO.astype(np.float64).transpose(0, 1, 3, 2)
                          @ q_like.astype(np.float64)).max() / np.abs(d1).max())
    rel_d2 = float(np.abs(d2 - dvnew.transpose(0, 1, 3, 2) @ W.astype(np.float64)).max()
                   / np.abs(d2).max())
    print(f"RELAY rel_pTdo={rel_a:.3e} rel_ktdl={rel_b:.3e} rel_edoTq={rel_d1:.3e} rel_dvnTW={rel_d2:.3e}")
    ok = max(rel_a, rel_b, rel_d1, rel_d2) < 5e-3
    print("RELAY verdict=%s" % ("PASS" if ok else "FAIL"))
    print("RELAY-DONE")


if __name__ == "__main__":
    main()

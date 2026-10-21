# _w11_split_gemm.py — 验"分裂精度 GEMM"：A = A_hi + A_lo 两次累加
#
# 动机：dq/dk/dbeta/dg 都比本地 fp16 模型差 15-80×，主因是 `ws_a`（solve_a 的求逆结果）
# 以 fp16 落 GM，信息已丢。若把 A 拆成 hi/lo 两片、两次 GEMM 累加到同一 L0C，
# 有效精度可从 ~11 位提到 ~22 位。
#
# 判据：
#   single（A_hi 单次）   rel 应 ~1e-4（fp16 的正常水平）
#   split （hi + lo 累加）rel 应显著更小（~1e-7 量级）
# 两个 GEMM 都是 no-transpose 同族 ⇒ 累加已被 dq5 的 K5 验证过。
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


@tilelang.jit(out_idx=[3], pass_configs=_pass)
def k_ab2(B, H, C, N, dtype="float16", accum="float"):
    """O = (A_hi + A_lo) @ X，两个 no-transpose GEMM 累加到同一 L0C。"""

    @T.prim_func
    def main(AH: T.Tensor([B, H, C, C], dtype), AL: T.Tensor([B, H, C, C], dtype),  # type: ignore
             X: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            ah_l1 = T.alloc_L1([C, C], dtype)   # A only
            al_l1 = T.alloc_L1([C, C], dtype)   # A only
            x_l1 = T.alloc_L1([C, N], dtype)    # B only
            o_l0 = T.alloc_L0C([C, N], accum)
            T.copy(AH[b, h, 0, 0], ah_l1)
            T.copy(AL[b, h, 0, 0], al_l1)
            T.copy(X[b, h, 0, 0], x_l1)
            T.gemm_v0(ah_l1, x_l1, o_l0, init=True)
            T.gemm_v0(al_l1, x_l1, o_l0, init=False)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_ab1(B, H, C, N, dtype="float16", accum="float"):
    """O = A @ X，单次（作为对照）。"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, C], dtype), X: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)
            x_l1 = T.alloc_L1([C, N], dtype)
            o_l0 = T.alloc_L0C([C, N], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(X[b, h, 0, 0], x_l1)
            T.gemm_v0(a_l1, x_l1, o_l0, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


def main():
    B, H, C, N = 1, 2, 64, 64
    # A 模拟 solve_a 的输出：下三角、量级 ~1e-1（与真实 A 相近）
    a64 = np.tril(np.random.randn(B, H, C, C), -1) + np.eye(C)[None, None]
    a64 = a64.astype(np.float64)
    x64 = np.random.randn(B, H, C, N)

    a_hi = a64.astype(np.float16)
    a_lo = (a64 - a_hi.astype(np.float64)).astype(np.float16)
    x16 = x64.astype(np.float16)
    print(f"|A|max={np.abs(a64).max():.3e}  |A_lo|max={np.abs(a_lo).max():.3e} "
          f"(lo/hi = {np.abs(a_lo).max() / np.abs(a_hi).max():.2e})")

    ref = a64 @ x64
    single = k_ab1(B, H, C, N)(torch.from_numpy(a_hi).npu(), torch.from_numpy(x16).npu())
    split = k_ab2(B, H, C, N)(
        torch.from_numpy(a_hi).npu(), torch.from_numpy(a_lo).npu(), torch.from_numpy(x16).npu())
    torch.npu.synchronize()

    for name, t in (("single", single), ("split", split)):
        got = t.float().cpu().numpy().astype(np.float64)
        rel = float(np.abs(got - ref).max() / np.abs(ref).max())
        print(f"SPLIT {name:6s} rel={rel:.3e}")
    print("SPLIT-DONE")


if __name__ == "__main__":
    main()

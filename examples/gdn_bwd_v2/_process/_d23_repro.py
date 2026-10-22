# d23_repro.py -- 最小复现：kernel 内循环里 `gemm_v0(init=True)` 只清一次
#
# 期望（文档语义）：每次 init=True 都清零 ⇒ 输出 = A @ X
# 实际（本 fork）：只有第一次清 ⇒ 输出 = 2 * (A @ X)
#
# 根因（源码级）：tilelang/language/ascend.py 的 gemm_v0 里
#     Cptr = _retrieve_ptr(C, "w" if init is True else "rw")
# 把 L0C 标成"只写"，依赖分析因此认为旧值 dead，清零可被提出循环。
#
# 运行：python d23_repro.py     （需要 NPU）
import numpy as np
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

PASS_CFG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=PASS_CFG)
def two_inits(C=64):
    """同一个 L0C 上连续两次 init=True 的 GEMM。"""

    @T.prim_func
    def main(A: T.Tensor([C, C], "float16"), X: T.Tensor([C, C], "float16"),  # type: ignore
             O: T.Tensor([C, C], "float16")):  # type: ignore
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            a_l1 = T.alloc_L1([C, C], "float16")
            x_l1 = T.alloc_L1([C, C], "float16")
            o_l0 = T.alloc_L0C([C, C], "float")
            T.copy(A[0, 0], a_l1)
            T.copy(X[0, 0], x_l1)
            for _ in T.serial(2):
                T.gemm_v0(a_l1, x_l1, o_l0, init=True)
            T.copy(o_l0, O[0, 0])

    return main


def main():
    torch.manual_seed(0)
    C = 64
    a = torch.randn(C, C, dtype=torch.float16)
    x = torch.randn(C, C, dtype=torch.float16)
    out = two_inits(C)(a.npu(), x.npu())
    torch.npu.synchronize()
    got = out.float().cpu().numpy().astype(np.float64)
    ref = a.double().numpy() @ x.double().numpy()
    twice = 2 * ref
    d1 = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    d2 = float(np.abs(got - twice).max() / (np.abs(twice).max() + 1e-30))
    print(f"D23 rel_vs_once={d1:.3e}  rel_vs_twice={d2:.3e}")
    if d2 < 1e-2 < d1:
        print("D23 verdict=REPRODUCED  （init=True 只在第一轮生效）")
    elif d1 < 1e-2:
        print("D23 verdict=NOT_REPRODUCED  （每次 init 都清零）")
    else:
        print("D23 verdict=INCONCLUSIVE")
    print("D23-DONE")


if __name__ == "__main__":
    main()

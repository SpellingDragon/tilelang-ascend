# _d23b_repro.py -- D23 的第二轮复现：循环内 GEMM(init=True) **并落 GM**
#
# 第一轮（_d23_repro.py）只做"纯 GEMM 循环"，两次都正确清零 ⇒ 不复现。
# 而 v2/v3 的真实写法是"每轮 GEMM 后把 L0C 拷到 GM 的第 c 块"，据此怀疑：
# 清零点与 GM 写回的交互才是根因，而不是 init 本身。
#
# 判读：
#   每块都 == A@X  ⇒ 该形态也不复现（D23 彻底不成立，只能报"更复杂的循环"）
#   第 2 块 == 2*(A@X) 或残留  ⇒ 复现，可提 issue
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
def loop_gemm_gm(C=64, NC=2):
    """每轮：GEMM(init=True) → L0C 拷到 GM 的第 c 块。"""

    @T.prim_func
    def main(A: T.Tensor([C, C], "float16"), X: T.Tensor([C, C], "float16"),  # type: ignore
             O: T.Tensor([NC, C, C], "float16")):  # type: ignore
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            a_l1 = T.alloc_L1([C, C], "float16")
            x_l1 = T.alloc_L1([C, C], "float16")
            o_l0 = T.alloc_L0C([C, C], "float")
            T.copy(A[0, 0], a_l1)
            T.copy(X[0, 0], x_l1)
            for c in T.serial(NC):
                T.gemm_v0(a_l1, x_l1, o_l0, init=True)
                T.copy(o_l0, O[c, 0, 0])

    return main


def run(name, ker):
    torch.manual_seed(0)
    C, NC = 64, 2
    a = torch.randn(C, C, dtype=torch.float16)
    x = torch.randn(C, C, dtype=torch.float16)
    try:
        out = ker(C, NC)(a.npu(), x.npu())
        torch.npu.synchronize()
        got = out.float().cpu().numpy().astype(np.float64)
        ref = a.double().numpy() @ x.double().numpy()
        for c in range(NC):
            rel = float(np.abs(got[c] - ref).max() / (np.abs(ref).max() + 1e-30))
            rel2 = float(np.abs(got[c] - 2 * ref).max() / (np.abs(2 * ref).max() + 1e-30))
            print(f"{name} c={c} rel_vs_once={rel:.3e} rel_vs_twice={rel2:.3e}")
    except Exception as exc:                              # noqa: BLE001
        print(f"{name}: RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")


def main():
    run("gemm_then_gm", loop_gemm_gm)
    print("D23B-DONE")


if __name__ == "__main__":
    main()

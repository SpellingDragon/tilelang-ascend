# _1198_repro.py -- #1198 的可执行证据：GM 切片里 `:` + 标量混用会让 extent 塌缩
#
# 三种写法搬同一块数据（W[0] 的 C×N 面板）：
#   A  W[0, 0, 0]        全标量起点（隐含整块）      —— 基线
#   B  W[0, :, 0]        `:` 与标量混用              —— #1198 报的形态
#   C  W[0, 0, 0] + 显式区间（用 slice 表达）        —— 推荐写法
# 期望三者输出一致；若 B 只搬一列（其余为 0/垃圾），即复现。
#
# 用途：作为 issue #1198 的最小证据（不是回归测试——上游未修时它会红）。
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


@tilelang.jit(out_idx=[1], pass_configs=PASS_CFG)
def variant(mode, NC=2, C=64, N=64):
    @T.prim_func
    def main(W: T.Tensor([NC, C, N], "float16"), O: T.Tensor([C, N], "float16")):  # type: ignore
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            l1 = T.alloc_L1([C, N], "float16")
            ub = T.alloc_ub([C, N], "float16")
            if mode == "A":
                T.copy(W[0, 0, 0], l1)
            elif mode == "B":
                T.copy(W[0, :, 0], l1)
            else:
                T.copy(W[0, 0: C, 0: N], l1)
            T.copy(l1, ub)
            T.copy(ub, O[0, 0])

    return main


def main():
    torch.manual_seed(0)
    NC, C, N = 2, 64, 64
    w = torch.randn(NC, C, N, dtype=torch.float16)
    ref = w.double().numpy()[0]
    outs = {}
    for mode in ("A", "B", "C"):
        try:
            o = variant(mode, NC, C, N)(w.npu())
            torch.npu.synchronize()
            outs[mode] = o.float().cpu().numpy().astype(np.float64)
            rel = float(np.abs(outs[mode] - ref).max() / np.abs(ref).max())
            nz_cols = int((np.abs(outs[mode]).sum(0) > 0).sum())
            print(f"slice {mode}: rel={rel:.3e} non_empty_cols={nz_cols}/{N}")
        except Exception as exc:                          # noqa: BLE001
            print(f"slice {mode}: RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
    if "A" in outs and "B" in outs:
        d = float(np.abs(outs["A"] - outs["B"]).max())
        print(f"slice A_vs_B maxdiff={d:.3e}")
        print("1198 verdict=%s" % ("REPRODUCED" if d > 1e-3 else "NOT_REPRODUCED"))
    print("1198-DONE")


if __name__ == "__main__":
    main()

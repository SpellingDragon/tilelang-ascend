# _1198_repro.py -- #1198 的同 rank 版复现（2-D GM → 2-D L1，末尾标量索引）
#
# 上一版把 [NC,C,N] 的 3-D GM 直接 T.copy 到 alloc_L1([C,N])，三种写法全部 InternalError
# ⇒ 崩在 rank 不匹配上，测不到 extent（见 notes [131]）。
#
# 本版把输入压成 2-D，让三种索引形态都在**同一 rank 内**，才可能暴露 extent 塌缩：
#   A  W[0, 0]          全标量起点（基线）
#   B  W[0, :]          前置标量 + 全切片   ← #1198 报的形态
#   C  W[0:C, 0:N]      显式区间
# 期望三者输出一致；若 B 只搬一行（其余为 0/垃圾），即复现。
import numpy as np
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

PASS_CFG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[1], pass_configs=PASS_CFG)
def variant(mode, C=64, N=64):
    """同 rank：GM[C,N] -> L1[C,N] -> UB -> GM[C,N]。"""

    @T.prim_func
    def main(W: T.Tensor([C, N], "float16"), O: T.Tensor([C, N], "float16")):  # type: ignore
        with T.Kernel(1, threads=1, is_npu=True) as (cid):  # noqa: F841  (单块，不用 cid)
            l1 = T.alloc_L1([C, N], "float16")
            ub = T.alloc_ub([C, N], "float16")
            if mode == "A":
                T.copy(W[0, 0], l1)
            elif mode == "B":
                T.copy(W[0, :], l1)
            else:
                T.copy(W[0:C, 0:N], l1)
            T.copy(l1, ub)
            T.copy(ub, O[0, 0])

    return main


def main():
    torch.manual_seed(0)
    C = N = 64
    w = torch.randn(C, N, dtype=torch.float16)
    ref = w.double().numpy()
    outs = {}
    for mode in ("A", "B", "C"):
        try:
            o = variant(mode, C, N)(w.npu())
            torch.npu.synchronize()
            got = o.float().cpu().numpy().astype(np.float64)
            outs[mode] = got
            rel = float(np.abs(got - ref).max() / np.abs(ref).max())
            nz_rows = int((np.abs(got).sum(1) > 0).sum())
            print(f"slice {mode}: rel={rel:.3e} non_empty_rows={nz_rows}/{C}", flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f"slice {mode}: RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
    if "A" in outs and "B" in outs:
        d = float(np.abs(outs["A"] - outs["B"]).max())
        print(f"slice A_vs_B maxdiff={d:.3e}", flush=True)
        print("1198 verdict=%s" % ("REPRODUCED" if d > 1e-3 else "NOT_REPRODUCED"), flush=True)
    else:
        print("1198 verdict=INCONCLUSIVE（有写法在 copy 阶段就崩）", flush=True)
    print("1198-DONE", flush=True)


if __name__ == "__main__":
    main()

# d24_repro.py -- 最小复现：同一 2-D `T.Parallel` 里读两个 1-D buffer ⇒ 507015
#
# 对照设计：
#   one_1d  —— 只读一个 1-D（be[i]），预期正常
#   two_1d  —— 同时读两个 1-D（be[i] 与 e[i]），预期触发 507015
# 两次都在同一进程里 try/except 捕获，便于一次运行给出对照结论。
#
# 官方 S2 FAQ 对 507015 的定性："UB 地址分配和切分出现问题，访问了错误的地址"。
# 本文件把"两个 1-D 同处一个 2-D 循环"这个具体形态固化下来。
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


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def one_1d(C=64, N=64):
    @T.prim_func
    def main(X: T.Tensor([C, N], "float16"), Be: T.Tensor([C], "float"),  # type: ignore
             E: T.Tensor([C], "float"),  # type: ignore
             O: T.Tensor([C, N], "float16")):  # type: ignore
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            x16 = T.alloc_ub([C, N], "float16")
            x32 = T.alloc_ub([C, N], "float")
            be = T.alloc_ub([C], "float")
            T.copy(X[0, 0], x16)
            T.copy(Be[0], be)
            T.copy(x16, x32)
            for i, j in T.Parallel(C, N):
                x32[i, j] = x32[i, j] * be[i]          # 只一个 1-D
            T.copy(x32, x16)
            T.copy(x16, O[0, 0])

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def two_1d(C=64, N=64):
    @T.prim_func
    def main(X: T.Tensor([C, N], "float16"), Be: T.Tensor([C], "float"),  # type: ignore
             E: T.Tensor([C], "float"),  # type: ignore
             O: T.Tensor([C, N], "float16")):  # type: ignore
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            x16 = T.alloc_ub([C, N], "float16")
            x32 = T.alloc_ub([C, N], "float")
            be = T.alloc_ub([C], "float")
            e = T.alloc_ub([C], "float")
            T.copy(X[0, 0], x16)
            T.copy(Be[0], be)
            T.copy(E[0], e)
            T.copy(x16, x32)
            for i, j in T.Parallel(C, N):
                x32[i, j] = x32[i, j] * (be[i] * e[i])   # 两个 1-D
            T.copy(x32, x16)
            T.copy(x16, O[0, 0])

    return main


def run(name, ker, x, be, e):
    try:
        out = ker(x.npu(), be.npu(), e.npu())
        torch.npu.synchronize()
        got = out.float().cpu().numpy().astype(np.float64)
        ref = x.double().numpy() * be.double().numpy()[:, None] * e.double().numpy()[:, None]
        rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
        print(f"{name}: OK rel={rel:.3e}")
        return "ok"
    except Exception as exc:                              # noqa: BLE001
        msg = str(exc).split("\n")[0][:120]
        print(f"{name}: RAISED {type(exc).__name__}: {msg}")
        return "raised"


def main():
    torch.manual_seed(0)
    C = N = 64
    x = torch.randn(C, N, dtype=torch.float16)
    be = torch.rand(C, dtype=torch.float32)
    e = torch.rand(C, dtype=torch.float32)
    a = run("one_1d", one_1d(C, N), x, be, e)
    b = run("two_1d", two_1d(C, N), x, be, e)
    if a == "ok" and b == "raised":
        print("D24 verdict=REPRODUCED  （一个 1-D 正常、两个 1-D 触发异常）")
    elif a == "ok" and b == "ok":
        print("D24 verdict=NOT_REPRODUCED")
    else:
        print("D24 verdict=INCONCLUSIVE")
    print("D24-DONE")


if __name__ == "__main__":
    main()

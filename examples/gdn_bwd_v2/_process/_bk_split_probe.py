# _bk_split_probe.py -- 把 `solve_a` 里 bk_num>1 才走的两条路径分开
#
# 背景（notes [152]）：DK=64 时 `bk_num = ceildiv(DK, BK) = 1`，`solve_a` 的 KK^T 分块循环
# 只跑 i=0，于是下面两条路径**从未真正执行过**；DK=128 时 bk_num=2，A 直接错 14%。
#
#   (1) 列偏移非零的 strided 拷贝：`T.copy(K[..., bx*C, i*BK], k_half)`，i=1 时列偏移 64
#   (2) 循环内动态 init：`T.gemm_v0(..., init=(i == 0))`
#
# 本支各配一条 CONTROL（偏移 0 / 单次 GEMM），任一条 CONTROL FAIL 即本支作废。
# 两条都 PASS 而 TEST FAIL，就把“哪一条坏”钉死；两条 TEST 都 PASS，
# 说明问题在 `solve_a` 里更靠后的东西（A = I - L* 的替换、掩码、broadcast），
# 那就转去看 A 的逐元素误差形状，而不是继续猜。
#
# 打印：CONTROL name=.. rel=.. PASS|FAIL ；BKS name=.. rel=.. PASS|FAIL
import numpy as np
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

_PASS = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}
C, DK, BK = 64, 128, 64
GATE = 5e-3


@tilelang.jit(out_idx=[1], pass_configs=_PASS)
def copy_col(B, C, DK, BK, off, dtype="float16"):
    """把 X[:, off:off+BK] 拷到 Y（全部打满，验证带列偏移的 strided 拷贝）"""

    @T.prim_func
    def main(X: T.Tensor([B, C, DK], dtype), Y: T.Tensor([B, C, BK], dtype)):  # type: ignore
        with T.Kernel(B, threads=1, is_npu=True) as (cid):
            h = T.alloc_ub([C, BK], dtype)
            T.tile.fill(h, 0.0)
            T.copy(X[cid, 0, off], h)
            T.copy(h, Y[cid, 0, 0])

    return main


@tilelang.jit(out_idx=[1], pass_configs=_PASS)
def kk_split(B, C, DK, BK, nb, dtype="float16", accum="float"):
    """KK^T = sum_i K[:, i*BK:(i+1)*BK] K[:, i*BK:(i+1)*BK]^T，用 `init=(i == 0)` 累加。"""

    @T.prim_func
    def main(K: T.Tensor([B, C, DK], dtype), O: T.Tensor([B, C, C], dtype)):  # type: ignore
        with T.Kernel(B, threads=1, is_npu=True) as (cid):
            k_l1 = T.alloc_L1([C, BK], dtype)
            kk_l0 = T.alloc_L0C([C, C], accum)
            for i in T.serial(nb):
                T.copy(K[cid, 0, i * BK], k_l1)
                T.gemm_v0(k_l1, k_l1, kk_l0, transpose_B=True, init=(i == 0))
            T.copy(kk_l0, O[cid, 0, 0])

    return main


def _rel(o, ref):
    got = o.float().cpu().numpy().astype(np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.abs(got - r).max()) / (float(np.abs(r).max()) + 1e-30)


def copy_case(off, tag):
    torch.manual_seed(7)
    X = torch.randn(1, C, DK, dtype=torch.float16)
    Y = copy_col(1, C, DK, BK, off)(X.npu())
    torch.npu.synchronize()
    rel = _rel(Y, X[:, :, off:off + BK].double().numpy())
    print(f"{tag} name=copy off={off:<3} rel={rel:.3e} "
          f"{'PASS' if rel < GATE else 'FAIL'}", flush=True)
    return rel < GATE


def kk_case(nb, tag):
    torch.manual_seed(7)
    K = torch.randn(1, C, DK, dtype=torch.float16)
    O = kk_split(1, C, DK, BK, nb)(K.npu())
    torch.npu.synchronize()
    Kd = K.double().numpy()
    ref = Kd @ Kd.transpose(0, 2, 1)                    # 全宽 KK^T（nb 个块之和）
    rel = _rel(O, ref)
    print(f"{tag} name=kk   nb={nb:<3} rel={rel:.3e} "
          f"{'PASS' if rel < GATE else 'FAIL'}", flush=True)
    return rel < GATE


def main():
    print("BKS start（两条 CONTROL 先跑；任一 FAIL 则本支作废）", flush=True)
    ctl = [
        copy_case(0, "CONTROL"),        # 列偏移 0：solve_a 在 DK=64 下走的就是这条
        kk_case(1, "CONTROL"),          # 单块累加：DK=64 下走的就是这条
    ]
    if not all(ctl):
        print("BKS verdict=CONTROL_FAILED", flush=True)
        print("BKS-DONE", flush=True)
        return
    copy_case(BK, "BKS")                # (1) 列偏移非零
    kk_case(2, "BKS")                   # (2) 两个块 + 动态 init
    print("BKS-DONE", flush=True)


if __name__ == "__main__":
    main()

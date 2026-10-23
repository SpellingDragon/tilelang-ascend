# _gemm_k_probe.py -- 最小 GEMM：只改收缩维 K（64/128/256），看后端在 K=128 是否丢项
#
# 为什么要这支：notes [148] 定住 DK/DV=128 是坏的维度。把六项的依赖摊开看：
#
#   dh0  = ta32(edo,Q) + ta32(dvnew,wm)      —— 只有 K=C=64 的收缩，**唯一达标**
#   dq   = dqkt@K + edo@H_in                 —— 含 K=DV / K=DK
#   dv   = beta .* ta(A^T, dvnew)            —— dvnew 含 Kt(Δ') 的 K=DK 项
#   dbeta= r1 - E*r2 + r3                    —— r2 经 dw = dvnew@H_in（K=DV）
#   dg   = ... qh=q@H^T, kk=K K^T            —— 两者都是 K=DK
#
# 即：**失败的五项都至少含一个 K=128 的收缩，唯一达标的 dh0 一个都没有。**
# 本支把这层单独拿出来：同一个 kernel 结构，只把 K 从 64 调到 128/256，
# 与 fp64 参考比。若 K=64 准、K=128 失准，那问题在 GEMM 的 K 维，不在 GDN 的公式。
#
# 打印：GEOMM-K M=.. N=.. K=.. rel=..   （fp16 输入、fp32 累加，与 v3 同款）
import numpy as np
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

_PASS = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gemm_mkn(B, M, K, N, dtype="float16", accum="float"):
    """O = A @ X,  A:[B,M,K], X:[B,K,N] -> [B,M,N]（无 batch/head 维，只测 K）"""

    @T.prim_func
    def main(A: T.Tensor([B, M, K], dtype), X: T.Tensor([B, K, N], dtype),  # type: ignore
             O: T.Tensor([B, M, N], dtype)):  # type: ignore
        with T.Kernel(B, threads=1, is_npu=True) as (cid):
            a_l1 = T.alloc_L1([M, K], dtype)
            x_l1 = T.alloc_L1([K, N], dtype)
            o_l0 = T.alloc_L0C([M, N], accum)
            T.copy(A[cid, 0, 0], a_l1)
            T.copy(X[cid, 0, 0], x_l1)
            T.gemm_v0(a_l1, x_l1, o_l0, init=True)
            T.copy(o_l0, O[cid, 0, 0])

    return main


def one(M, N, K, C=64):
    """K = C*n，模拟 GDN 里 C=64 的 chunk 结构与"K 由 DK/DV 决定"的事实。"""
    torch.manual_seed(7)
    A = torch.randn(1, M, K, dtype=torch.float16).npu()
    X = torch.randn(1, K, N, dtype=torch.float16).npu()
    ref = (A[0].float().cpu().numpy().astype(np.float64)
           @ X[0].float().cpu().numpy().astype(np.float64))
    out = gemm_mkn(1, M, K, N)(A, X)
    got = out[0].float().cpu().numpy().astype(np.float64)
    rel = float(np.abs(got - ref).max()) / (float(np.abs(ref).max()) + 1e-30)
    print(f"GEMM-K M={M} N={N} K={K:<4} rel={rel:.3e} "
          f"verdict={'PASS' if rel < 5e-3 else 'FAIL'}", flush=True)
    return rel


def main():
    print("GEMM-K start（fp16 入、fp32 累加；K=64/128/256）", flush=True)
    # 两个形状各测一遍：v3 里的代表形状
    for M, N in ((64, 128), (128, 128)):
        for K in (64, 128, 256):
            try:
                one(M, N, K)
            except Exception as exc:                        # noqa: BLE001
                print(f"GEMM-K M={M} N={N} K={K} RAISED {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:110]}", flush=True)
    print("GEMM-K-DONE", flush=True)


if __name__ == "__main__":
    main()

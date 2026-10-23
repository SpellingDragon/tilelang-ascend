"""gdn_bwd_v3 -- GDN 反向的 A 路实现（每 kernel 单 GEMM，chunk 循环在 host）。

每一条约束都来自本仓实测（notes [72]-[88]、`release/ascend/port910b/npu_memory_model.md`）：

  * 一个 kernel 只做一个 GEMM，或只做一段纯 vector 运算（不混 cube/vec）
  * chunk 循环放 host：每次 launch 都是 kernel 的"首次"，`init=True` 才可靠（D23）
  * 每条 L1 单一职责：A 或 B（D22：L0A/L0B 分形不同，共用会静默算 0）
  * GM 切片用显式区间；fp32 参数（Gam/Msk/E）以 fp32 传入

**能力边界（务必按此使用）**：
  * `dq` —— 两项（`dqkt K` 与 `e_do H_in`）合起来就是完整梯度。
  * `dk` —— 只含 `dqkt^T Q`（B4'）；**缺**跨 chunk 的 `Kt^T Delta'` 与 `th` 缩放。
  * `dv` —— 只含 `beta .* (A^T dVnew)`（B7）；**缺** `dKt` 相关的跨 chunk 项。
  因此多 chunk 下 dk/dv 与 golden 会有系统差；单 chunk（末 chunk 的 Delta'=0）一致。
dbeta / dg / dh0 尚未接进本文件——形态已分别由 rows / dgss / dh0b 三个探针验证。
"""

import os

import torch
import tilelang
import tilelang.language as T

_PASS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


# ---------------------------- cube: 一个 kernel 一个 GEMM ----------------------------
@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_ab(B, H, C, N, dtype="float16", accum="float"):
    """U = A @ X   （A 是下三角 [C,C]，X 是 [C,N]）"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, C], dtype), X: T.Tensor([B, H, C, N], dtype),  # type: ignore
             U: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)   # A only
            x_l1 = T.alloc_L1([C, N], dtype)   # B only
            o_l0 = T.alloc_L0C([C, N], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(X[b, h, 0, 0], x_l1)
            T.gemm_v0(a_l1, x_l1, o_l0, init=True)
            T.copy(o_l0, U[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_wh(B, H, C, DV, DK, dtype="float16", accum="float"):
    """WH = W @ H^T   （W [C,DK], H [DV,DK]）"""

    @T.prim_func
    def main(W: T.Tensor([B, H, C, DK], dtype), Hh: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
             WH: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            w_l1 = T.alloc_L1([C, DK], dtype)     # A only
            hin_l1 = T.alloc_L1([DV, DK], dtype)  # B only
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(W[b, h, 0, 0], w_l1)
            T.copy(Hh[b, h, 0, 0], hin_l1)
            T.gemm_v0(w_l1, hin_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, WH[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[4], pass_configs=_PASS)
def gdn_v3_vnew(B, H, C, DV, DK, dtype="float16", accum="float"):
    """Vnew = A (beta V) + (-W) H^T，两项累加在同一 L0C。

    相比 `A@(βV)` → GM → `W@H^T` → GM → 减法，少三次 fp16 舍入与三次 GM 往返；
    **注意** 这是 no-transpose 与 transpose_B 混合同族累加，需上卡确认（D22 家族）。
    """

    @T.prim_func
    def main(A: T.Tensor([B, H, C, C], dtype), BV: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             WN: T.Tensor([B, H, C, DK], dtype), Hh: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
             VN: T.Tensor([B, H, C, DV], accum)):  # type: ignore  fp32：下游 hi/lo 拆分的源
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)      # A only
            bv_l1 = T.alloc_L1([C, DV], dtype)    # B only
            wn_l1 = T.alloc_L1([C, DK], dtype)    # A only
            hin_l1 = T.alloc_L1([DV, DK], dtype)  # B only
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(BV[b, h, 0, 0], bv_l1)
            T.copy(WN[b, h, 0, 0], wn_l1)
            T.copy(Hh[b, h, 0, 0], hin_l1)
            T.gemm_v0(a_l1, bv_l1, o_l0, init=True)
            T.gemm_v0(wn_l1, hin_l1, o_l0, transpose_B=True, init=False)
            T.copy(o_l0, VN[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_raw(B, H, C, DK, dtype="float16", accum="float"):
    """raw = Q K^T"""

    @T.prim_func
    def main(Q: T.Tensor([B, H, C, DK], dtype), K: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             R: T.Tensor([B, H, C, C], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            q_l1 = T.alloc_L1([C, DK], dtype)  # A only
            k_l1 = T.alloc_L1([C, DK], dtype)  # B only
            o_l0 = T.alloc_L0C([C, C], accum)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.copy(K[b, h, 0, 0], k_l1)
            T.gemm_v0(q_l1, k_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, R[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_dp(B, H, C, DV, dtype="float16", accum="float"):
    """dP = dO Vnew^T"""

    @T.prim_func
    def main(DO: T.Tensor([B, H, C, DV], dtype), VN: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             DP: T.Tensor([B, H, C, C], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            do_l1 = T.alloc_L1([C, DV], dtype)  # A only
            vn_l1 = T.alloc_L1([C, DV], dtype)  # B only
            o_l0 = T.alloc_L0C([C, C], accum)
            T.copy(DO[b, h, 0, 0], do_l1)
            T.copy(VN[b, h, 0, 0], vn_l1)
            T.gemm_v0(do_l1, vn_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, DP[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[4], pass_configs=_PASS)
def gdn_v3_dq(B, H, C, DK, DV, dtype="float16", accum="float"):
    """dq = dqkt K + e_do H_in（同族 no-transpose，两项累加；dq5 已证）"""

    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), K: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             EDO: T.Tensor([B, H, C, DV], dtype), HH: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
             DQ: T.Tensor([B, H, C, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            dqkt_l1 = T.alloc_L1([C, C], dtype)   # A
            k_l1 = T.alloc_L1([C, DK], dtype)     # B
            edo_l1 = T.alloc_L1([C, DV], dtype)   # A
            hin_l1 = T.alloc_L1([DV, DK], dtype)  # B
            o_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(DQKT[b, h, 0, 0], dqkt_l1)
            T.copy(K[b, h, 0, 0], k_l1)
            T.copy(EDO[b, h, 0, 0], edo_l1)
            T.copy(HH[b, h, 0, 0], hin_l1)
            T.gemm_v0(dqkt_l1, k_l1, o_l0, init=True)
            T.gemm_v0(edo_l1, hin_l1, o_l0, init=False)
            T.copy(o_l0, DQ[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[3], pass_configs=_PASS)
def gdn_v3_dk(B, H, C, DK, dtype="float16", accum="float"):
    """dk = dqkt^T Q"""

    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), DLO: T.Tensor([B, H, C, C], dtype),  # type: ignore
             Q: T.Tensor([B, H, C, DK], dtype), O: T.Tensor([B, H, C, DK], accum)):  # type: ignore
        # compensated GEMM（同 _ref:412-470）：hi 一次 + lo 一次累加在同一个 L0C。
        # O 走 fp32，避免最后一次 2^-11 的舍入。
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)   # A (transposed)
            q_l1 = T.alloc_L1([C, DK], dtype)  # B
            lo_l1 = T.alloc_L1([C, C], dtype)   # lo 半
            o_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(DQKT[b, h, 0, 0], a_l1)
            T.copy(DLO[b, h, 0, 0], lo_l1)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.gemm_v0(a_l1, q_l1, o_l0, transpose_A=True, init=True)
            T.gemm_v0(lo_l1, q_l1, o_l0, transpose_A=True, init=False)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_dv(B, H, C, DV, dtype="float16", accum="float"):
    """t = A^T dVnew"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, C], dtype), VN: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             O: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)   # A (transposed)
            v_l1 = T.alloc_L1([C, DV], dtype)  # B
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(VN[b, h, 0, 0], v_l1)
            T.gemm_v0(a_l1, v_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


# ---------------------------- vector: 一段纯 UB 运算 ----------------------------
@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_scale(B, H, C, N, dtype="float16", accum="float"):
    """Y = beta .* X   （逐行缩放；也用于 beta .* E .* K 的第二段）"""

    @T.prim_func
    def main(X: T.Tensor([B, H, C, N], dtype), Beta: T.Tensor([B, H, C], "float"),  # type: ignore
             Y: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x16 = T.alloc_ub([C, N], dtype)
            x32 = T.alloc_ub([C, N], accum)
            be = T.alloc_ub([C], accum)
            T.copy(X[b, h, 0, 0], x16)
            T.copy(Beta[b, h, 0], be)
            T.copy(x16, x32)
            for i, j in T.Parallel(C, N):
                x32[i, j] = x32[i, j] * be[i]
            T.copy(x32, x16)
            T.copy(x16, Y[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[3], pass_configs=_PASS)   # Y is the 4th param; E (fp32) is an input
def gdn_v3_scale_be(B, H, C, N, dtype="float16", accum="float"):
    """Y = (beta .* E) .* X   （βEK 用；E 为 per-token fp32）"""

    @T.prim_func
    def main(X: T.Tensor([B, H, C, N], dtype), Beta: T.Tensor([B, H, C], "float"),  # type: ignore
             E: T.Tensor([B, H, C], "float"),  # type: ignore
             Y: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x16 = T.alloc_ub([C, N], dtype)
            x32 = T.alloc_ub([C, N], accum)
            be = T.alloc_ub([C], accum)
            e = T.alloc_ub([C], accum)
            T.copy(X[b, h, 0, 0], x16)
            T.copy(Beta[b, h, 0], be)
            T.copy(E[b, h, 0], e)
            T.copy(x16, x32)
            for i, j in T.Parallel(C, N):
                x32[i, j] = x32[i, j] * (be[i] * e[i])
            T.copy(x32, x16)
            T.copy(x16, Y[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_sub(B, H, C, DV, dtype="float16", accum="float"):
    """Vnew = U0 - WH"""

    @T.prim_func
    def main(U0: T.Tensor([B, H, C, DV], dtype), WH: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             VN: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            u16 = T.alloc_ub([C, DV], dtype)
            u32 = T.alloc_ub([C, DV], accum)
            w16 = T.alloc_ub([C, DV], dtype)
            o16 = T.alloc_ub([C, DV], dtype)
            T.copy(U0[b, h, 0, 0], u16)   # same-dtype load, cast in UB
            T.copy(u16, u32)
            T.copy(WH[b, h, 0, 0], w16)
            for i, j in T.Parallel(C, DV):
                u32[i, j] = u32[i, j] - T.cast(w16[i, j], accum)
            T.copy(u32, o16)
            T.copy(o16, VN[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[3], pass_configs=_PASS)
def gdn_v3_gate(B, H, C, dtype="float16", accum="float"):
    """O = X .* Gam .* M"""

    @T.prim_func
    def main(X: T.Tensor([B, H, C, C], dtype), Gam: T.Tensor([B, H, C, C], "float"),  # type: ignore
             Msk: T.Tensor([B, H, C, C], "float"),  # type: ignore
             O: T.Tensor([B, H, C, C], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x16 = T.alloc_ub([C, C], dtype)
            x32 = T.alloc_ub([C, C], accum)
            g32 = T.alloc_ub([C, C], accum)
            m32 = T.alloc_ub([C, C], accum)
            T.copy(X[b, h, 0, 0], x16)
            T.copy(Gam[b, h, 0, 0], g32)
            T.copy(Msk[b, h, 0, 0], m32)
            T.copy(x16, x32)
            for i, j in T.Parallel(C, C):
                x32[i, j] = x32[i, j] * g32[i, j] * m32[i, j]
            T.copy(x32, x16)
            T.copy(x16, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[5, 6, 7], pass_configs=_PASS)
def gdn_v3_gate_edo(B, H, C, DV, dtype="float16", accum="float"):
    """dqkt = dP .* Gam .* M ;  e_do = E .* dO"""

    @T.prim_func
    def main(DP: T.Tensor([B, H, C, C], dtype), Gam: T.Tensor([B, H, C, C], "float"),  # type: ignore
             Msk: T.Tensor([B, H, C, C], "float"), E: T.Tensor([B, H, C], "float"),  # type: ignore
             DO: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             DQKT: T.Tensor([B, H, C, C], dtype), DQKTL: T.Tensor([B, H, C, C], dtype),  # type: ignore
             EDO: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        # compensated GEMM 的 prep（同 _ref:412-470）：hi = fp16(dqkt)，lo = dqkt - hi。
        # 两半都出 fp16 —— UB→GM 没有 fp32 重载（copy_ub_to_gm 只收 fp16/bf16）。
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            d16 = T.alloc_ub([C, C], dtype)
            d32 = T.alloc_ub([C, C], accum)
            g32 = T.alloc_ub([C, C], accum)
            m32 = T.alloc_ub([C, C], accum)
            e32 = T.alloc_ub([C], accum)
            o16 = T.alloc_ub([C, DV], dtype)
            o32 = T.alloc_ub([C, DV], accum)
            T.copy(DP[b, h, 0, 0], d16)
            T.copy(Gam[b, h, 0, 0], g32)
            T.copy(Msk[b, h, 0, 0], m32)
            T.copy(E[b, h, 0], e32)
            T.copy(DO[b, h, 0, 0], o16)
            T.copy(d16, d32)
            for i, j in T.Parallel(C, C):
                d32[i, j] = d32[i, j] * g32[i, j] * m32[i, j]
            T.copy(d32, d16)
            T.copy(d16, DQKT[b, h, 0, 0])          # hi
            for i, j in T.Parallel(C, C):
                d32[i, j] = d32[i, j] - d16[i, j]  # lo = dqkt - hi（fp32 内减）
            T.copy(d32, d16)
            T.copy(d16, DQKTL[b, h, 0, 0])         # lo
            T.copy(o16, o32)
            for i, j in T.Parallel(C, DV):
                o32[i, j] = o32[i, j] * e32[i]
            T.copy(o32, o16)
            T.copy(o16, EDO[b, h, 0, 0])

    return main



# ---------------------------- relay：反向扫描（P4） ----------------------------
@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_mm(B, H, C, M, N, dtype="float16", accum="float"):
    """O = A @ X,  A:[B,H,C,M], X:[B,H,M,N] -> [B,H,C,N]   （per batch-head）"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, M], dtype), X: T.Tensor([B, H, M, N], dtype),  # type: ignore
             O: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, M], dtype)
            x_l1 = T.alloc_L1([M, N], dtype)
            o_l0 = T.alloc_L0C([C, N], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(X[b, h, 0, 0], x_l1)
            T.gemm_v0(a_l1, x_l1, o_l0, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main

@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_ta(B, H, C, M, N, dtype="float16", accum="float"):
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


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_ta32(B, H, C, M, N, dtype="float16", accum="float"):
    """O = A^T B，**fp32 输出**：Δ' 的两个分量（e_do^T Q、dvnew^T(-W)）用它算，
    print 存到 fp16 GM 会先把 Δ' 的精度压到 5e-4（[109] 的下一层）。"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, M], dtype), Bm: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, M, N], accum)):  # type: ignore
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


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_tb(B, H, C, M, N, dtype="float16", accum="float"):
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

@tilelang.jit(out_idx=[2], pass_configs=_PASS)   # (Ones, X, O) -> O is index 2
def gdn_v3_colsum(B, H, C, N, dtype="float16", accum="float"):
    """列和：ones(16,C) @ X -> [16,N]，第 0 行即 colsum(X)。"""

    @T.prim_func
    def main(Ones: T.Tensor([16, C], dtype), X: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, 16, N], accum)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            ones_l1 = T.alloc_L1([16, C], dtype)
            x_l1 = T.alloc_L1([C, N], dtype)
            o_l0 = T.alloc_L0C([16, N], accum)
            T.copy(Ones[0, 0], ones_l1)
            T.copy(X[b, h, 0, 0], x_l1)
            T.gemm_v0(ones_l1, x_l1, o_l0, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_suffix(B, H, C, dtype="float16", accum="float"):
    """dg[i] = carry + sum_{r>=i} dgam[r]  （纯 vec，与 cumsum 同族）"""

    @T.prim_func
    def main(DG: T.Tensor([B, H, C], "float"), Carry: T.Tensor([B, H], "float"),  # type: ignore
             O: T.Tensor([B, H, C], "float")):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            s = T.alloc_ub([C], accum)
            c0 = T.alloc_ub([1], accum)
            T.copy(DG[b, h, 0], s)
            T.copy(Carry[b, h], c0)
            for i in T.serial(C):
                c0[0] = c0[0] + s[C - 1 - i]
                s[C - 1 - i] = c0[0]
            T.copy(s, O[b, h, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_mul(B, H, C, N, dtype="float16", accum="float"):
    """O = X .* Y （逐元素，两个 [B,H,C,N]）"""

    @T.prim_func
    def main(X: T.Tensor([B, H, C, N], dtype), Y: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, C, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x16 = T.alloc_ub([C, N], dtype)
            x32 = T.alloc_ub([C, N], accum)
            y16 = T.alloc_ub([C, N], dtype)
            o16 = T.alloc_ub([C, N], dtype)
            T.copy(X[b, h, 0, 0], x16)
            T.copy(Y[b, h, 0, 0], y16)
            T.copy(x16, x32)
            for i, j in T.Parallel(C, N):
                x32[i, j] = x32[i, j] * T.cast(y16[i, j], accum)
            T.copy(x32, o16)
            T.copy(o16, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_rowsum(B, H, C, N, dtype="float16", accum="float"):
    """rows = X @ ones(N,16) 的第 0 列 == rowsum(X)"""

    @T.prim_func
    def main(X: T.Tensor([B, H, C, N], dtype), Ones: T.Tensor([N, 16], dtype),  # type: ignore
             R: T.Tensor([B, H, C, 16], accum)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x_l1 = T.alloc_L1([C, N], dtype)
            ones_l1 = T.alloc_L1([N, 16], dtype)
            o_l0 = T.alloc_L0C([C, 16], accum)
            T.copy(X[b, h, 0, 0], x_l1)
            T.copy(Ones[0, 0], ones_l1)
            T.gemm_v0(x_l1, ones_l1, o_l0, init=True)
            T.copy(o_l0, R[b, h, 0, 0])

    return main


# ---------------------------- host：chunk 循环（D23） ----------------------------
def _bcast_last(x, n):
    """把 [B,H,C] 显式扩成 [B,H,C,n]。

    D28：卡上 `x[B,H,C] * y[B,H,C,n]` 这样的跨 rank 广播会按行退化（每行用同一标量），
    dk 因此差 2.529。显式 expand 后降到 1.3e-2。**任何逐元素运算都先过这个函数。**
    """
    return x.unsqueeze(-1).expand(-1, -1, -1, n)



def gdn_bwd_v3(q, k, v, g, beta, h0, do, chunk_size=64):
    """返回 (dq, dk, dv)。前置 S/A/H 复用 v2 的三件（均已真机验证）。"""
    # 卡上是扁平的 /tmp 布局，仓内是包路径；两条都试，避免部署形态差异
    try:
        from gdn_bwd_v2 import (  # noqa: E402
            gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a,
        )
    except ImportError:
        from examples.gdn_bwd_v2.gdn_bwd_v2 import (  # noqa: E402
            gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a,
        )

    B, H, T_len, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    NC = (T_len + C - 1) // C
    dt = q.dtype
    dev = q.device
    dtype = "float16" if dt == torch.float16 else "float32"

    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
    idt = torch.eye(C, dtype=torch.float32, device=dev)
    msk_m2 = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
    msk_m = msk_m2[None, None].expand(B, H, C, C).contiguous()   # kernel declares [B,H,C,C]

    ws_g, ws_gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype=dtype)(
        k, beta, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, 64, 64, dtype=dtype)(
        k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()

    k_ab_k = gdn_v3_ab(B, H, C, DK)
    k_vnew = gdn_v3_vnew(B, H, C, DV, DK)
    k_dp = gdn_v3_dp(B, H, C, DV)
    k_dq = gdn_v3_dq(B, H, C, DK, DV)
    k_dk = gdn_v3_dk(B, H, C, DK)
    k_dv = gdn_v3_dv(B, H, C, DV)
    k_scale = gdn_v3_scale(B, H, C, DV)
    # A1：Kc 的宽度是 DK，不能复用按 DV 声明的 k_scale —— JITKernel 不校验形状，
    # T.copy 按声明形状推 extent ⇒ DK≠DV 时按错 stride 读、静默错（真实 GDN DK=128/C=64）
    k_scale_k = gdn_v3_scale(B, H, C, DK)
    k_raw = gdn_v3_raw(B, H, C, DK)
    k_gate = gdn_v3_gate(B, H, C)
    k_ta = gdn_v3_ta(B, H, C, C, DV)
    k_ge = gdn_v3_gate_edo(B, H, C, DV)

    dq = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dk = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dv = torch.empty(B, H, T_len, DV, dtype=dt, device=dev)

    _DBG = os.environ.get("GDN_V3_DBG") == "1"

    def _step(tag):
        if _DBG:
            torch.npu.synchronize()
            print(f"V3DBG c@{tag}", flush=True)

    for c in range(NC):                              # D23: one launch per chunk
        sl = slice(c * C, (c + 1) * C)
        Qc = q[:, :, sl, :]
        Kc = k[:, :, sl, :]
        Vc = v[:, :, sl, :]
        DOc = do[:, :, sl, :]
        Ac = ws_a[:, :, sl, :]
        Gc = ws_g[:, :, sl]
        Ec = torch.exp(Gc)
        Hc = ws_h[:, :, c, :, :]
        Betac = beta[:, :, sl]
        Gamc = ws_gam[:, :, c, :, :]

        bv = k_scale(Vc, Betac); _step("scale")      # beta .* V
        # -W 的构造：A @ (-(beta E) K) = -W，于是 Vnew 可以在一个 L0C 里累加出来
        bek_n = k_scale_k(Kc, -Betac * Ec); _step("sbe")
        wm_n = k_ab_k(Ac, bek_n); _step("ab_k")      # -W
        vnew32 = k_vnew(Ac, bv, wm_n, Hc); _step("vnew")
        # vnew 以 fp32 落 GM；两个消费点各用 hi/lo 两次 GEMM（[108]）
        v_hi = vnew32.to(dt)
        v_lo = (vnew32 - v_hi.float()).to(dt)

        # P = (QK^T) .* Gam .* M 不是 dq 链的输入（dq 用 dP），只在 dv 链的
        # `dvnew = P^T dO`（golden B1）里需要，因此放到下面 dv 那一步再算。
        dP = (k_dp(DOc, v_hi).float() + k_dp(DOc, v_lo).float()).to(dt); _step("dp")
        dqkt_hi, dqkt_lo, edo = k_ge(dP, Gamc, msk_m, Ec, DOc); _step("ge")
        dq[:, :, sl, :] = k_dq(dqkt_hi, Kc, edo, Hc); _step("dq")
        dk[:, :, sl, :] = k_dk(dqkt_hi, dqkt_lo, Qc); _step("dk")
        # A2：dv = beta .* (A^T dvnew)，dvnew 是**反向**的 `P^T dO`（golden B1），
        # 不是前向的 Vnew。精简版没有 Delta' 反向扫描：单 chunk（Delta'_{c+1}=0）
        # 与 golden 一致，多 chunk 缺 `Kt Delta'^T` 项（见文件头「能力边界」）。
        _P = k_gate(k_raw(Qc, Kc), Gamc, msk_m); _step("p")   # P = (QK^T) .* Gam .* M
        dvnew = k_ta(_P, DOc); _step("dvnew")                 # dvnew = P^T dO
        dv[:, :, sl, :] = k_scale(k_dv(Ac, dvnew), Betac); _step("dv")
    torch.npu.synchronize()
    return dq, dk, dv


def gdn_bwd_v3_relay(q, k, v, g, beta, h0, do, chunk_size=64,
                     hs_override=None, a_override=None):
    """完整反向：先正向缓存 vnew，再反向扫 Delta'，输出 (dq, dk, dv, dh0)。

    依据 `_np_chunk_bwd` 的 B1-B10（见 notes [91]）：
        dvnew_c = P_c^T dO_c + Kt_c (Delta'_{c+1})^T
        dw_c    = -dvnew_c H_in_c
        Delta'_c = e_do_c^T Q_c - dvnew_c^T W_c + glast_c * Delta'_{c+1}
        dk_c    = dqkt_c^T Q_c + (beta E) .* t2_c + th_c .* (vnew_c Delta'_{c+1})
        dv_c    = beta_c .* (A_c^T dvnew_c)
    `dq` 与 relay 无关（`dqkt K + e_do H_in`），直接沿用第一遍。

    **状态：初版，尚未上卡验证。** 已知的简化假设：
      * `dbeta` 已接入（`rowsum(V*t1) + E*rowsum(K*t2)`，两项都用已验证的 rowsum 形态）
      * `dg` 尚未接入：它还需要 `dA` 的两项与 `A^T dA A^T`（求逆的导数），共约 7 次 launch/chunk
      * `dk` 的 `(beta E) .* t2` 已接入（符号折进 `-beta*E`）
      * `glast` 已按 (b,h) 保留，不再跨 batch 取均值
    """
    try:
        from gdn_bwd_v2 import (  # noqa: E402
            gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a,
        )
    except ImportError:
        from examples.gdn_bwd_v2.gdn_bwd_v2 import (  # noqa: E402
            gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a,
        )

    B, H, T_len, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    # A5：尾块未处理（NC 按 ceil 取会读到越界），入口直接拒绝，禁止静默截断
    assert T_len % C == 0, f"T_len={T_len} 必须是 C={C} 的整数倍"
    NC = (T_len + C - 1) // C
    dt = q.dtype
    dev = q.device
    dtype = "float16" if dt == torch.float16 else "float32"

    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
    idt = torch.eye(C, dtype=torch.float32, device=dev)
    msk_m2 = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
    msk_m = msk_m2[None, None].expand(B, H, C, C).contiguous()   # kernel declares [B,H,C,C]
    ws_g, ws_gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype=dtype)(
        k, beta, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, 64, 64, dtype=dtype)(
        k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()

    k_ab_k = gdn_v3_ab(B, H, C, DK)
    k_vnew = gdn_v3_vnew(B, H, C, DV, DK)
    k_raw = gdn_v3_raw(B, H, C, DK)
    k_dp = gdn_v3_dp(B, H, C, DV)
    k_dq = gdn_v3_dq(B, H, C, DK, DV)
    k_dk = gdn_v3_dk(B, H, C, DK)
    k_scale = gdn_v3_scale(B, H, C, DV)
    # A1：喂 Kc/t2n 的三处（bek_n / kt_c / _q2）宽度是 DK，必须用按 DK 声明的实例
    k_scale_k = gdn_v3_scale(B, H, C, DK)
    k_gate = gdn_v3_gate(B, H, C)
    k_ge = gdn_v3_gate_edo(B, H, C, DV)
    k_ta = gdn_v3_ta(B, H, C, C, DV)      # P^T dO
    k_ta_dvn32 = gdn_v3_ta32(B, H, C, DV, DK)  # Δ' 的两个分量，fp32 输出
    k_tb_kt = gdn_v3_tb(B, H, C, DV, DK)   # Kt Delta'^T
    k_mm = gdn_v3_mm(B, H, C, DV, DK)
    k_vnew = gdn_v3_vnew(B, H, C, DV, DK)      # vnew @ Delta'  /  dvnew @ H_in
    k_ta_cdk = gdn_v3_ta(B, H, C, C, DK)   # A^T dw
    k_tb_cv = gdn_v3_tb(B, H, C, C, DV)    # da 的第一项：dvnew (beta V)^T
    k_tb_ck = gdn_v3_tb(B, H, C, C, DK)    # da 的第二项：dw (beta E K)^T
    k_ta_cc = gdn_v3_ta(B, H, C, C, C)     # A^T da
    k_tb_cc = gdn_v3_tb(B, H, C, C, C)     # (A^T da) A^T
    k_rawk = gdn_v3_raw(B, H, C, DK)       # KKt = K K^T（同一形态）
    k_ab_dv = gdn_v3_ab(B, H, C, DV)        # (dx + dx^T) K
    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
    k_mul = gdn_v3_mul(B, H, C, DK)
    k_mulv = gdn_v3_mul(B, H, C, DV)
    k_rows_k = gdn_v3_rowsum(B, H, C, DK)
    k_rows_v = gdn_v3_rowsum(B, H, C, DV)
    ones_k = torch.ones(DK, 16, dtype=dt, device=dev)
    ones_v = torch.ones(DV, 16, dtype=dt, device=dev)

    dq = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dk = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dv = torch.empty(B, H, T_len, DV, dtype=dt, device=dev)
    dbeta = torch.empty(B, H, T_len, dtype=torch.float32, device=dev)
    dg = torch.empty(B, H, T_len, dtype=torch.float32, device=dev)
    # GDN_V3_DK_SPLIT=1 时额外产出 dk 的四项单独结果，用于一次车定位是哪一项错
    # A4：默认关（开启会多返回 1 项并多占 4 份 [B,H,T,DK] 显存），排查时显式置 1
    _split = os.environ.get("GDN_V3_DK_SPLIT", "0") == "1"
    # dk 的写入方式：默认"逐 chunk 切片写"；GDN_V3_ONESHOT_WRITE=1 时先收集再 cat，
    # 用来排除"循环内多次 dk[:,:,sl,:]= 的路径问题"（多占一份 [B,H,T,DK] 内存）
    _oneshot = os.environ.get("GDN_V3_ONESHOT_WRITE", "0") == "1"
    dk_buf = [] if _oneshot else None
    dk_parts = {k: torch.zeros(B, H, T_len, DK, dtype=dt, device=dev)
                for k in ("q1", "q2", "q3", "q4")} if _split else None
    carry = torch.zeros(B, H, dtype=torch.float32, device=dev)
    m_m = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
    k_suffix = gdn_v3_suffix(B, H, C)
    k_tb_qh = gdn_v3_tb(B, H, C, DV, DK)     # qh = q @ H_in^T
    cache = []

    # A4：默认关（开着每 chunk 9 次 print + 每步 npu.synchronize()），排查时显式置 1
    _DBG = os.environ.get("GDN_V3_DBG", "0") == "1"

    def _step(tag):
        if _DBG:
            torch.npu.synchronize()
            print(f"RELAYDBG c@{tag}", flush=True)

    # ---- pass 1: 正向，存 vnew / W / Kt / th / glast ----
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        Qc, Kc, Vc, DOc = q[:, :, sl, :], k[:, :, sl, :], v[:, :, sl, :], do[:, :, sl, :]
        Ac = ws_a[:, :, sl, :] if a_override is None else a_override[:, :, c]
        Gc = ws_g[:, :, sl]
        Ec = torch.exp(Gc)
        Hc = ws_h[:, :, c, :, :] if hs_override is None else hs_override[:, :, c]
        Betac, Gamc = beta[:, :, sl], ws_gam[:, :, c, :, :]
        bv = k_scale(Vc, Betac)
        # 只算 -W：Vnew 需要它做累加，Delta' 的 dvnew^T W 用 -W 也等价（符号进 host 相加）
        bek_n = k_scale_k(Kc, -Betac * Ec)      # -(beta E K)，da 的第二项要用它
        wm_n = k_ab_k(Ac, bek_n)              # -W
        vnew32 = k_vnew(Ac, bv, wm_n, Hc)
        v_hi = vnew32.to(dt)
        v_lo = (vnew32 - v_hi.float()).to(dt)
        raw = k_raw(Qc, Kc)
        _P = k_gate(raw, Gamc, msk_m)
        dP = (k_dp(DOc, v_hi).float() + k_dp(DOc, v_lo).float()).to(dt); _step("dp")
        dqkt32, edo = k_ge(dP, Gamc, msk_m, Ec, DOc); _step("ge")
        # hi/lo 拆分（同 #1802）：hi 给 dq 与 dk 的第一次 GEMM，lo 只给 dk 的第二次
        dqkt_hi = dqkt32.to(dt)
        dqkt_lo = (dqkt32 - dqkt_hi.float()).to(dt)
        dq[:, :, sl, :] = k_dq(dqkt_hi, Kc, edo, Hc)
        cache.append(dict(sl=sl, Qc=Qc, Kc=Kc, Vc=Vc, DOc=DOc, Ac=Ac, Ec=Ec, Hc=Hc,
                          Betac=Betac, v_hi=v_hi, v_lo=v_lo, wm_n=wm_n, bv=bv, bek_n=bek_n, Gamc=Gamc,
                          P=_P, dqkt=dqkt_hi, dqkt_lo=dqkt_lo, edo=edo, dP=dP, raw=raw,
                          glast=torch.exp(Gc[:, :, -1]),
                          th=torch.exp(Gc[:, :, -1:].expand(-1, -1, C) - Gc)))
    torch.npu.synchronize()

    # ---- pass 2: 反向扫 Delta' ----
    dh0 = None
    for c in range(NC - 1, -1, -1):
        d = cache[c]
        sl = d["sl"]
        if _DBG:
            print(f"RELAYP2 c={c}", flush=True)
        # dvnew = P^T dO + Kt Delta'^T
        base = k_ta(d["P"], d["DOc"])
        # Δ' 一律从 cache 的 fp32 现派生 hi/lo，不在循环间保留变量（避免"跨迭代变量"）
        if c >= NC - 1:
            dvnew = base
        else:
            _hi = cache[c + 1]["delta_f32"].to(dt)
            kt_c = k_scale_k(d["Kc"], d["th"])            # golden 用 Kt = th .* K（B1）
            dvnew = base + k_tb_kt(kt_c, _hi)
        # Delta'_c = e_do^T Q - dvnew^T W + glast * Delta'
        tq = k_ta_dvn32(d["edo"], d["Qc"])           # e_do^T Q      (fp32)
        tw_n = k_ta_dvn32(dvnew, d["wm_n"])          # dvnew^T (-W)  (fp32)
        nxt = tq + tw_n                               # Δ' 全程 fp32
        if c < NC - 1:
            nxt = nxt + d["glast"][:, :, None, None] * cache[c + 1]["delta_f32"]
        # 只存 fp32；hi/lo 由消费者现派生（[103]：fp16 每轮舍入一次，拆 hi/lo 得 22 位）
        delta_f32 = nxt
        # 四项各自打印 abs.max，与本地基线对表（3.8~6.3 / 0.13~0.33 / 0~2.5 / 0.38~0.44）
        _dk1 = k_dk(d["dqkt"], d["dqkt_lo"], d["Qc"])
        if _DBG:
            print(f"RELAYQ c={c} q1={_dk1.abs().max():.4e}", flush=True)
        # dk += th .* (vnew Delta')（末 chunk 的 Delta' 为 0，dkt 自然为 0）
        if c < NC - 1:
            nxt_delta = cache[c + 1]["delta_f32"].to(dt)
        else:
            nxt_delta = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
        # H1：DV==C 时 A@X 与 A@[C,N] 同义，优先用已验证的 ab（mm 从未单独验过）
        if c >= NC - 1:
            dkt = torch.zeros_like(dk[:, :, sl, :])
        elif DV == C:
            dkt = k_ab_dv(d["v_hi"], nxt_delta) + k_ab_dv(d["v_lo"], nxt_delta)
        else:
            dkt = k_mm(d["v_hi"], nxt_delta) + k_mm(d["v_lo"], nxt_delta)
        # H2：th 显式展开成 4-D，不依赖 [B,H,C] x [B,H,C,DK] 的广播
        _th4 = _bcast_last(d["th"].float(), DK)
        _q3 = _th4 * dkt.float()          # 跨 chunk 项，保持 fp32 落 GM（T=128 时占 0.29x）
        if _DBG:
            print(f"RELAYQ c={c} q3={_q3.abs().max():.4e}", flush=True)
        # 四项各自是 fp16 kernel 的输出；累加放在 fp32，最后一次性 cast，
        # 避免 3 次 fp16 加法（每次 ulp≈4e-3）把 dk 的误差推到 1e-2 量级
        _dk_acc = _dk1.float() + _q3.float()   # 两项都是 fp32，无额外舍入
        dth = (d["Kc"].float() * dkt.float()).sum(-1)
        # dv = beta .* (A^T dvnew) ;  t1 顺带给出 dbeta 的第一项
        t1 = k_ta(d["Ac"], dvnew)                       # [B,H,C,DV]
        dv[:, :, sl, :] = k_scale(t1, d["Betac"])
        # B6/B8: dw = -dvnew H_in ; t2 = A^T dw = -A^T(dvnew H_in)
        dw = k_mm(dvnew, d["Hc"])                       # [B,H,C,DK]
        t2n = k_ta_cdk(d["Ac"], dw)                     # A^T dw（含负号）
        neg_be = -d["Betac"] * d["Ec"]                  # -(beta E)
        _q2 = k_scale_k(t2n, neg_be)
        if _DBG:
            print(f"RELAYQ c={c} q2={_q2.abs().max():.4e}", flush=True)
        _dk_acc = _dk_acc + _q2.float()
        # ---- 以下按 golden 的 B7/B8/B11/B12/B10/B13/B9 顺序组装 ----
        gam2 = d["Gamc"].float()
        Ec = d["Ec"]
        r1 = k_rows_v(k_mulv(d["Vc"], t1), ones_v)[..., 0]        # B7
        r2 = k_rows_k(k_mul(d["Kc"], t2n), ones_k)[..., 0]        # B8（t2n 自带负号）
        # B11: dA -> dL -> dk 与 dbeta
        da = k_tb_cv(dvnew, d["bv"]).float() - k_tb_ck(dw, d["bek_n"]).float()
        t_ = k_ta_cc(d["Ac"], da.to(dt))
        o_ = k_tb_cc(t_, d["Ac"]).float()
        dl = msk_j[None, None] * (-o_)                             # [B,H,C,C] fp32
        dx = dl * d["Betac"][:, :, :, None] * gam2
        dxs = (dx + dx.transpose(-1, -2)).to(dt)
        _q4 = k_ab_k(dxs, d["Kc"])
        if _DBG:
            print(f"RELAYQ c={c} q4={_q4.abs().max():.4e}", flush=True)
        _dk_full = (_dk_acc + _q4.float()).to(dt)
        if _oneshot:
            dk_buf.append(_dk_full)             # 收集，循环后一次 cat 写回
        else:
            dk[:, :, sl, :] = _dk_full          # 逐 chunk 切片写
        if dk_parts is not None:
            dk_parts["q1"][:, :, sl, :] = _dk1
            dk_parts["q2"][:, :, sl, :] = _q2
            dk_parts["q3"][:, :, sl, :] = _q3
            dk_parts["q4"][:, :, sl, :] = _q4
        kk = k_rawk(d["Kc"], d["Kc"]).float()                      # KKt
        r3 = (dl * gam2 * kk).sum(-1)                              # B11 的 dbeta 项
        if _DBG:
            print(f"RELAYDBG c={c} |r3|={r3.abs().max():.3e}", flush=True)
        dbeta[:, :, sl] = r1 - Ec * r2 + r3                        # B7 + B8 + B11
        # B3+B11 合成 dgam2，B12 统一乘 Gam 做 rowsum - colsum
        dgam2 = d["dP"].float() * m_m * d["raw"].float() + dl * d["Betac"][:, :, :, None] * kk
        w12 = dgam2 * gam2
        dgam = w12.sum(-1) - w12.sum(-2)                           # B12
        # B10：只保留 -dth*th。`carry`（跨 chunk 归约）与 golden 的
        # `dgam[m-1] += sum(dth*th)` 补项是**同一恒等式的两种写法，只能取其一**：
        # 补上会双计，导致 dg 错 63%（fp64 复刻：只加 carry rel=6.5e-15、只加 m-1
        # 两项 rel=3.6e-15、两者皆无或皆有 rel=0.634；两项实测量级 1.41）。
        # 因此这里不加 m-1 补项。
        th = d["th"].float()
        dgam = dgam - dth * th
        # B13: dgam += E .* dE
        qh = k_tb_qh(d["Qc"], d["Hc"])
        de1 = (d["DOc"].float() * qh.float()).sum(-1)
        de2 = d["Betac"] * (-r2)
        de = de1 + de2
        if _DBG:
            print(f"RELAYDE c={c} de1m={de1.mean():+.4e} de2m={de2.mean():+.4e} "
                  f"de_em={(Ec * de).sum().item():+.4e}", flush=True)
        dgam = dgam + Ec * de
        if _DBG:
            print(f"RELAYDBG c={c} |da|={da.abs().max():.3e} |dl|={dl.abs().max():.3e} "
                  f"|kk|={kk.abs().max():.3e} |de|={de.abs().max():.3e} "
                  f"|dgam|={dgam.abs().max():.3e} carry={carry.abs().max():.3e} "
                  f"|r1|={r1.abs().max():.3e} |r2|={r2.abs().max():.3e}", flush=True)
        dg[:, :, sl] = k_suffix(dgam, carry)
        carry = carry + dgam.sum(-1)
        d["delta_f32"] = delta_f32
        if c == 0:
            dh0 = delta_f32.to(dt)
    if _oneshot:
        dk = torch.cat(dk_buf, dim=2)          # [B,H,T,DK]，一次 aclnn
    torch.npu.synchronize()
    if dk_parts is not None:
        return dq, dk, dv, dbeta, dg, dh0, dk_parts
    return dq, dk, dv, dbeta, dg, dh0

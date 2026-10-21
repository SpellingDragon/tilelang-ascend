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


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
def gdn_v3_dk(B, H, C, DK, dtype="float16", accum="float"):
    """dk = dqkt^T Q"""

    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), Q: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             O: T.Tensor([B, H, C, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)   # A (transposed)
            q_l1 = T.alloc_L1([C, DK], dtype)  # B
            o_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(DQKT[b, h, 0, 0], a_l1)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.gemm_v0(a_l1, q_l1, o_l0, transpose_A=True, init=True)
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


@tilelang.jit(out_idx=[2], pass_configs=_PASS)
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
            u32 = T.alloc_ub([C, DV], accum)
            w16 = T.alloc_ub([C, DV], dtype)
            o16 = T.alloc_ub([C, DV], dtype)
            T.copy(U0[b, h, 0, 0], u32)
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


@tilelang.jit(out_idx=[5, 6], pass_configs=_PASS)
def gdn_v3_gate_edo(B, H, C, DV, dtype="float16", accum="float"):
    """dqkt = dP .* Gam .* M ;  e_do = E .* dO"""

    @T.prim_func
    def main(DP: T.Tensor([B, H, C, C], dtype), Gam: T.Tensor([B, H, C, C], "float"),  # type: ignore
             Msk: T.Tensor([B, H, C, C], "float"), E: T.Tensor([B, H, C], "float"),  # type: ignore
             DO: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             DQKT: T.Tensor([B, H, C, C], dtype), EDO: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
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
            T.copy(d16, DQKT[b, h, 0, 0])
            T.copy(o16, o32)
            for i, j in T.Parallel(C, DV):
                o32[i, j] = o32[i, j] * e32[i]
            T.copy(o32, o16)
            T.copy(o16, EDO[b, h, 0, 0])

    return main



# ---------------------------- relay：反向扫描（P4） ----------------------------
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

# ---------------------------- host：chunk 循环（D23） ----------------------------
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
    msk_m = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))

    ws_g, ws_gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype=dtype)(
        k, beta, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, 64, 64, dtype=dtype)(
        k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()

    k_ab = gdn_v3_ab(B, H, C, DV)
    k_ab_k = gdn_v3_ab(B, H, C, DK)
    k_wh = gdn_v3_wh(B, H, C, DV, DK)
    k_raw = gdn_v3_raw(B, H, C, DK)
    k_dp = gdn_v3_dp(B, H, C, DV)
    k_dq = gdn_v3_dq(B, H, C, DK, DV)
    k_dk = gdn_v3_dk(B, H, C, DK)
    k_dv = gdn_v3_dv(B, H, C, DV)
    k_scale = gdn_v3_scale(B, H, C, DV)
    k_scale_k = gdn_v3_scale(B, H, C, DK)
    k_sbe = gdn_v3_scale_be(B, H, C, DK)
    k_sub = gdn_v3_sub(B, H, C, DV)
    k_gate = gdn_v3_gate(B, H, C)
    k_ge = gdn_v3_gate_edo(B, H, C, DV)

    dq = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dk = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dv = torch.empty(B, H, T_len, DV, dtype=dt, device=dev)

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

        bv = k_scale(Vc, Betac)                      # beta .* V
        u0 = k_ab(Ac, bv)                            # U0 = A beta V
        bek = k_sbe(Kc, Betac, Ec)                   # beta .* E .* K
        wm = k_ab_k(Ac, bek)                         # W = A bek
        wh = k_wh(wm, Hc)                            # W H^T
        vnew = k_sub(u0, wh)                         # Vnew

        # raw/P 只在 relay 的 `P^T dO` 里需要（B1）；当前的 dq 链用不到，
        # 先跳过以免每次 launch 白算一次 [C,C] 的 gate。
        dP = k_dp(DOc, vnew)
        dqkt, edo = k_ge(dP, Gamc, msk_m, Ec, DOc)
        dq[:, :, sl, :] = k_dq(dqkt, Kc, edo, Hc)
        dk[:, :, sl, :] = k_dk(dqkt, Qc)
        dv[:, :, sl, :] = k_scale(k_dv(Ac, vnew), Betac)
    torch.npu.synchronize()
    return dq, dk, dv


def gdn_bwd_v3_relay(q, k, v, g, beta, h0, do, chunk_size=64):
    """完整反向：先正向缓存 vnew，再反向扫 Delta'，输出 (dq, dk, dv, dh0)。

    依据 `_np_chunk_bwd` 的 B1-B10（见 notes [91]）：
        dvnew_c = P_c^T dO_c + Kt_c (Delta'_{c+1})^T
        dw_c    = -dvnew_c H_in_c
        Delta'_c = e_do_c^T Q_c - dvnew_c^T W_c + glast_c * Delta'_{c+1}
        dk_c    = dqkt_c^T Q_c + (beta E) .* t2_c + th_c .* (vnew_c Delta'_{c+1})
        dv_c    = beta_c .* (A_c^T dvnew_c)
    `dq` 与 relay 无关（`dqkt K + e_do H_in`），直接沿用第一遍。
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
    NC = (T_len + C - 1) // C
    dt = q.dtype
    dev = q.device
    dtype = "float16" if dt == torch.float16 else "float32"

    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
    idt = torch.eye(C, dtype=torch.float32, device=dev)
    msk_m = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
    ws_g, ws_gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype=dtype)(
        k, beta, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, 64, 64, dtype=dtype)(
        k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()

    k_ab = gdn_v3_ab(B, H, C, DV)
    k_ab_k = gdn_v3_ab(B, H, C, DK)
    k_wh = gdn_v3_wh(B, H, C, DV, DK)
    k_raw = gdn_v3_raw(B, H, C, DK)
    k_dp = gdn_v3_dp(B, H, C, DV)
    k_dq = gdn_v3_dq(B, H, C, DK, DV)
    k_dk = gdn_v3_dk(B, H, C, DK)
    k_scale = gdn_v3_scale(B, H, C, DV)
    k_sbe = gdn_v3_scale_be(B, H, C, DK)
    k_sub = gdn_v3_sub(B, H, C, DV)
    k_gate = gdn_v3_gate(B, H, C)
    k_ge = gdn_v3_gate_edo(B, H, C, DV)
    k_ta = gdn_v3_ta(B, H, C, C, DV)      # P^T dO
    k_ta_dvn = gdn_v3_ta(B, H, C, DV, DK)  # dvnew^T W  /  e_do^T Q
    k_tb_kt = gdn_v3_tb(B, H, C, DV, DK)   # Kt Delta'^T
    k_tb_vn = gdn_v3_tb(B, H, C, DV, DK)   # vnew Delta'

    dq = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dk = torch.empty(B, H, T_len, DK, dtype=dt, device=dev)
    dv = torch.empty(B, H, T_len, DV, dtype=dt, device=dev)
    cache = []

    # ---- pass 1: 正向，存 vnew / W / Kt / th / glast ----
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        Qc, Kc, Vc, DOc = q[:, :, sl, :], k[:, :, sl, :], v[:, :, sl, :], do[:, :, sl, :]
        Ac, Gc = ws_a[:, :, sl, :], ws_g[:, :, sl]
        Ec = torch.exp(Gc)
        Hc, Betac, Gamc = ws_h[:, :, c, :, :], beta[:, :, sl], ws_gam[:, :, c, :, :]
        bv = k_scale(Vc, Betac)
        u0 = k_ab(Ac, bv)
        bek = k_sbe(Kc, Betac, Ec)
        wm = k_ab_k(Ac, bek)
        wh = k_wh(wm, Hc)
        vnew = k_sub(u0, wh)
        raw = k_raw(Qc, Kc)
        _P = k_gate(raw, Gamc, msk_m)
        dP = k_dp(DOc, vnew)
        dqkt, edo = k_ge(dP, Gamc, msk_m, Ec, DOc)
        dq[:, :, sl, :] = k_dq(dqkt, Kc, edo, Hc)
        cache.append(dict(sl=sl, Qc=Qc, Kc=Kc, DOc=DOc, Ac=Ac, Ec=Ec, Hc=Hc,
                          Betac=Betac, vnew=vnew, wm=wm, P=_P, dqkt=dqkt, edo=edo,
                          glast=float(torch.exp(Gc[:, :, -1]).mean().item()),
                          th=torch.exp(Gc[:, :, -1:].expand(-1, -1, C) - Gc)))
    torch.npu.synchronize()

    # ---- pass 2: 反向扫 Delta' ----
    delta = None
    dh0 = None
    for c in range(NC - 1, -1, -1):
        d = cache[c]
        sl = d["sl"]
        # dvnew = P^T dO + Kt Delta'^T
        base = k_ta(d["P"], d["DOc"])
        if delta is None:
            dvnew = base
        else:
            dvnew = base + k_tb_kt(d["Kc"], delta)   # Kt [C,DK] x Delta' [DV,DK]^T
        # Delta'_c = e_do^T Q - dvnew^T W + glast * Delta'
        tq = k_ta_dvn(d["edo"], d["Qc"])
        tw = k_ta_dvn(dvnew, d["wm"])
        nxt = tq.float() - tw.float()
        if delta is not None:
            nxt = nxt + d["glast"] * delta.float()
        delta = nxt.to(dt)
        # dk += th .* (vnew Delta')
        if c < NC - 1:
            dkt = k_tb_vn(d["vnew"], cache[c + 1]["delta_next"])
            dk[:, :, sl, :] = k_dk(d["dqkt"], d["Qc"]) + (d["th"] * dkt.float()).to(dt)
        else:
            dk[:, :, sl, :] = k_dk(d["dqkt"], d["Qc"])
        # dv = beta .* (A^T dvnew)
        dv[:, :, sl, :] = k_scale(k_ta(d["Ac"], dvnew), d["Betac"])
        d["delta_next"] = delta
        if c == 0:
            dh0 = delta
    torch.npu.synchronize()
    return dq, dk, dv, dh0

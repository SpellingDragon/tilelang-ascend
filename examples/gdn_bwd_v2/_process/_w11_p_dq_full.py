# _w11_p_dq_full.py — v3 的 dq 全 kernel 版：5 段、每段单侧、host 只做循环
#
#   K1 (cube) raw   = Q K^T
#   K2 (vec)  P     = raw .* Gam .* M
#   K3 (cube) dP    = dO Vnew^T
#   K4 (vec)  dqkt  = dP .* Gam .* M ;  e_do = E .* dO
#   K5 (cube) dq    = dqkt K + e_do H_in
# 判据：逐 chunk dq rel < 5e-3（golden adjoint）。
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


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_raw(B, H, C, DK, dtype="float16", accum="float"):
    @T.prim_func
    def main(Q: T.Tensor([B, H, C, DK], dtype), K: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             R: T.Tensor([B, H, C, C], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            q_l1 = T.alloc_L1([C, DK], dtype)
            k_l1 = T.alloc_L1([C, DK], dtype)
            o_l0 = T.alloc_L0C([C, C], accum)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.copy(K[b, h, 0, 0], k_l1)
            T.gemm_v0(q_l1, k_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, R[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[3], pass_configs=_pass)
def k_gate(B, H, C, dtype="float16", accum="float"):
    """vec: O = X .* Gam .* M"""

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


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_dp(B, H, C, DV, dtype="float16", accum="float"):
    @T.prim_func
    def main(DO: T.Tensor([B, H, C, DV], dtype), VN: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             DP: T.Tensor([B, H, C, C], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            do_l1 = T.alloc_L1([C, DV], dtype)
            vn_l1 = T.alloc_L1([C, DV], dtype)
            o_l0 = T.alloc_L0C([C, C], accum)
            T.copy(DO[b, h, 0, 0], do_l1)
            T.copy(VN[b, h, 0, 0], vn_l1)
            T.gemm_v0(do_l1, vn_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, DP[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[5, 6], pass_configs=_pass)
def k_gate_edo(B, H, C, DV, dtype="float16", accum="float"):
    """vec: dqkt = dP .* Gam .* M ; e_do = E .* dO"""

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


@tilelang.jit(out_idx=[4], pass_configs=_pass)
def k_dq(B, H, C, DK, DV, dtype="float16", accum="float"):
    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), K: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             EDO: T.Tensor([B, H, C, DV], dtype), HH: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
             DQ: T.Tensor([B, H, C, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            dqkt_l1 = T.alloc_L1([C, C], dtype)
            k_l1 = T.alloc_L1([C, DK], dtype)
            edo_l1 = T.alloc_L1([C, DV], dtype)
            hin_l1 = T.alloc_L1([DV, DK], dtype)
            o_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(DQKT[b, h, 0, 0], dqkt_l1)
            T.copy(K[b, h, 0, 0], k_l1)
            T.copy(EDO[b, h, 0, 0], edo_l1)
            T.copy(HH[b, h, 0, 0], hin_l1)
            T.gemm_v0(dqkt_l1, k_l1, o_l0, init=True)
            T.gemm_v0(edo_l1, hin_l1, o_l0, init=False)
            T.copy(o_l0, DQ[b, h, 0, 0])

    return main


def main():
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd, _golden_item

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    NC = T_len // C
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn, DOn = (beta.double().numpy()[0, 0], g.double().numpy()[0, 0],
                       h0.double().numpy()[0, 0], do.double().numpy()[0, 0])
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)
    dq_g = np.asarray(_golden_item(gold, "dq", 0), np.float64)[0, 0]

    def f16(x):
        return torch.from_numpy(np.asarray(x, np.float64).astype(np.float16))[None, None].to("npu")

    def f32(x):
        return torch.from_numpy(np.asarray(x, np.float64).astype(np.float32))[None, None].to("npu")

    k1, k2, k3, k4, k5 = (k_raw(B, H, C, DK), k_gate(B, H, C), k_dp(B, H, C, DV),
                          k_gate_edo(B, H, C, DV), k_dq(B, H, C, DK, DV))
    ok = True
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        ca = cache[c]
        raw = k1(f16(Qn[sl]), f16(Kn[sl]))
        P = k2(raw, f32(np.asarray(ca["gam_2d"])), f32(np.asarray(ca["msk_m"])))
        dp = k3(f16(DOn[sl]), f16(np.asarray(ca["vnew"])))
        dqkt, edo = k4(dp, f32(np.asarray(ca["gam_2d"])), f32(np.asarray(ca["msk_m"])),
                       f32(np.asarray(ca["e_vec"])), f16(DOn[sl]))
        out = k5(dqkt, f16(Kn[sl]), edo, f16(np.asarray(ca["hin"]))).float().cpu().numpy()[0, 0].astype(np.float64)
        rel = float(np.abs(out - dq_g[sl]).max() / (np.abs(dq_g[sl]).max() + 1e-30))
        print(f"DQF cc={c} rel={rel:.3e} goldmax={float(np.abs(dq_g[sl]).max()):.3e}")
        ok &= rel < 5e-3
    print("DQF verdict=%s" % ("PASS" if ok else "FAIL"))
    print("DQF-DONE")


if __name__ == "__main__":
    main()

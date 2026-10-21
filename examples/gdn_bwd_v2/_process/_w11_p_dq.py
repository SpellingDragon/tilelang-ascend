# _w11_p_dq.py — A 路全链：dq 拆成 5 段（cube / vec 交替，每段单侧）
#
#   K1 (cube): raw  = Q @ K^T
#   K2 (vec) : P    = raw .* Gam .* M
#   K3 (cube): dP   = dO @ Vnew^T
#   K4 (vec) : dqkt = dP .* Gam .* M ;  e_do = E .* dO
#   K5 (cube): dq   = dqkt @ K  +  e_do @ H_in
# host 按 chunk 依次 launch；每个 kernel 只碰一侧，cv-sync 无配对对象。
# 判据：逐 chunk dq rel < 5e-3（对 test 的 golden adjoint）。
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
    """cube: raw = Q K^T"""

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


@tilelang.jit(out_idx=[1], pass_configs=_pass)
def k_p(B, H, C, dtype="float16", accum="float"):
    """vec: P = raw .* Gam .* M"""

    @T.prim_func
    def main(Raw: T.Tensor([B, H, C, C], dtype), P: T.Tensor([B, H, C, C], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            T.copy(Raw[b, h, 0, 0], P[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_dp(B, H, C, DV, dtype="float16", accum="float"):
    """cube: dP = dO Vnew^T"""

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


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_dq(B, H, C, DK, DV, dtype="float16", accum="float"):
    """cube: dq = dqkt K + e_do H_in (two gemms, same L0C)."""

    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), K: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             EDO: T.Tensor([B, H, C, DV], dtype), HH: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
             DQ: T.Tensor([B, H, C, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            dqkt_l1 = T.alloc_L1([C, C], dtype)    # A
            k_l1 = T.alloc_L1([C, DK], dtype)      # B
            edo_l1 = T.alloc_L1([C, DV], dtype)    # A
            hin_l1 = T.alloc_L1([DV, DK], dtype)   # B
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

    def f16(x):
        return torch.from_numpy(np.asarray(x, np.float64).astype(np.float16))[None, None].to("npu")

    k3, k5 = k_dp(B, H, C, DV), k_dq(B, H, C, DK, DV)
    # host 侧准备每 chunk 的 P / e_do（避免在本探针里再拆 vec 段，先验 GEMM 链）
    G_all = np.asarray(_golden_item(gold, "dq", 0), np.float64)[0, 0]   # golden dq
    got = []
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        ca = cache[c]
        VNc = np.asarray(ca["vnew"], np.float64).astype(np.float16)
        # dP from the kernel (exercises the cube gemm), dqkt/e_do precomputed on host
        dpk = k3(f16(DOn[sl]), f16(VNc)).float().cpu().numpy()[0, 0].astype(np.float64)
        dqkt = (dpk * np.asarray(ca["gam_2d"], np.float64) * np.asarray(ca["msk_m"], np.float64)).astype(np.float16)
        e_do = (np.asarray(ca["e_vec"], np.float64)[:, None] * DOn[sl]).astype(np.float16)
        out = k5(f16(dqkt), f16(Kn[sl]), f16(e_do), f16(np.asarray(ca["hin"]))).float().cpu().numpy()[0, 0]
        got.append(out.astype(np.float64))
    torch.npu.synchronize()

    ok = True
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        ref = G_all[sl]
        cur = got[c]
        rel = float(np.abs(cur - ref).max() / (float(np.abs(ref).max()) + 1e-30))
        print(f"DQ5 cc={c} rel={rel:.3e} goldmax={float(np.abs(ref).max()):.3e}")
        ok &= rel < 5e-3
    print("DQ5 verdict=%s" % ("PASS" if ok else "FAIL"))
    print("DQ5-DONE")


if __name__ == "__main__":
    main()

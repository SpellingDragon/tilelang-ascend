# _w11_v3_dqdkdv.py — v3 三梯度（dq/dk/dv）的完整链：复用已验证积木，每 kernel 单 GEMM
#
# 复用 _w11_p_dq_full 的 k_raw / k_gate / k_dp / k_gate_edo / k_dq（已 PASS），
# 新增 k_dk（dqkt^T Q）与 k_dv（A^T dVnew）。每 chunk 共 7 次 launch。
# 判据：
#   dq 对 golden（其两项即完整梯度）
#   dk/dv 对同一单项的 numpy（golden 含跨 chunk / th 项）
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

from _w11_p_dq_full import k_raw, k_gate, k_dp, k_gate_edo, k_dq  # noqa: E402

torch.manual_seed(7)

_pass = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_dk(B, H, C, DK, dtype="float16", accum="float"):
    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), Q: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             O: T.Tensor([B, H, C, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)
            q_l1 = T.alloc_L1([C, DK], dtype)
            o_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(DQKT[b, h, 0, 0], a_l1)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.gemm_v0(a_l1, q_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_dv(B, H, C, DV, dtype="float16", accum="float"):
    @T.prim_func
    def main(A: T.Tensor([B, H, C, C], dtype), DVN: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             O: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)
            v_l1 = T.alloc_L1([C, DV], dtype)
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(DVN[b, h, 0, 0], v_l1)
            T.gemm_v0(a_l1, v_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

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
    k6, k7 = k_dk(B, H, C, DK), k_dv(B, H, C, DV)
    ok = True
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        ca = cache[c]
        raw = k1(f16(Qn[sl]), f16(Kn[sl]))
        _P = k2(raw, f32(np.asarray(ca["gam_2d"])), f32(np.asarray(ca["msk_m"])))
        dp = k3(f16(DOn[sl]), f16(np.asarray(ca["vnew"])))
        dqkt, edo = k4(dp, f32(np.asarray(ca["gam_2d"])), f32(np.asarray(ca["msk_m"])),
                       f32(np.asarray(ca["e_vec"])), f16(DOn[sl]))
        out_dq = k5(dqkt, f16(Kn[sl]), edo, f16(np.asarray(ca["hin"]))).float().cpu().numpy()[0, 0].astype(np.float64)
        out_dk = k6(dqkt, f16(Qn[sl])).float().cpu().numpy()[0, 0].astype(np.float64)
        t_dv = k7(f16(np.asarray(ca["a_mat"])), f16(np.asarray(ca["vnew"]))).float().cpu().numpy()[0, 0].astype(np.float64)

        rel_dq = float(np.abs(out_dq - dq_g[sl]).max() / (np.abs(dq_g[sl]).max() + 1e-30))
        dk_ref = dqkt.float().cpu().numpy()[0, 0].astype(np.float64).T @ Qn[sl]
        rel_dk = float(np.abs(out_dk - dk_ref).max() / (np.abs(dk_ref).max() + 1e-30))
        dv_ref = Bn[sl][:, None] * (np.asarray(ca["a_mat"], np.float64).T @ np.asarray(ca["vnew"], np.float64))
        out_dv = Bn[sl][:, None] * t_dv
        rel_dv = float(np.abs(out_dv - dv_ref).max() / (np.abs(dv_ref).max() + 1e-30))
        print(f"V3G cc={c} dq={rel_dq:.3e} dk={rel_dk:.3e} dv={rel_dv:.3e}")
        ok &= (rel_dq < 5e-3 and rel_dk < 5e-3 and rel_dv < 5e-3)
    print("V3G verdict=%s" % ("PASS" if ok else "FAIL"))
    print("V3G-DONE")


if __name__ == "__main__":
    main()

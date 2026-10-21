# _w11_v3_split.py — A 路合流：Vnew 拆成"纯 cube + 纯 vec"两个 kernel
#
#   K1 (cube): WH = W @ H^T        —— 只有 L1/L0C，L0C 直出 GM
#   K2 (vec) : VN = U0 - WH        —— 只有 UB 运算
# host 侧按 chunk 各 launch 一次（每 launch 单侧 ⇒ cv-sync 无配对对象）。
# 判据：逐 chunk rel < 5e-3 ⇒ 拆 kernel 路线成立，可以按同样办法拆 P/dq。
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
def k_wh(B, H, C, DV, DK, dtype="float16", accum="float"):
    """Pure cube: WH = W @ H^T."""

    @T.prim_func
    def main(
        W: T.Tensor([B, H, C, DK], dtype),    # type: ignore
        Hh: T.Tensor([B, H, DV, DK], dtype),  # type: ignore
        WH: T.Tensor([B, H, C, DV], dtype),   # type: ignore
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            w_l1 = T.alloc_L1([C, DK], dtype)     # A operand only
            hin_l1 = T.alloc_L1([DV, DK], dtype)  # B operand only
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(W[b, h, 0, 0], w_l1)
            T.copy(Hh[b, h, 0, 0], hin_l1)
            T.gemm_v0(w_l1, hin_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, WH[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_sub(B, H, C, DV, dtype="float16", accum="float"):
    """Pure vector: VN = U0 - WH."""

    @T.prim_func
    def main(
        U0: T.Tensor([B, H, C, DV], dtype),  # type: ignore
        WH: T.Tensor([B, H, C, DV], dtype),  # type: ignore
        VN: T.Tensor([B, H, C, DV], dtype),  # type: ignore
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            u16 = T.alloc_ub([C, DV], dtype)
            w16 = T.alloc_ub([C, DV], dtype)
            o16 = T.alloc_ub([C, DV], dtype)
            u32 = T.alloc_ub([C, DV], accum)
            T.copy(U0[b, h, 0, 0], u16)
            T.copy(WH[b, h, 0, 0], w16)
            T.copy(u16, u32)
            for i, j in T.Parallel(C, DV):
                o16[i, j] = T.cast(u32[i, j] - T.cast(w16[i, j], accum), dtype)
            T.copy(o16, VN[b, h, 0, 0])

    return main


def main():
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    NC = T_len // C
    (q, k, v, g, beta, h0, do), _ = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn = beta.double().numpy()[0, 0], g.double().numpy()[0, 0], h0.double().numpy()[0, 0]
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)

    u0 = np.stack([np.asarray(cache[c]["u0"], np.float64) for c in range(NC)]).astype(np.float16)
    wm = np.stack([np.asarray(cache[c]["w_mat"], np.float64) for c in range(NC)]).astype(np.float16)
    hin = np.stack([np.asarray(cache[c]["hin"], np.float64) for c in range(NC)]).astype(np.float16)
    t_u0 = torch.from_numpy(u0)[None, None].to("npu")
    t_w = torch.from_numpy(wm)[None, None].to("npu")
    t_h = torch.from_numpy(hin)[None, None].to("npu")

    k1 = k_wh(B, H, C, DV, DK)
    k2 = k_sub(B, H, C, DV)
    got = []
    with torch.no_grad():
        for c in range(NC):
            wh = k1(t_w[:, :, c], t_h[:, :, c])
            got.append(k2(t_u0[:, :, c], wh).float().cpu().numpy()[0, 0])
    torch.npu.synchronize()

    ok = True
    for c in range(NC):
        ref = np.asarray(cache[c]["vnew"], np.float64)
        cur = np.asarray(got[c], np.float64)
        rel = float(np.abs(cur - ref).max() / (float(np.abs(ref).max()) + 1e-30))
        print(f"SPLIT cc={c} rel={rel:.3e}")
        ok &= rel < 5e-3
    print("SPLIT verdict=%s" % ("PASS" if ok else "FAIL"))
    print("SPLIT-DONE")


if __name__ == "__main__":
    main()

# _w11_v3_vnew.py — v3 骨架第一枪：块内 chunk 循环 + 每 chunk 的 Vnew = U0 + W·H^T
#
# 刻意只保留 GEMM 部分（U0 与 W 由 host 用 fp64 算好传入），这样本步只验证
# v3 骨架的三件事：
#   1) 单 kernel、块内 `for c in T.serial(chunk_num)` 的循环正确
#   2) 两条 L1 各司其职：w_l1 只作 A（转置 B 的 GEMM），hin_l1 只作 B
#   3) L0C 先 `init=True` 装 U0、再 `init=False` 累加 W·H^T，最后一次性出 GM
# 判读：VNEW3 的逐 chunk rel < 5e-3 ⇒ v3 骨架成立，可继续挂 vector 段（βV/βEK）。
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


@tilelang.jit(out_idx=[3], pass_configs=_pass)
def v3_vnew(B, H, NC, C, DV, DK, dtype="float16", accum="float"):
    """NC = chunk_num; inputs are per-chunk U0 [B,H,NC,C,DV] and W [B,H,NC,C,DK]."""

    @T.prim_func
    def main(
        U0: T.Tensor([B, H, NC, C, DV], dtype),   # type: ignore
        Wm: T.Tensor([B, H, NC, C, DK], dtype),   # type: ignore
        Hh: T.Tensor([B, H, NC, DV, DK], dtype),  # type: ignore  H_in per chunk
        VN: T.Tensor([B, H, NC, C, DV], dtype),   # type: ignore
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            w_l1 = T.alloc_L1([C, DK], dtype)      # A operand only
            hin_l1 = T.alloc_L1([DV, DK], dtype)   # B operand only
            vn_l0 = T.alloc_L0C([C, DV], accum)
            u0_16 = T.alloc_ub([C, DV], dtype)
            u0_32 = T.alloc_ub([C, DV], accum)
            wh_ub = T.alloc_ub([C, DV], accum)
            io_ub = T.alloc_ub([C, DV], dtype)

            # python range: every c is a compile-time constant, so the
            # per-chunk loads cannot be hoisted out of the loop (cc=3 alone was
            # exact with T.serial, notes [80]).
            for c in range(NC):
                T.copy(Wm[b, h, c, 0, 0], w_l1)
                T.copy(Hh[b, h, c, 0, 0], hin_l1)
                T.copy(U0[b, h, c, 0, 0], u0_16)
                T.copy(u0_16, u0_32)
                # W @ H^T in L0C, then Vnew = U0 - that, on the vector side
                T.gemm_v0(w_l1, hin_l1, vn_l0, transpose_B=True, init=True)
                T.copy(vn_l0, wh_ub)
                for i, j in T.Parallel(C, DV):
                    wh_ub[i, j] = u0_32[i, j] - wh_ub[i, j]
                T.copy(wh_ub, io_ub)
                T.copy(io_ub, VN[b, h, c, 0, 0])

    return main


def main():
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    NC = T_len // C
    (q, k, v, g, beta, h0, do), _ = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    dt = torch.float16

    Qn = q.double().numpy()[0, 0]
    Kn, Vn = k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn = beta.double().numpy()[0, 0], g.double().numpy()[0, 0]
    Hn = h0.double().numpy()[0, 0]
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)

    u0 = np.stack([np.asarray(cache[c]["u0"], np.float64) for c in range(NC)])          # [NC,C,DV]
    wm = np.stack([np.asarray(cache[c]["w_mat"], np.float64) for c in range(NC)])       # [NC,C,DK]
    hin = np.stack([np.asarray(cache[c]["hin"], np.float64) for c in range(NC)])        # [NC,DV,DK]

    t_u0 = torch.from_numpy(u0.astype(np.float16))[None, None].to("npu")
    t_w = torch.from_numpy(wm.astype(np.float16))[None, None].to("npu")
    t_h = torch.from_numpy(hin.astype(np.float16))[None, None].to("npu")

    got = v3_vnew(B, H, NC, C, DV, DK)(t_u0, t_w, t_h).float().cpu().numpy()[0, 0]
    torch.npu.synchronize()

    ok = True
    for c in range(NC):
        ref = np.asarray(cache[c]["vnew"], np.float64)
        # vnew = u0 - W H^T  (cache's w_mat is +W, the kernel gets -W)
        cur = np.asarray(got[c], np.float64)
        d = float(np.abs(cur - ref).max())
        rel = d / (float(np.abs(ref).max()) + 1e-30)
        print(f"VNEW3 cc={c} maxdiff={d:.3e} goldmax={float(np.abs(ref).max()):.3e} rel={rel:.3e}")
        ok &= rel < 5e-3
    print("VNEW3 verdict=%s" % ("PASS" if ok else "FAIL"))
    print("VNEW3-DONE")


if __name__ == "__main__":
    main()

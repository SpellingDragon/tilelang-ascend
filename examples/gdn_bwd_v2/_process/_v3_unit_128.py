# _v3_unit_128.py -- v3 kernel 的**核级**单元对拍：喂 golden 的 A / H_in，逐个比中间量
#
# 为什么需要：端到端对比只能告诉我们"128 坏了"，不能告诉我们是哪个 kernel。
# 本支把 v2 前置件整个绕开：`A` 与 `H_in` 直接用 golden 的 `a_mat` / `hin`，
# 然后按 relay 的调用顺序逐个跑 v3 kernel，与 golden cache 里的同名中间量比。
#
# 每个 kernel 都跑两遍：DK=DV=64 与 128，直接看哪个 kernel 在 128 上掉精度。
#
# 覆盖 dq/dv/dbeta 链路上的：raw / gate(P) / scale_k(Kt、beta*E*K) / ab_k(-W) /
# vnew / dp / ta(A^T dvnew) / tb_qh(q H^T)。不含 dk 的跨 chunk 项。
#
# 打印：UNIT dim=64|128 name=.. rel=.. max_abs=..
import numpy as np
import torch
import torch_npu  # noqa: F401

from example_gdn_bwd import _np_chunk_fwd
from gdn_bwd_v3 import (
    gdn_v3_ab, gdn_v3_dp, gdn_v3_gate, gdn_v3_raw, gdn_v3_scale, gdn_v3_ta,
    gdn_v3_tb, gdn_v3_vnew,
)

ITEMS = ("raw", "P", "kt", "bek", "negW", "vnew", "dP", "t1", "qh")


def _rel(got, ref):
    a = got.float().cpu().numpy().astype(np.float64)
    b = np.asarray(ref, np.float64)
    return float(np.abs(a - b).max()) / (float(np.abs(b).max()) + 1e-30)


def one(dim, T_len=256, B=1, H=1, C=64):
    DK = DV = dim
    torch.manual_seed(7)
    dt = torch.float16
    rq = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    q32 = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rk = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k32 = rk / rk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    g32 = -torch.randn(B, H, T_len).abs() * 0.5 - 0.05
    beta32 = torch.sigmoid(torch.randn(B, H, T_len))
    h032 = torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1
    do32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    cpu = [q32.to(dt), k32.to(dt), v32.to(dt), g32, beta32, h032.to(dt), do32.to(dt)]
    q, k, v, g, beta, h0, do = [x.npu() for x in cpu]
    npc = [x.numpy() for x in cpu]

    # golden cache（单 head）：给出 a_mat / hin / 各样中间量
    qn, kn, vn, gn, betan, h0n, don = npc
    _o, cache = _np_chunk_fwd(T_len, DK, DV, qn[0, 0].astype(np.float64),
                              kn[0, 0].astype(np.float64), vn[0, 0].astype(np.float64),
                              betan[0, 0].astype(np.float64), gn[0, 0].astype(np.float64),
                              h0n[0, 0].astype(np.float64), C)

    msk = torch.tril(torch.ones(C, C, dtype=torch.float32, device="npu"))
    sl = slice(0, C)
    Qc, Kc, Vc, DOc = q[:, :, sl, :], k[:, :, sl, :], v[:, :, sl, :], do[:, :, sl, :]
    c0 = cache[0]
    # 直接用 golden 的 A 与 H_in（[C,C] / [DV,DK]），绕开 solve_a / scan1
    Ac = torch.from_numpy(np.asarray(c0["a_mat"], np.float64).astype(np.float16))[None, None].npu()
    Hc = torch.from_numpy(np.asarray(c0["hin"], np.float64).astype(np.float16))[None, None].npu()
    # gate 的第二个参数是 [C,C] 的 Gam 矩阵（不是向量）—— 用 cache 里的 gam_2d
    Gamc = torch.from_numpy(np.ascontiguousarray(c0["gam_2d"])).float()[None, None].npu()

    res = {}
    raw = gdn_v3_raw(B, H, C, DK)(Qc, Kc)
    res["raw"] = (raw, c0["q"] @ c0["k"].T)
    P = gdn_v3_gate(B, H, C)(raw, Gamc, msk)
    res["P"] = (P, c0["p_mat"])
    th = torch.from_numpy(np.exp(c0["gam"][-1] - c0["gam"]).astype(np.float32))[None, None].npu()
    kt = gdn_v3_scale(B, H, C, DK)(Kc, th)
    res["kt"] = (kt, c0["kt"])
    bee = torch.from_numpy((-(c0["beta"] * c0["e_vec"])).astype(np.float32))[None, None].npu()
    bek = gdn_v3_scale(B, H, C, DK)(Kc, bee)
    res["bek"] = (bek, -(c0["beta"] * c0["e_vec"])[:, None] * c0["k"])
    wm = gdn_v3_ab(B, H, C, DK)(Ac, bek)
    res["negW"] = (wm, -c0["w_mat"])
    bv = gdn_v3_scale(B, H, C, DV)(Vc, torch.from_numpy(c0["beta"].astype(np.float32))[None, None].npu())
    vnew = gdn_v3_vnew(B, H, C, DV, DK)(Ac, bv, wm, Hc)
    res["vnew"] = (vnew, c0["vnew"])
    v_hi = vnew.to(dt)
    v_lo = (vnew - v_hi.float()).to(dt)
    dP = (gdn_v3_dp(B, H, C, DV)(DOc, v_hi).float()
          + gdn_v3_dp(B, H, C, DV)(DOc, v_lo).float()).to(dt)
    # cache 不存 dP（它是反向量），参考值现算：dP = dO @ Vnew^T
    dP_ref = don[0, 0][:C].astype(np.float64) @ c0["vnew"].T
    res["dP"] = (dP, dP_ref)
    # t1 = A^T dP（dv 链的第一步）
    t1 = gdn_v3_ta(B, H, C, C, DV)(Ac, dP)
    res["t1"] = (t1, c0["a_mat"].T @ dP_ref)
    qh = gdn_v3_tb(B, H, C, DV, DK)(Qc, Hc)
    res["qh"] = (qh, c0["qh"])

    print(f"UNIT start dim={dim} T={T_len} C={C}", flush=True)
    worst = 0.0
    for name in ITEMS:
        got, ref = res[name]
        r = _rel(got, ref)
        worst = max(worst, r)
        print(f"UNIT dim={dim} name={name:<5} rel={r:.3e} "
              f"verdict={'PASS' if r < 5e-3 else 'FAIL'}", flush=True)
    print(f"UNIT dim={dim} worst={worst:.3e} "
          f"verdict={'ALL_PASS' if worst < 5e-3 else 'SOME_FAIL'}", flush=True)
    return worst


def main():
    for dim in (64, 128):
        try:
            one(dim)
        except Exception as exc:                            # noqa: BLE001
            print(f"UNIT dim={dim} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:120]}", flush=True)
    print("UNIT-DONE", flush=True)


if __name__ == "__main__":
    main()

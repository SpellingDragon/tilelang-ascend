# _ws_cmp.py -- 直接量嫌疑量：v2 前置件的 `ws_h` / `ws_a` 与 golden 的 `hin` / `a_mat`
#
# 为什么不继续用下游六项反推：`dh0 = Delta'_0` 不经过 `H_in`，而 dq/dk/dv/dbeta/dg 都要经过；
# 实测"五项 ~10% + dh0 好一个量级"正是 `H_in` 错、`Delta'` 对的指纹。但指纹不是证据，
# 这里直接把 `ws_h` 与 golden 的 `hin` 逐 (b,h,c) 比。
#
# 同时比 `ws_a`（solve_a 的输出）与 golden 的 `a_mat`：若 A 也错，`dh0` 本该一起坏，
# 所以它应当是对的——这条也当场验。
#
# 打印：WSCMP H=.. T=.. chunks=.. h_rel=.. a_rel=..  (>5e-3 即嫌疑成立)
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from gdn_bwd_v2 import gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a
from example_gdn_bwd import _np_chunk_fwd

CASES = ((1, 256), (1, 512))
GATE = 5e-3
# 注意：本支只能当"粗查"用。`a_rel` 把 `ws_a` 的 [B,H,T,C] 按行切块与 golden 的 a_mat 比，
# 但 solve_a 的存储约定（块在 L 轴上的排布/是否转置）尚未证实；实测 a_rel=14% 而
# harness 在**同一配置**下 dq 只有 4e-4，二者矛盾 ⇒ 在把约定核实之前，
# **h_rel/a_rel 不能当成"前置件不准"的证据**。可信的是 `_relay_attr.py` 的末梢对比。


def one(H, T_len, B=1, DK=128, DV=128, C=64):
    torch.manual_seed(7)
    dt = torch.float16
    dev = "npu"
    rq = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    q32 = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rk = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k32 = rk / rk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    g32 = -torch.randn(B, H, T_len).abs() * 0.5 - 0.05
    beta32 = torch.sigmoid(torch.randn(B, H, T_len))
    h032 = torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1

    q, k, v = q32.to(dt), k32.to(dt), v32.to(dt)
    g, beta, h0 = g32, beta32, h032.to(dt)
    npc = [x.numpy() for x in (q, k, v, g, beta, h0)]

    # 与 relay 逐字对齐地构造前置件
    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
    idt = torch.eye(C, dtype=torch.float32, device=dev)
    t0 = time.perf_counter()
    ws_g, _gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g.to(dev))
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype="float16")(
        k.to(dev), beta.to(dev), ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, 64, 64, dtype="float16")(
        k.to(dev), v.to(dev), beta.to(dev), ws_g, ws_a, h0.to(dev))
    torch.npu.synchronize()
    pre_s = time.perf_counter() - t0

    a_np = ws_a.float().cpu().numpy()
    h_np = ws_h.float().cpu().numpy()
    nc = T_len // C
    qn, kn, vn, gn, betan, h0n = npc

    h_num = h_den = a_num = a_den = 0.0
    for b in range(B):
        for h in range(H):
            _o, cache = _np_chunk_fwd(T_len, DK, DV,
                                      qn[b, h].astype(np.float64), kn[b, h].astype(np.float64),
                                      vn[b, h].astype(np.float64), betan[b, h].astype(np.float64),
                                      gn[b, h].astype(np.float64), h0n[b, h].astype(np.float64), C)
            for c in range(nc):
                hin = np.asarray(cache[c]["hin"], np.float64)
                d = np.abs(h_np[b, h, c].astype(np.float64) - hin)
                h_num = max(h_num, float(d.max()))
                h_den = max(h_den, float(np.abs(hin).max()))
                am = np.asarray(cache[c]["a_mat"], np.float64)
                d2 = np.abs(a_np[b, h, c * C:(c + 1) * C, :].astype(np.float64) - am)
                a_num = max(a_num, float(d2.max()))
                a_den = max(a_den, float(np.abs(am).max()))

    h_rel = h_num / (h_den + 1e-30)
    a_rel = a_num / (a_den + 1e-30)
    print(f"WSCMP H={H:<2} T={T_len:<5} chunks={nc:<2} pre_s={pre_s:5.1f} "
          f"h_rel={h_rel:.3e} a_rel={a_rel:.3e} "
          f"h={'SUSPECT' if h_rel >= GATE else 'ok'} "
          f"a={'SUSPECT' if a_rel >= GATE else 'ok'}", flush=True)
    return h_rel, a_rel


def main():
    print("WSCMP start（ws_h vs golden hin；ws_a vs golden a_mat）", flush=True)
    for H, T_len in CASES:
        try:
            one(H, T_len)
        except Exception as exc:                            # noqa: BLE001
            print(f"WSCMP H={H} T={T_len} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
    print("WSCMP-DONE", flush=True)


if __name__ == "__main__":
    main()

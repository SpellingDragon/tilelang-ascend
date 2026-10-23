# _param_ab.py -- 无 golden 的参数 A/B：只改核内分块参数，看前置件输出是否变
#
# 背景：DK/DV=128 时六项从 ~1e-3 跳到 ~1e-1（notes [148]，同探针只改这一个编译参数）。
# DK≠DV 的拆分跑不通——relay 在 host 侧就 `dim 3: 128 vs 64` 报错（DK≠DV 不是受支持配置）。
# 那就绕开形状，直接对**核内分块循环**做 A/B：
#
#   * `scan1(BV=64)` vs `scan1(BV=128)`：DV=128 时前者 bv_num=2、后者 bv_num=1。
#     两者数学上必须逐元素相同。若不同 ⇒ `scan1` 的 bv 循环有缺陷。
#   * `solve_a(BK=64)` vs `solve_a(BK=128)`：DK=128 时前者 bk_num=2、后者 bk_num=1。
#     同理，若不同 ⇒ `solve_a` 的 bk 循环有缺陷。
#
# 这条 A/B 不涉及 golden、不涉及 h0、不涉及 chunk 数，只动一个整数参数。
#
# 打印：PAB scan1 DV=.. BV=.. vs BV=.. rel=.. ; PAB solve_a DK=.. BK=.. vs BK=.. rel=..
import numpy as np
import torch
import torch_npu  # noqa: F401

from gdn_bwd_v2 import gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a

GATE = 5e-3


def _inputs(H, T_len, B, DK, DV):
    torch.manual_seed(7)
    dt = torch.float16
    rq = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    rk = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k = (rk / rk.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(dt).npu()
    q = (rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(dt).npu()
    v = torch.randn(B, H, T_len, DV, dtype=torch.float32).to(dt).npu()
    g = (-torch.randn(B, H, T_len).abs() * 0.5 - 0.05).npu()
    beta = torch.sigmoid(torch.randn(B, H, T_len)).npu()
    h0 = (torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1).to(dt).npu()
    return q, k, v, g, beta, h0


def _rel(a, b):
    aa, bb = a.float().cpu().numpy().astype(np.float64), b.float().cpu().numpy().astype(np.float64)
    return float(np.abs(aa - bb).max()) / (float(np.abs(bb).max()) + 1e-30)


def scan1_ab(H, T_len, B, DK, DV, C, bv_a, bv_b):
    _q, k, v, g, beta, h0 = _inputs(H, T_len, B, DK, DV)
    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device="npu"), -1)
    idt = torch.eye(C, dtype=torch.float32, device="npu")
    ws_g, _gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype="float16")(k, beta, ws_g, msk_j, idt)
    ha = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, bv_a, 64, dtype="float16")(k, v, beta, ws_g, ws_a, h0)
    hb = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, bv_b, 64, dtype="float16")(k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()
    r = _rel(ha, hb)
    print(f"PAB scan1 H={H} T={T_len} DK={DK} DV={DV} BV={bv_a} vs BV={bv_b} rel={r:.3e} "
          f"verdict={'SAME' if r < GATE else 'DIFFER'}", flush=True)


def solve_a_ab(H, T_len, B, DK, DV, C, bk_a, bk_b):
    _q, k, v, g, beta, h0 = _inputs(H, T_len, B, DK, DV)
    msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device="npu"), -1)
    idt = torch.eye(C, dtype=torch.float32, device="npu")
    ws_g, _gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    aa = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, bk_a, dtype="float16")(k, beta, ws_g, msk_j, idt)
    ab = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, bk_b, dtype="float16")(k, beta, ws_g, msk_j, idt)
    torch.npu.synchronize()
    r = _rel(aa, ab)
    print(f"PAB solve_a H={H} T={T_len} DK={DK} DV={DV} BK={bk_a} vs BK={bk_b} rel={r:.3e} "
          f"verdict={'SAME' if r < GATE else 'DIFFER'}", flush=True)


def main():
    T, B, H, C = 256, 1, 1, 64
    print("PAB start（只改核内分块参数，无 golden）", flush=True)
    for name, fn in (
        ("scan1 DV=128 BV=64/128", lambda: scan1_ab(H, T, B, 128, 128, C, 64, 128)),
        ("solve_a DK=128 BK=64/128", lambda: solve_a_ab(H, T, B, 128, 128, C, 64, 128)),
        ("scan1 DV=64  BV=64/64 ", lambda: scan1_ab(H, T, B, 64, 64, C, 64, 64)),
        ("solve_a DK=64 BK=64/64 ", lambda: solve_a_ab(H, T, B, 64, 64, C, 64, 64)),
    ):
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001
            print(f"PAB {name} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
    print("PAB-DONE", flush=True)


if __name__ == "__main__":
    main()

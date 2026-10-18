# _w11_gate.py — 三档输入门（G5）：静态核对 L0 用例是否真的覆盖 g≠0 / 多 chunk / h0≠0
#
# 背景（复盘 P2）：solve_a/scan1 长期只用 g≡0 验证，使列广播退化（D15）潜伏七个窗口；
# L0 basic 是单 chunk + h0=0，使 Δ′ 未初始化、dKt 未实现、spill 别名全部被掩盖。
# 本门禁不跑 kernel，只检查"验证输入是否具备区分力"，缺档即 fail。
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from test_gdn_bwd_v2 import L0_CASES, _case_inputs  # noqa: E402


def main():
    rows = []
    have = {"g_ne0": False, "multichunk": False, "h0_ne0": False, "fp32": False}
    for name, B, H, T, DK, DV, dtype, h0m, sp in L0_CASES:
        (q, k, v, g, beta, h0, do), _ = _case_inputs(B, H, T, DK, DV, dtype, h0m, sp)
        gg = g.float().numpy()
        # g != 0 means: values differ across the chunk (not all equal), and any
        # non-zero element. A constant g (e.g. all zeros) hides index bugs.
        finite = np.isfinite(gg)
        n_nan = int((~finite).sum())
        g_var = float(np.nanmax(np.abs(gg - gg[..., :1]))) if gg.size else 0.0
        g_any = float(np.nanmax(np.abs(gg))) if gg.size else 0.0
        g_ne0 = g_var > 1e-6 and g_any > 1e-6
        multichunk = T > 64
        h0_ne0 = float(h0.float().abs().max()) > 0 if h0 is not None else False
        is_fp32 = dtype == "float32"
        rows.append((f"{name}/{dtype}/h0={h0m}", T, multichunk, f"{g_var:.2e}", g_ne0,
                     f"{h0_ne0}", is_fp32, n_nan))
        have["g_ne0"] |= g_ne0
        have["multichunk"] |= multichunk
        have["h0_ne0"] |= h0_ne0
        have["fp32"] |= is_fp32

    print(f"{'case':28s} {'T':>4s} {'multi':>5s} {'g_spread':>9s} {'g!=0':>5s} {'h0!=0':>5s} {'fp32':>4s} {'nan':>4s}")
    for r in rows:
        print(f"{r[0]:28s} {r[1]:4d} {str(r[2]):>5s} {r[3]:>9s} {str(r[4]):>5s} {r[5]:>5s} {str(r[6]):>4s} {r[7]:4d}")
    print("COVERAGE", {k: bool(v) for k, v in have.items()})
    missing = [k for k, v in have.items() if not v]
    if missing:
        print(f"GATE-FAIL missing={missing}")
        return 1
    print("GATE-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

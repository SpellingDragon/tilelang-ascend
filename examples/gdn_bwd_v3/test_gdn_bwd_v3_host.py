#!/usr/bin/env python3
"""test_gdn_bwd_v3_host.py -- gdn_bwd_v3 的公式级测试（纯 CPU，不需要 NPU）。

把 `gdn_bwd_v3_relay` 的每一步用 numpy 复刻一遍，与仓内 golden 比。目的是把
"公式错"与"卡上实现错"分开：本文件全绿时，卡上的偏差只可能来自实现。

为什么能在 CPU 上跑：只 import `test_gdn_bwd_v2` 的三个纯 numpy helper
（`_case_inputs` / `_np_chunk_fwd` / `_golden_item`），`torch_npu` 打桩。

运行：
    python3 examples/gdn_bwd_v3/test_gdn_bwd_v3_host.py
    # 或
    python3 -m pytest examples/gdn_bwd_v3/test_gdn_bwd_v3_host.py -q
"""

import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, os.pardir, "gdn_bwd_v2"))
for _p in (_V2, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                    # 卡上有 torch_npu；本机打桩，纯 numpy 可跑
    import torch_npu  # noqa: F401
except ImportError:
    _tm = types.ModuleType("torch_npu")
    _tm.npu = types.SimpleNamespace(synchronize=lambda *a, **k: None,
                                    set_device=lambda *a, **k: None)
    sys.modules["torch_npu"] = _tm

from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd, _golden_item  # noqa: E402

TOL = 1e-9


def relay_reference(T_len=256, C=64, DK=64, DV=64):
    """按 gdn_bwd_v3_relay 的顺序复刻全序列反向，返回六梯度（fp64）。"""
    NC = T_len // C
    (q, k, v, g, beta, h0, do), gold = _case_inputs(1, 1, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn, DOn = (beta.double().numpy()[0, 0], g.double().numpy()[0, 0],
                       h0.double().numpy()[0, 0], do.double().numpy()[0, 0])
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)

    dq = np.zeros((T_len, DK))
    dk = np.zeros((T_len, DK))
    dv = np.zeros((T_len, DV))
    dbeta = np.zeros(T_len)
    dg = np.zeros(T_len)
    carry = 0.0
    delta_next = np.zeros((DV, DK))          # Delta'_{c+1}

    for c in range(NC - 1, -1, -1):
        ca = cache[c]
        sl = slice(c * C, (c + 1) * C)
        gam2, kk, msk_m, msk_j = ca["gam_2d"], ca["kk"], ca["msk_m"], ca["msk_j"]
        a_mat, e_vec, th, glast = ca["a_mat"], ca["e_vec"], ca["th"], ca["glast"]
        vnew, hin, kt = ca["vnew"], ca["hin"], ca["kt"]
        bv = ca["beta"][:, None] * Vn[sl]
        bek = (ca["beta"] * e_vec)[:, None] * Kn[sl]

        # B1
        dvnew = ca["p_mat"].T @ DOn[sl] + kt @ delta_next.T
        # B2/B3/B4
        dpk = DOn[sl] @ vnew.T
        dqkt = dpk * gam2 * msk_m
        dq[sl] = dqkt @ Kn[sl] + (e_vec[:, None] * DOn[sl]) @ hin
        # B6/B7/B8
        dw = -dvnew @ hin
        t1 = a_mat.T @ dvnew
        t2 = a_mat.T @ dw
        dv[sl] = ca["beta"][:, None] * t1
        # B9/B10
        dkt = vnew @ delta_next
        dth = (Kn[sl] * dkt).sum(1)
        # B11
        da = dvnew @ bv.T + dw @ bek.T
        dl = msk_j * (-(a_mat.T @ da @ a_mat.T))
        dx = dl * ca["beta"][:, None] * gam2
        # dk
        dk[sl] = (dqkt.T @ Qn[sl]
                  + (ca["beta"] * e_vec)[:, None] * t2
                  + th[:, None] * dkt
                  + (dx + dx.T) @ Kn[sl])
        # dbeta
        dbeta[sl] = ((Vn[sl] * t1).sum(1)
                     + e_vec * (Kn[sl] * t2).sum(1)
                     + (dl * gam2 * kk).sum(1))
        # dgam
        dgam2 = dpk * msk_m * (Qn[sl] @ Kn[sl].T) + dl * ca["beta"][:, None] * kk
        w12 = dgam2 * gam2
        dgam = w12.sum(1) - w12.sum(0) - dth * th
        qh = Qn[sl] @ hin.T
        de = (DOn[sl] * qh).sum(1) + ca["beta"] * (Kn[sl] * t2).sum(1)
        dgam = dgam + e_vec * de
        dg[sl] = np.cumsum(dgam[::-1])[::-1] + carry
        carry += dgam.sum()
        # B15: 传给更早的 chunk
        delta_next = (e_vec[:, None] * DOn[sl]).T @ Qn[sl] - dvnew.T @ ca["w_mat"] + glast * delta_next

    ref = {name: np.asarray(_golden_item(gold, name, 0), np.float64)[0, 0]
           for name in ("dq", "dk", "dv", "dbeta", "dg")}
    ref["dh0"] = np.asarray(_golden_item(gold, "dh0", 0), np.float64)[0, 0]
    got = {"dq": dq, "dk": dk, "dv": dv, "dbeta": dbeta, "dg": dg, "dh0": delta_next}
    return got, ref


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / (np.abs(np.asarray(b)).max() + 1e-30))


def main():
    got, ref = relay_reference()
    worst = 0.0
    for name in ("dq", "dk", "dv", "dbeta", "dg", "dh0"):
        r = _rel(got[name], ref[name])
        worst = max(worst, r)
        print(f"HOSTV3 {name:6s} rel={r:.3e}  {'PASS' if r < TOL else 'FAIL'}")
    print(f"HOSTV3 worst={worst:.3e} verdict={'PASS' if worst < TOL else 'FAIL'}")
    return 0 if worst < TOL else 1


def test_relay_formulas_match_golden():
    got, ref = relay_reference()
    for name in ("dq", "dk", "dv", "dbeta", "dg", "dh0"):
        assert _rel(got[name], ref[name]) < TOL, name


if __name__ == "__main__":
    sys.exit(main())

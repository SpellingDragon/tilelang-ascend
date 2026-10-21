# _w11_relay_host.py — 先在 host 上把 relay 的公式复刻一遍，判断是"公式错"还是"卡上实现错"
#
# 用 golden 的逐 chunk 量（cache）与 dO，按 gdn_bwd_v3_relay 的顺序算：
#   dvnew = P^T dO + Kt Delta'^T ; dl/dx ; dk ; dbeta ; dgam ; dg
# 与 golden 的 dk/dbeta/dg 比。host 版若对不上，就别再去卡上找。
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, os.pardir))
for _p in ("/tmp", _V2):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:                                  # 卡上有 torch_npu；本机没有就打桩，纯 numpy 可跑
    import torch_npu  # noqa: F401
except ImportError:
    _tm = types.ModuleType("torch_npu")
    _tm.npu = types.SimpleNamespace(synchronize=lambda *a, **k: None, set_device=lambda *a, **k: None)
    sys.modules["torch_npu"] = _tm
import numpy as np  # noqa: E402


def main():
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd, _golden_item

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    NC = T_len // C
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn, DOn = (beta.double().numpy()[0, 0], g.double().numpy()[0, 0],
                       h0.double().numpy()[0, 0], do.double().numpy()[0, 0])
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)

    dk_g = np.asarray(_golden_item(gold, "dk", 0), np.float64)[0, 0]
    db_g = np.asarray(_golden_item(gold, "dbeta", 0), np.float64)[0, 0]
    dg_g = np.asarray(_golden_item(gold, "dg", 0), np.float64)[0, 0]

    dk = np.zeros_like(dk_g)
    db = np.zeros(T_len, np.float64)
    dg = np.zeros(T_len, np.float64)
    carry = 0.0
    delta_next = np.zeros((DV, DK), np.float64)     # Delta'_{c+1}

    for c in range(NC - 1, -1, -1):
        ca = cache[c]
        sl = slice(c * C, (c + 1) * C)
        gam2, kk, msk_m, msk_j = ca["gam_2d"], ca["kk"], ca["msk_m"], ca["msk_j"]
        a_mat, e_vec, glast, th = ca["a_mat"], ca["e_vec"], ca["glast"], ca["th"]
        kt, vnew, hin = ca["kt"], ca["vnew"], ca["hin"]
        bv = ca["beta"][:, None] * Vn[sl]
        bek = (ca["beta"] * e_vec)[:, None] * Kn[sl]
        # B1: dvnew = P^T dO + Kt Delta'^T
        dvnew = ca["p_mat"].T @ DOn[sl] + kt @ delta_next.T
        # B6/B7/B8
        dw = -dvnew @ hin
        t1 = a_mat.T @ dvnew
        t2 = a_mat.T @ dw
        # B11
        da = dvnew @ bv.T + dw @ bek.T
        dl = msk_j * (-(a_mat.T @ da @ a_mat.T))
        dx = dl * ca["beta"][:, None] * gam2
        # dk
        dpk = DOn[sl] @ vnew.T
        dqkt = dpk * gam2 * msk_m
        dkt = vnew @ delta_next
        dk[sl] = (dqkt.T @ Qn[sl]
                  + (ca["beta"] * e_vec)[:, None] * t2
                  + th[:, None] * dkt
                  + (dx + dx.T) @ Kn[sl])
        # dbeta
        db[sl] = ((Vn[sl] * t1).sum(1) + e_vec * (Kn[sl] * t2).sum(1)
                  + (dl * gam2 * kk).sum(1))
        # Delta'_c
        delta_c = (e_vec[:, None] * DOn[sl]).T @ Qn[sl] - dvnew.T @ ca["w_mat"] + glast * delta_next
        # dgam
        dgam2 = dpk * msk_m * (Qn[sl] @ Kn[sl].T) + dl * ca["beta"][:, None] * kk
        w12 = dgam2 * gam2
        dgam = w12.sum(1) - w12.sum(0)
        dth = (Kn[sl] * dkt).sum(1)
        dgam = dgam - dth * th
        dgam[C - 1] += (dth * th).sum()
        qh = Qn[sl] @ hin.T
        de = (DOn[sl] * qh).sum(1) + ca["beta"] * (Kn[sl] * t2).sum(1)
        dgam = dgam + e_vec * de
        dgam[C - 1] += glast * float((hin * delta_next).sum())
        dg[sl] = np.cumsum(dgam[::-1])[::-1] + carry
        carry += dgam.sum()
        delta_next = delta_c

    for name, got, ref in (("dk", dk, dk_g), ("dbeta", db, db_g), ("dg", dg, dg_g)):
        rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
        print(f"HOST {name} rel={rel:.3e} goldmax={float(np.abs(ref).max()):.3e}")
    print("HOST-DONE")


if __name__ == "__main__":
    main()

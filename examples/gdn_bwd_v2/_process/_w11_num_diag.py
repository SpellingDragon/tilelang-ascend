# w11 numeric diag — 逐 stage / 逐边界 box 统计 nan/inf/absmax（fp32 basic 复现全 nan）
import os
import sys

sys.path.insert(0, "/tmp")
import torch  # noqa: E402
# fixed seed: _case_inputs has none, and without it every run draws
# different data -- cross-car comparisons become meaningless (10-21 finding).
torch.manual_seed(7)
import torch_npu  # noqa: E402,F401
from gdn_bwd_v2 import (  # noqa: E402
    gdn_bwd_v2_cumsum,
    gdn_bwd_v2_scan1,
    gdn_bwd_v2_scan2a,
    gdn_bwd_v2_scan2b,
    gdn_bwd_v2_solve_a,
)
from test_gdn_bwd_v2 import _case_inputs, _npu  # noqa: E402


def stat(name, x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    xf = x.detach().float().cpu()
    print(f"STAT-{name} nan={int(torch.isnan(xf).sum())} "
          f"inf={int(torch.isinf(xf).sum())} absmax={float(xf.abs().max()):.3e} "
          f"mean={float(xf.mean()):.3e}")


def run(tag, args):
    print(f"=== CASE {tag} ===")
    (q, k, v, g, beta, h0, do), gold = _case_inputs(*args)
    B, H, T_len, DK = k.shape
    DV = v.shape[-1]
    C, BV, BK = 64, 64, 64
    dtype = args[5]
    q, k, v, g, beta, h0, do = (_npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
    ws_g, ws_gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    torch.npu.synchronize()
    stat("ws_g", ws_g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, BK, dtype=dtype)(k, beta, ws_g, msk_j, idt)
    torch.npu.synchronize()
    stat("ws_a", ws_a)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()
    stat("ws_h", ws_h)
    # H2 check: scan1's cross-chunk state must match the fp64 replay per chunk.
    # chunk 0's H_in is h0 itself, so a growing error means the recurrence is
    # the culprit (the dq per-chunk split showed t00 exact, t01..t03 off 2-4%).
    import numpy as _np
    from test_gdn_bwd_v2 import _np_chunk_fwd as _nf
    Qn, Kn, Vn = (q.double().cpu().numpy()[0, 0], k.double().cpu().numpy()[0, 0],
                  v.double().cpu().numpy()[0, 0])
    Bn, Gn, Hn = (beta.double().cpu().numpy()[0, 0], g.double().cpu().numpy()[0, 0],
                  h0.double().cpu().numpy()[0, 0])
    _, _cache = _nf(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)
    _wh = ws_h.float().cpu().numpy()
    # H3 check: Gamma per chunk must match the fp64 replay as well. The dq
    # error is 1-4% on chunks 1..3 while WS_H is exact at 5e-4, and Gamma is
    # the only other chunk-dependent input of dqkt.
    _wg = ws_gam.float().cpu().numpy()
    for _cc in range(len(_cache)):
        _gg = _np.asarray(_cache[_cc]["gam_2d"], _np.float64)
        _gc = _np.asarray(_wg[0, 0, _cc], _np.float64)
        _dd = float(_np.abs(_gc - _gg).max())
        print(f"GAM cc={_cc} maxdiff={_dd:.3e} goldmax={float(_np.abs(_gg).max()):.3e} "
              f"rel={_dd / (float(_np.abs(_gg).max()) + 1e-30):.3e}")
    for _cc in range(len(_cache)):
        _g = _np.asarray(_cache[_cc]["hin"], _np.float64)
        _c = _np.asarray(_wh[0, 0, _cc], _np.float64)
        _d = float(_np.abs(_c - _g).max())
        print(f"S1G cc={_cc} maxdiff={_d:.3e} goldmax={float(_np.abs(_g).max()):.3e} "
              f"rel={_d / (float(_np.abs(_g).max()) + 1e-30):.3e}")
    outs_a = gdn_bwd_v2_scan2a(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        q, k, v, do, beta, ws_g, ws_a, msk_m, ones_dv, ws_h, ws_gam)
    torch.npu.synchronize()
    # H4: split dq per chunk into its two terms (B4 = dqkt@K, B5b = e_do@H_in)
    # and compare the card's box against each -- the two chunk inputs are exact,
    # so the 1-4% has to be inside one of these.
    _dq_card = outs_a[0].float().cpu().numpy()
    _doc = do.double().cpu().numpy()[0, 0]
    _kc = k.double().cpu().numpy()[0, 0]
    for _cc in range(len(_cache)):
        _ca = _cache[_cc]
        _sl = slice(_cc * C, (_cc + 1) * C)
        _P = _np.asarray(_ca["p_mat"], _np.float64)
        _G = _np.asarray(_ca["gam_2d"], _np.float64)
        _M = _np.asarray(_ca["msk_m"], _np.float64)
        _VN = _np.asarray(_ca["vnew"], _np.float64)
        _DP = _doc[_sl] @ _VN.T
        _B4 = ((_DP * _G) * _M) @ _kc[_sl]
        _E = _np.asarray(_ca["e_vec"], _np.float64)
        _B5 = (_E[:, None] * _doc[_sl]) @ _np.asarray(_ca["hin"], _np.float64)
        _gold = _B4 + _B5
        _card = _np.asarray(_dq_card[0, 0, _sl], _np.float64)
        _gm = float(_np.abs(_gold).max()) + 1e-30
        # H5: if the card used a neighbouring chunk's H_in, replacing hin here
        # reproduces it exactly. chunk 0's B5 is zero (h0=zero), so a shift is
        # invisible there and shows up only on chunks 1..3 -- as observed.
        def _b5(_c):
            if _c < 0 or _c >= len(_cache):
                return None
            _h = _np.asarray(_cache[_c]["hin"], _np.float64)
            return (_E[:, None] * _doc[_sl]) @ _h
        _rel = lambda x: float(_np.abs(_card - x).max()) / _gm
        _prev, _next = _b5(_cc - 1), _b5(_cc + 1)
        _sp = f"{_rel(_prev):.3e}" if _prev is not None else "-"
        _sn = f"{_rel(_next):.3e}" if _next is not None else "-"
        print(f"DQSPLIT cc={_cc} card_vs_gold={_rel(_gold):.3e} "
              f"card_vs_B4={_rel(_B4):.3e} card_vs_B5={_rel(_B5):.3e} "
              f"card_vs_prevH={_sp} card_vs_nextH={_sn} "
              f"B4max={float(_np.abs(_B4).max()):.3e} B5max={float(_np.abs(_B5).max()):.3e}")
    for i, o in enumerate(outs_a):
        stat(f"s2a_out{i}", o)
    outs_b = gdn_bwd_v2_scan2b(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        k, v, beta, ws_g, ws_a, msk_j, ones_dv, ones_dk, ones_c,
        outs_a[2], outs_a[3], outs_a[4], outs_a[5], outs_a[6], outs_a[7], outs_a[8])
    torch.npu.synchronize()
    if os.environ.get("GDN_S2A_VN_DBG") == "1":
        # the probe overwrote WS_DKT (scan2a out4) with the Vnew of chunk
        # GDN_S2A_VN_CC; compare it with the fp64 replay of the same chunk.
        _vcc = int(os.environ.get("GDN_S2A_VN_CC", "0"))
        _vc = _np.asarray(outs_a[4].float().cpu().numpy()[0, 0]
                          [_vcc * C:(_vcc + 1) * C], _np.float64)
        _vg = _np.asarray(_cache[_vcc]["vnew"], _np.float64)
        _dd = float(_np.abs(_vc - _vg).max())
        print(f"VN cc={_vcc} maxdiff={_dd:.3e} goldmax={float(_np.abs(_vg).max()):.3e} "
              f"rel={_dd / (float(_np.abs(_vg).max()) + 1e-30):.3e}")
        # split Vnew into U0 and Wterm: Wterm is the only term that vanishes on
        # chunk 0, so this tells which half carries the 4.19%.
        _u0g = _np.asarray(_cache[_vcc]["u0"], _np.float64)
        _wg = _np.asarray(_cache[_vcc]["w_mat"], _np.float64)   # +W
        _hg = _np.asarray(_cache[_vcc]["hin"], _np.float64)
        _wt_host = -(_wg @ _hg.T)
        _wt_card = _vc - _u0g
        _rr = lambda a, b: float(_np.abs(a - b).max()) / (float(_np.abs(b).max()) + 1e-30)
        print(f"VNSPLIT cc={_vcc} card_vs_u0={_rr(_vc, _u0g):.3e} "
              f"wt_card_vs_host={_rr(_wt_card, _wt_host):.3e} "
              f"vnew_vs_gold={_rr(_vc, _u0g + _wt_host):.3e} "
              f"u0max={float(_np.abs(_u0g).max()):.3e} wtmax={float(_np.abs(_wt_host).max()):.3e}")
    if os.environ.get("GDN_S2B_DBG") == "1":
        # the probe overwrote this scan2a output with what scan2b read for
        # dVnew; compare against the box's out2 (both are the same tensor).
        stat("s2b_probe_dvread", outs_a[8])
    names = ["dkg", "dvg", "dbeta", "dg"]
    for i, o in enumerate(outs_b):
        stat(f"s2b_{names[i] if i < 4 else i}", o)


def _ones(rows, dtype):
    od = torch.float32 if dtype == "float32" else torch.float16
    return torch.ones(rows, 16, dtype=od).npu()


if __name__ == "__main__":
    msk_m = torch.tril(torch.ones(64, 64, dtype=torch.float32)).npu()
    msk_j = torch.tril(torch.ones(64, 64, dtype=torch.float32), -1).npu()
    idt = torch.eye(64, dtype=torch.float32).npu()
    import sys as _sys

    _cases = [
        ("l0_basic_fp16", (1, 1, 64, 64, 64, "float16", "zero", False)),
        # four chunks, single batch: the multi-chunk relay stages (WS_D / WS_H
        # / WS_DGAM) first get exercised here; dq was 2.2e-1 on card while the
        # single-chunk case sits at 1.9e-3.
        ("l0_multi_fp16", (1, 1, 256, 64, 64, "float16", "zero", False)),
        ("l0_typical_fp16", (1, 2, 256, 64, 64, "float16", "zero", False)),
        ("l0_basic_fp32", (1, 1, 64, 64, 64, "float32", "zero", False)),
    ]
    if len(_sys.argv) > 1:
        _cases = [c for c in _cases if c[0] == _sys.argv[1]] or _cases
    for _tag, _args in _cases:
        dt_str = _args[5]
        ones_dv = _ones(64, dt_str)
        ones_dk = _ones(64, dt_str)
        ones_c = _ones(64, dt_str)
        run(_tag, _args)
    print("W11DIAG-DONE")

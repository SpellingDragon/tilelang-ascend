# w11 numeric diag — 逐 stage / 逐边界 box 统计 nan/inf/absmax（fp32 basic 复现全 nan）
import sys

sys.path.insert(0, "/tmp")
import torch  # noqa: E402
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
    outs_a = gdn_bwd_v2_scan2a(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        q, k, v, do, beta, ws_g, ws_a, msk_m, ones_dv, ws_h, ws_gam)
    torch.npu.synchronize()
    for i, o in enumerate(outs_a):
        stat(f"s2a_out{i}", o)
    outs_b = gdn_bwd_v2_scan2b(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)(
        k, v, beta, ws_g, ws_a, msk_j, ones_dv, ones_dk, ones_c,
        outs_a[2], outs_a[3], outs_a[4], outs_a[5], outs_a[6], outs_a[7], outs_a[8])
    torch.npu.synchronize()
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

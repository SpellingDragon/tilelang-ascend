# _w11_s2b_zero.py — scan2b 边界 box 全零对照：分离「box 带 nan」vs「scan2b 内部 nan」
#   构造：cumsum/solve_a/scan1 正常跑出 ws_g/ws_a/ws_h；七个边界 box 全部置 0；
#   调 scan2b，统计四个输出的 nonfinite。
#   判读：输出全有限  ⇒ nan 由某个 box 带入（回到 box 审计）
#         输出仍 nan  ⇒ scan2b 内部（未初始化/溢出），再按生成码查
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401,E402
except ImportError:
    pass

from test_gdn_bwd_v2 import _case_inputs, _npu  # noqa: E402
from gdn_bwd_v2 import (  # noqa: E402
    gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_scan2b, gdn_bwd_v2_solve_a,
)


def stat(name, t):
    a = t.float().cpu().numpy()
    print(f"STAT-{name} nonfinite={int((~np.isfinite(a)).sum())} "
          f"absmax={float(np.nanmax(np.abs(a))):.3e} mean={float(np.nanmean(a)):.3e}", flush=True)


def main(dtype_str="float16"):
    (q, k, v, g, beta, h0, do), _ = _case_inputs(1, 1, 64, 64, 64, dtype_str, "zero", False)
    qn, kn, vn, gn, bn, hn, don = (_npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
    dt = torch.float16 if dtype_str == "float16" else torch.float32
    msk_j = torch.tril(torch.ones(64, 64, dtype=torch.float32)).npu()
    msk_j = torch.tril(msk_j, -1)
    idt = torch.eye(64, dtype=torch.float32).npu()
    ones_dv = torch.ones(64, 16, dtype=dt).npu()
    ones_dk = torch.ones(64, 16, dtype=dt).npu()
    ones_c = torch.ones(64, 16, dtype=dt).npu()
    ws_g, ws_gam = gdn_bwd_v2_cumsum(1, 1, 64, 64)(gn)
    ws_a = gdn_bwd_v2_solve_a(1, 1, 64, 64, 64, 64, dtype=dtype_str)(kn, bn, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(1, 1, 64, 64, 64, 64, 64, 64, dtype=dtype_str)(kn, vn, bn, ws_g, ws_a, hn)
    torch.npu.synchronize()
    print("STAT-prep done", flush=True)
    z_dv = torch.zeros(1, 1, 64, 64, dtype=dt).npu()
    z_dw = torch.zeros(1, 1, 64, 64, dtype=dt).npu()
    z_dkt = torch.zeros(1, 1, 64, 64, dtype=torch.float32).npu()
    z_de = torch.zeros(1, 1, 64, dtype=torch.float32).npu()
    z_dg2 = torch.zeros(1, 1, 64, 64, dtype=torch.float32).npu()
    z_dg = torch.zeros(1, 1, 64, dtype=torch.float32).npu()
    z_dk = torch.zeros(1, 1, 64, 64, dtype=torch.float32).npu()
    outs = gdn_bwd_v2_scan2b(1, 1, 64, 64, 64, 64, 64, 64, dtype=dtype_str)(
        kn, vn, bn, ws_g, ws_a, msk_j, ones_dv, ones_dk, ones_c,
        z_dv, z_dw, z_dkt, z_de, z_dg2, z_dg, z_dk)
    torch.npu.synchronize()
    for i, nm in enumerate(("dkg", "dvg", "dbeta", "dg")):
        stat(f"s2bzero_{nm}", outs[i])
    print("S2BZERO-DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "float16")

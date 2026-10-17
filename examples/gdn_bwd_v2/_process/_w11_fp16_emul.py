# _w11_fp16_emul.py — ideal-fp16 pipeline emulation for scan2a's boxes (CPU only).
#   Every tensor is rounded to its kernel storage dtype (inputs/A/P/dqkt fp16,
#   S and dot-products fp32/fp64 accumulator), GEMMs are exact.
#   => this is the BEST accuracy any correct kernel can reach with these dtypes.
#   Compare with the on-card rels (DQ/DH0/WS_DV/WS_DW/WS_DK ~1.0, WS_DGAM2 4e-2):
#     emul_rel << card_rel  => the card is a bug, not precision
#     emul_rel ~= card_rel  => fp16 storage itself loses that much
import sys

import numpy as np

sys.path.insert(0, "/tmp")
from test_gdn_bwd_v2 import _case_inputs  # noqa: E402


def f16(x):
    return np.asarray(x, np.float64).astype(np.float16).astype(np.float64)


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(d.max() / (np.abs(b).max() + 1e-30))


def run():
    (q, k, v, g, beta, h0, do), _ = _case_inputs(1, 1, 64, 64, 64, "float16", "zero", False)
    C = 64
    Q, K, V, DO = f16(q[0, 0].numpy()), f16(k[0, 0].numpy()), f16(v[0, 0].numpy()), f16(do[0, 0].numpy())
    BETA = beta[0, 0].numpy().astype(np.float64)   # kernel: Beta is fp32
    G = g[0, 0].numpy().astype(np.float64)          # kernel: S (cumsum) is fp32
    HIN = f16(h0[0, 0].numpy())

    gam = np.cumsum(G)
    Gam = np.exp(gam[:, None] - gam[None, :])
    E = np.exp(gam)
    glast = np.exp(gam[-1])
    th = np.exp(gam[-1] - gam)
    KKt = K @ K.T
    M = np.tril(np.ones((C, C)))
    J = np.tril(np.ones((C, C)), -1)
    L = J * (BETA[:, None] * (Gam * KKt))
    A = np.linalg.inv(np.eye(C) + L)
    A16 = f16(A)                                    # ws_a is fp16

    u0 = A16 @ (BETA[:, None] * V)
    W = A16 @ ((BETA * E)[:, None] * K)
    Vnew = f16(u0 - W @ HIN.T)                      # stored fp16 (p_l1/vnew_l1)
    raw = Q @ K.T
    P = f16(Gam * M * raw)                          # p_l1 fp16
    dvnew = P.T @ DO
    dp = DO @ Vnew.T
    dqkt = f16(dp * Gam * M)                        # da_l1 fp16
    dq = dqkt @ K
    dh = (E[:, None] * DO).T @ Q - Vnew.T @ W
    dw = -(dvnew @ HIN)
    dk = dqkt.T @ Q
    dgam2 = (dp * M) * raw

    # exact fp64 reference of the same formulas (no fp16 rounding anywhere)
    u0e = A @ (BETA[:, None] * V)
    We = A @ ((BETA * E)[:, None] * K)
    Vne = u0e - We @ HIN.T
    Pe = Gam * M * raw
    dvne = Pe.T @ DO
    dpe = DO @ Vne.T
    dqkte = dpe * Gam * M
    ref = {
        "DQ": dqkte @ K,
        "DH0": (E[:, None] * DO).T @ Q - Vne.T @ We,
        "WS_DV": dvne,
        "WS_DW": -(dvne @ HIN),
        "WS_DK": dqkte.T @ Q,
        "WS_DGAM2": (dpe * M) * raw,
    }
    got = {
        "DQ": dq, "DH0": dh, "WS_DV": dvnew, "WS_DW": dw,
        "WS_DK": dk, "WS_DGAM2": dgam2,
    }
    print("=== FP16-EMUL (ideal fp16 storage + exact GEMM) ===")
    for name in ref:
        print(f"{name:10s} emul_rel={rel(got[name], ref[name]):.3e} "
              f"ref_max={float(np.abs(ref[name]).max()):.3e}")
    print("FP16EMUL-DONE")


if __name__ == "__main__":
    run()

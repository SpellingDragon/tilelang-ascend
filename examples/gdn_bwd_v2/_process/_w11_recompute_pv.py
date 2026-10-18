# _w11_recompute_pv.py — 用 dump 出的 P 与 dO 在 host 重算 dVnew = P^T dO
#
# 为什么需要它：CV 混合核里加一条 dump 会改变 liveness 与生成时序，所以
# "卡上 dump 出的 dVnew"与"生产构建里的 dVnew"不是同一份代码路径。判定的
# 可靠方式是：把两个**操作数**（P、dO）dump 出来，在 host 用同一算式重算，
# 再与 golden 和卡上 WS_DV 三方对照。
#
# 用法（本地，无需 NPU）：
#   python _w11_recompute_pv.py            # 默认读 /tmp/w11dump_1.npy (P)、
#                                          # /tmp/w11dump_2.npy (dO)、
#                                          # /tmp/w11box_out2.npy (卡上 WS_DV)
# 判读：
#   rel(gold, P^T dO) 小  ⇒ 操作数正确，GEMM 本身错（换 tB 形态）
#   rel(gold, P^T dO) 大  ⇒ 看 P / dO 谁非有限，回到对应交接链
import sys

import numpy as np

sys.path.insert(0, "/tmp")  # 卡上导出时用；本地运行时忽略
sys.path.insert(0, "/Users/pengweiye/Documents/codes/tilelang-ascend/examples/gdn_bwd_v2")

from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd  # noqa: E402


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.abs(a[m] - b[m]).max() / (np.abs(b[m]).max() + 1e-30))


def stats(name, a):
    a = np.asarray(a)
    print(f"STAT-{name} absmax={float(np.nanmax(np.abs(a))):.3e} "
          f"nan={int(np.isnan(a).sum())} inf={int(np.isinf(a).sum())}")


def load(path, what):
    try:
        a = np.load(path)
    except FileNotFoundError:
        print(f"MISSING {what}: {path}")
        return None
    print(f"LOADED {what}: {path} shape={a.shape}")
    return a


def main():
    P = load("/tmp/w11dump_1.npy", "P (fp16 cast to fp32)")
    dO = load("/tmp/w11dump_2.npy", "dO (fp32)")
    if P is None or dO is None:
        print("RECOMPUTE-ABORT need both dumps")
        return
    P = P.reshape(64, 64)
    dO = dO.reshape(64, 64)
    stats("P", P)
    stats("dO", dO)

    # golden dVnew for the same case (L0 basic fp16, h0=zero, single chunk)
    (q, k, v, g, beta, h0, do), _ = _case_inputs(1, 1, 64, 64, 64, "float16", "zero", False)
    Q, K, V = q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy()
    BETA, G, H0 = beta[0, 0].numpy(), g[0, 0].numpy(), h0[0, 0].numpy()
    dO_g = do[0, 0].numpy()
    _, cache = _np_chunk_fwd(64, 64, 64, Q, K, V, BETA, G, H0, 64)
    p_g, dv_g = cache[0]["p_mat"], cache[0]["p_mat"].T @ dO_g
    stats("golden_P", p_g)
    stats("golden_dVnew", dv_g)

    # host recompute from the card's own operands
    dv_host = P.astype(np.float64).T @ dO.astype(np.float64)
    stats("host_PT_dO", dv_host)

    print(f"CHK-P    rel={rel(P, p_g):.3e}")
    print(f"CHK-dO   rel={rel(dO, dO_g):.3e}")
    print(f"CHK-dVnew host_vs_gold rel={rel(dv_host, dv_g):.3e}")

    card = load("/tmp/w11box_out2.npy", "card WS_DV")
    if card is not None:
        card = card.reshape(64, 64)
        print(f"CHK-card_vs_gold rel={rel(card, dv_g):.3e}")
        print(f"CHK-card_vs_host rel={rel(card, dv_host):.3e}")

    ok = rel(dv_host, dv_g) < 5e-3
    print("VERDICT operands-OK => GEMM itself is wrong (switch to tB form)"
          if ok else "VERDICT operands-BAD => inspect P / dO hand-off chains")
    print("RECOMPUTE-DONE")


if __name__ == "__main__":
    main()

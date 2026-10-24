# _pad_invariance.py -- 尾块 host pad 不变性的本地对照（裁定 5-C / 教材 06 §6.9）
#
# 契约（哑 token）：q 任意、k=0、v=0、g=0、beta=0、dO=0。把 T 补到 C 的整倍数后，
# 前 T 个有效位置的六项梯度必须与不补时逐项一致（纯 CPU fp64，rel < 1e-12）。
#
# 分母与参照：`test_gdn_bwd_v2` 的分块 fp64 参考（`_np_chunk_fwd` / `_np_chunk_bwd`），
# 与内核同源（notes [91] B1-B15）。本脚本不上卡、不 import tilelang。
#
# 已知边界（2026-10-24 实测，不是推测）：对"尾块 pad"而言，六项梯度对 pad 区的
# **任何取值**都是零灵敏度 —— 损失只覆盖有效 token，pad 位于序列末尾，因果性使
# 有效位置的输出与梯度不依赖其后 token。实测把 g_pad 从 0 改成 -0.5，全张量
# （含 pad 行）的六项梯度差都是 0.00e+00。因此负锚不能靠梯度比较自证，改由
# pad 契约检查器承担：把 g 误填 -0.5 时必须报 PADINV-NEGFAIL（证明检查器活着），
# 并附 §6.9 实验 C 的终态证据（g 误填会让终态被错误衰减）。
#
# 打印：PADINV-CASE / PADINV-CONTRACT-* / PADINV T=... item=... rel=... verdict=...
#       PADINV-NEGFAIL / PADINV-NEGSTATE / PADINV worst=... verdict=...
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(_HERE)
if _V2 not in sys.path:          # `python -P` 不预置脚本目录，这里显式补
    sys.path.insert(0, _V2)

from test_gdn_bwd_v2 import _np_chunk_bwd, _np_chunk_fwd, _np_sample  # noqa: E402

C = 64
DK = 64
DV = 64
GATE = 1e-12
ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
# 哑 token 契约：这些字段在 pad 区必须为 0（q 任意，这里取 0）
ZERO_FIELDS = ("k", "v", "do", "g", "beta")


def _sample(T_len, seed=7):
    """单 head 的确定性 fp64 输入（(H,T,*) -> (T,*)）。"""
    Q, K, V, BETA, G, H0, dO = _np_sample(1, T_len, DK, DV, seed)
    return {"q": Q[0], "k": K[0], "v": V[0], "beta": BETA[0], "g": G[0],
            "h0": H0[0], "do": dO[0]}


def _grads(T_len, d):
    """六项梯度：分块 fp64 前向 + 伴随（与内核同源）。"""
    _, cache = _np_chunk_fwd(T_len, DK, DV, d["q"], d["k"], d["v"], d["beta"],
                             d["g"], d["h0"], C)
    return _np_chunk_bwd(T_len, DK, DV, cache, d["do"])


def _pad(x, n_pad, fill=0.0):
    """沿 token 轴（单 head 数组的第 0 维）补 n_pad 行。"""
    if n_pad == 0:
        return np.array(x, np.float64, copy=True)
    tail = np.full((n_pad,) + x.shape[1:], fill, dtype=np.float64)
    return np.concatenate([np.asarray(x, np.float64), tail], axis=0)


def _padded_inputs(d, n_pad, g_fill=0.0):
    """按哑 token 契约补齐 token 轴字段；h0 与序列长度无关，不参与 pad。"""
    out = {name: _pad(d[name], n_pad) for name in d if name != "h0"}
    out["g"] = _pad(d["g"], n_pad, g_fill)
    out["h0"] = d["h0"]
    return out


def _contract(T_len, padded, n_pad):
    """pad 区契约检查：k/v/do/g/beta 必须全 0；返回违规 {字段: peak}。"""
    bad = {}
    for name in ZERO_FIELDS:
        tail = padded[name][-n_pad:] if n_pad else padded[name][:0]
        peak = float(np.abs(tail).max()) if tail.size else 0.0
        if peak != 0.0:
            bad[name] = peak
    if bad:
        detail = ", ".join(f"{k} peak={v:.3e}" for k, v in bad.items())
        print(f"PADINV-CONTRACT-FAIL T={T_len} pad 区违约: {detail}")
    else:
        print(f"PADINV-CONTRACT-OK T={T_len} pad 区 k/v/do/g/beta 全 0")
    return bad


def _final_state(cache):
    """最后一块的递推结果 H_out = exp(gam_last) H_in + Vnew^T Kt。"""
    c = cache[-1]
    return c["glast"] * c["hin"] + c["vnew"].T @ c["kt"]


def _compare(T_len, ref, got):
    """有效前 T 行逐项 rel（dq/dk/dv 按行、dbeta/dg 按 token；dh0 全量）。"""
    worst, ok = 0.0, True
    for i, name in enumerate(ITEMS):
        r = np.asarray(ref[i], np.float64)
        g = np.asarray(got[i], np.float64)
        if name != "dh0":
            g = g[:T_len]
        den = max(float(np.abs(r).max()), 1e-30)
        rel = float(np.abs(g - r).max()) / den
        passed = rel < GATE
        ok &= passed
        worst = max(worst, rel)
        print(f"PADINV T={T_len} item={name:<5} rel={rel:.3e} "
              f"verdict={'PASS' if passed else 'FAIL'}")
    return worst, ok


def run_case(T_len, seed=7):
    """T -> 整倍数：有效前 T 行的六梯度逐项对照 + pad 区契约检查。"""
    T_pad = -(-T_len // C) * C
    n_pad = T_pad - T_len
    print(f"PADINV-CASE T={T_len} T_pad={T_pad} pad={n_pad} C={C} DK={DK} DV={DV}")
    d = _sample(T_len, seed)
    ref = _grads(T_len, d)
    padded = _padded_inputs(d, n_pad)
    clean = not _contract(T_len, padded, n_pad)
    got = _grads(T_pad, padded)
    worst, ok = _compare(T_len, ref, got)
    return worst, ok and clean


def neg_anchor(T_len, seed=7):
    """负锚：g 的 pad 值误填 -0.5，契约检查器必须报 PADINV-NEGFAIL。

    梯度对照抓不到这一违例（见文件头"已知边界"），所以负锚由契约检查器承担；
    再附 §6.9 实验 C 的终态证据：误填 g 会让终态被错误衰减。
    """
    T_pad = -(-T_len // C) * C
    n_pad = T_pad - T_len
    d = _sample(T_len, seed)
    padded = _padded_inputs(d, n_pad, g_fill=-0.5)
    bad = _contract(T_len, padded, n_pad)
    caught = "g" in bad
    if caught:
        print(f"PADINV-NEGFAIL T={T_len} g_pad=-0.5 被契约检查捕获"
              f"（peak={bad['g']:.3e}）")
    else:
        print(f"PADINV-NEGMISS T={T_len} g_pad=-0.5 未被捕获 —— 检查器失效")
    _, cache_bad = _np_chunk_fwd(T_pad, DK, DV, padded["q"], padded["k"],
                                 padded["v"], padded["beta"], padded["g"],
                                 d["h0"], C)
    padded = _padded_inputs(d, n_pad)
    _, cache_ok = _np_chunk_fwd(T_pad, DK, DV, padded["q"], padded["k"],
                                padded["v"], padded["beta"], padded["g"],
                                d["h0"], C)
    st_bad, st_ok = _final_state(cache_bad), _final_state(cache_ok)
    srel = float(np.abs(st_bad - st_ok).max()) / max(float(np.abs(st_ok).max()), 1e-30)
    print(f"PADINV-NEGSTATE T={T_len} final_state rel={srel:.3e}"
          f"（g_pad=-0.5 使终态被错误衰减，§6.9 实验 C）")
    return caught and srel > 1e-6


def main():
    worst, ok = 0.0, True
    for T_len in (100, 130):
        w, case_ok = run_case(T_len)
        worst = max(worst, w)
        ok &= case_ok
        ok &= neg_anchor(T_len)
    print(f"PADINV worst={worst:.3e} gate={GATE:.0e} "
          f"verdict={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

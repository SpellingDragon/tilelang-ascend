# _relay_attr.py -- 归因矩阵：H>1 的 rel≈0.1 到底是"head 数""chunk 数"还是"H_in 错"
#
# 已知（notes [142]）：
#   * H=1、T=64/256 原生路径六项 ≈5e-4（example 实测）；
#   * H=16、T=512（8 chunk）原生路径五项 0.086~0.116，**只有 dh0 达标**。
#
# 指纹：`dh0 = Delta'_0`，它经 `wm_n`（由 A 得出）与 `dvnew` 构造，**不经过 H_in**；
# 而 dq/dk/dv/dbeta/dg 都要经 `Hc`（vnew / qh / de1）。所以"五项坏 + dh0 好"指向
# **H_in（v2 `scan1` 的输出）在 H>1 或更多 chunk 时不准**，而非 Delta' 递推坏。
#
# 本探针把三件事一次问清（全部同一份输入、同一份 golden）：
#   (H=1 ,T=512)  原生        —— 只变 chunk 数（对照 example 的 ≤4 chunk）
#   (H=1 ,T=512)  golden-H    —— 若变好：H=1 下 H_in 也已经不准
#   (H=16,T=256)  原生        —— 只变 head 数
#   (H=16,T=512)  原生        —— 两者都变
#   (H=16,T=512)  golden-H    —— 若六项全部变好，凶手锁定 H_in
#
# 注意：`test_gdn_bwd_v3.py` 的 golden-H 分支只建 `[None, None]`（H=1 专用），
# 在 H>1 上用不了；本支按 head 逐个建再 stack，才能测 H=16。
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from gdn_bwd_v3 import gdn_bwd_v3_relay
from example_gdn_bwd import _np_chunk_bwd, _np_chunk_fwd

ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
# 已排除：H 数、chunk 数、golden H_in、golden 本身（golden 互查 1e-15）、h0、探针方法学
# （本支在 DK=64 两档都复现了 harness 的 ~1e-3）。
# 剩下：**DK 与 DV 各自**。DV=128 会打开 scan1 的 bv_num=2 与 dkt 的 k_mm 分支；
# DK=128 会打开 solve_a 的 bk_num=2。三行单变量拆开。
CASES = (
    (1, 256, 128, 64, False),  # 只 DK=128
    (1, 256, 64, 128, False),  # 只 DV=128
    (1, 256, 128, 128, False), # 两者都 128（已知 5%~12%）
    (1, 512, 128, 128, False), # DoD 形，留作基准
)


def _build(H, T_len, B, DK, DV):
    torch.manual_seed(7)
    dt = torch.float16
    rq = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    q32 = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rk = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k32 = rk / rk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    g32 = -torch.randn(B, H, T_len).abs() * 0.5 - 0.05
    beta32 = torch.sigmoid(torch.randn(B, H, T_len))
    h032 = torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1
    do32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    # dtype 门与 `_realshape_probe.py` 对齐：q/k/v/h0/do=fp16；g/beta=fp32
    cpu = [q32.to(dt), k32.to(dt), v32.to(dt), g32, beta32, h032.to(dt), do32.to(dt)]
    return cpu, [x.numpy() for x in cpu]


def _golden(npc, B, H, T_len, DK, DV, C):
    """逐 head 跑 fp64 golden，返回 (cache 列表, 六项 ref)。H>1 也能用。"""
    qn, kn, vn, gn, betan, h0n, don = npc
    caches, refs = {}, {}
    for b in range(B):
        for h in range(H):
            _o, cache = _np_chunk_fwd(T_len, DK, DV,
                                      qn[b, h].astype(np.float64), kn[b, h].astype(np.float64),
                                      vn[b, h].astype(np.float64), betan[b, h].astype(np.float64),
                                      gn[b, h].astype(np.float64), h0n[b, h].astype(np.float64), C)
            caches[(b, h)] = cache
            refs[(b, h)] = _np_chunk_bwd(T_len, DK, DV, cache, don[b, h].astype(np.float64))
    return caches, refs


def _golden_h(caches, B, H, T_len, C, dev):
    """把每个 head 的 H_in 按 chunk 堆成 [B,H,NC,DV,DK] fp16（替换 scan1 的输出）。"""
    nc = -(-T_len // C)
    rows = []
    for b in range(B):
        for h in range(H):
            rows.append(np.stack([np.asarray(caches[(b, h)][i]["hin"], np.float64)
                                  for i in range(nc)]).astype(np.float16))
    return torch.from_numpy(np.stack(rows).reshape(B, H, nc, *rows[0].shape[1:])).to(dev)


def one(H, T_len, DK, DV, use_golden_h, B=1, C=64):
    cpu, npc = _build(H, T_len, B, DK, DV)
    dev = "npu"
    gpu = [x.to(dev) for x in cpu]
    caches, refs = _golden(npc, B, H, T_len, DK, DV, C)
    _hs = _golden_h(caches, B, H, T_len, C, dev) if use_golden_h else None

    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = gdn_bwd_v3_relay(*gpu, chunk_size=C, hs_override=_hs)
    torch.npu.synchronize()
    card_s = time.perf_counter() - t0
    got = [x.float().cpu() for x in out[:6]]

    mx = {n: 0.0 for n in ITEMS}
    gm = {n: 0.0 for n in ITEMS}
    for b in range(B):
        for h in range(H):
            ref = refs[(b, h)]
            for i, n in enumerate(ITEMS):
                r = np.abs(ref[i])
                gm[n] = max(gm[n], float(r.max()) if r.size else 0.0)
                d = np.abs(got[i][b, h].numpy().astype(np.float64) - ref[i])
                mx[n] = max(mx[n], float(d.max()) if d.size else 0.0)

    rels = {n: mx[n] / (gm[n] + 1e-30) for n in ITEMS}
    tag = "goldenH" if use_golden_h else "native "
    worst = max(rels.items(), key=lambda kv: kv[1])
    print(f"ATTR H={H:<2} T={T_len:<5} DK={DK:<3} DV={DV:<3} chunks={T_len // C:<2} {tag} "
          f"card_s={card_s:6.1f} worst={worst[0]}{worst[1]:.2e} "
          + " ".join(f"{n}={rels[n]:.2e}" for n in ITEMS), flush=True)
    return rels


def main():
    print("ATTR start（DK / DV 单变量拆分）", flush=True)
    for H, T_len, DK, DV, use_golden_h in CASES:
        try:
            one(H, T_len, DK, DV, use_golden_h)
        except Exception as exc:                            # noqa: BLE001
            print(f"ATTR H={H} T={T_len} DK={DK} DV={DV} goldenH={int(use_golden_h)} RAISED "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}", flush=True)
    print("ATTR-DONE", flush=True)


if __name__ == "__main__":
    main()

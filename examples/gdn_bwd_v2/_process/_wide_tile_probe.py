# _wide_tile_probe.py -- 宽 tile 的核级对照：把"宽度 128"与"输出 dtype"拆开
#
# 先记约束（本轮教训）：**探针必须有已知-good 对照臂**，否则 FAIL 分不清是产品错
# 还是探针错。本文件所有 CONTROL 行都是卡上已实测的格（notes [99] 及 v3t 的 64 形）。
#
# 已排除的假设（都有数据，不重复花钱）：
#   * K=128 的收缩不准 —— 伪。[99] 实测 `MM C=64 M=128 N=64 rel=3.330e-04 PASS`，
#     而 M 就是收缩长度。写这个假设前没查自己的笔记，已作废。
#   * H 数 / chunk 数 / H_in / golden / h0 / 片上预算 —— 见 notes [145][146][149]。
#
# 现在还剩下的可分辨事实是两组：
#   A) `gdn_v3_mm` 只测过 (64,64)(128,64)(64,128)，**缺 (128,128)** —— 而 DoD 形正是 (128,128)。
#   B) `dh0` 由 `ta32` 在 (M=DV=128, N=DK=128) 上算出，实测 2.4e-3 通过；而五项失败的链路里
#      凡"宽度 128 + **fp16 输出**"的都在链上（dv 的 ta、dq 的 k_dq、dbeta 的 dw=mm、dg 的 qh=tb）。
#      ⇒ 怀疑落在"宽 128 的 fp16 L0C→GM 拷贝"，而不是收缩长度。
#
# `ta` 与 `ta32` 是同构 kernel，**只差输出 dtype**（一个 fp16、一个 fp32 accum）。
# 同输入、同 (M,N) 上二者一分叉，就锁定输出 dtype 这一层，不必再猜。
#
# 判读规则（先写死，防止事后解释）：
#   1. CONTROL 任一 FAIL        ⇒ 本支失效，结果全部作废（先修量具）
#   2. mm(128,128) FAIL 其余 PASS ⇒ 双宽 L0C 本身
#   3. ta(64,128) FAIL 而 ta32(64,128) PASS ⇒ fp16 输出拷贝
#   4. 全部 PASS                 ⇒ v3 kernel 清白，转 v2 的两个核内循环（pab 的 scan1/solve_a）
#
# 打印：CONTROL name=.. rel=.. PASS|FAIL ；WIDE name=.. rel=.. PASS|FAIL
import numpy as np
import torch
import torch_npu  # noqa: F401

from gdn_bwd_v3 import gdn_v3_mm, gdn_v3_raw, gdn_v3_ta, gdn_v3_ta32, gdn_v3_tb

GATE = 5e-3


def _rel(o, ref):
    got = o.float().cpu().numpy().astype(np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.abs(got - r).max()) / (float(np.abs(r).max()) + 1e-30)


def _rep(tag, name, rel):
    print(f"{tag} name={name:<16} rel={rel:.3e} "
          f"{'PASS' if rel < GATE else 'FAIL'}", flush=True)
    return rel < GATE


def mm(C, M, N, tag="WIDE"):
    torch.manual_seed(7)
    B, H = 1, 2
    a = torch.randn(B, H, C, M, dtype=torch.float16)
    x = torch.randn(B, H, M, N, dtype=torch.float16)
    o = gdn_v3_mm(B, H, C, M, N)(a.npu(), x.npu())
    torch.npu.synchronize()
    return _rep(tag, f"mm C{C} M{M} N{N}", _rel(o, a.double().numpy() @ x.double().numpy()))


def ta_like(fn, C, M, N, tag, fp32_out):
    torch.manual_seed(7)
    B, H = 1, 2
    a = torch.randn(B, H, C, M, dtype=torch.float16) * 0.5
    b = torch.randn(B, H, C, N, dtype=torch.float16) * 0.5
    o = fn(B, H, C, M, N)(a.npu(), b.npu())
    torch.npu.synchronize()
    ref = a.double().numpy().transpose(0, 1, 3, 2) @ b.double().numpy()
    nm = "ta32" if fp32_out else "ta"
    return _rep(tag, f"{nm} C{C} M{M} N{N}", _rel(o, ref))


def tb(C, M, N, tag):
    """gdn_v3_tb(B,H,C,M,N): O = A @ Bm^T, **A:[C,N], Bm:[M,N]** -> [C,M]

    注意参数名与形状是**交叉**的（A 的最后一维是 N 不是 M）；按直觉造 A:[C,M]
    会得到一个形状对不上或参照错的假结果。本文件第一版就是这么写错的。
    """
    torch.manual_seed(7)
    B, H = 1, 2
    a = torch.randn(B, H, C, N, dtype=torch.float16) * 0.5      # [C,N]
    bm = torch.randn(B, H, M, N, dtype=torch.float16) * 0.5     # [M,N]
    o = gdn_v3_tb(B, H, C, M, N)(a.npu(), bm.npu())             # [C,M]
    torch.npu.synchronize()
    ref = a.double().numpy() @ bm.double().numpy().transpose(0, 1, 3, 2)
    return _rep(tag, f"tb C{C} M{M} N{N}", _rel(o, ref))


def raw(DK, tag):
    """gdn_v3_raw(B,H,C,DK): 原始 QK^T，[C,DK]x[C,DK]^T -> [C,C]"""
    torch.manual_seed(7)
    B, H, C = 1, 2, 64
    qq = torch.randn(B, H, C, DK, dtype=torch.float16) * 0.5
    kk = torch.randn(B, H, C, DK, dtype=torch.float16) * 0.5
    o = gdn_v3_raw(B, H, C, DK)(qq.npu(), kk.npu())
    torch.npu.synchronize()
    ref = qq.double().numpy() @ kk.double().numpy().transpose(0, 1, 3, 2)
    return _rep(tag, f"raw C64 DK{DK}", _rel(o, ref))


def main():
    print("WIDE start（CONTROL 臂先跑；任一 CONTROL FAIL 则本支作废）", flush=True)
    C = 64
    ctl = []
    # ---- 已知-good 对照臂（卡上实测过）----
    ctl.append(mm(C, 64, 64, "CONTROL"))
    ctl.append(ta_like(gdn_v3_ta, C, 64, 64, "CONTROL", False))
    ctl.append(ta_like(gdn_v3_ta32, C, 64, 64, "CONTROL", True))
    ctl.append(tb(C, 64, 64, "CONTROL"))
    ctl.append(raw(64, "CONTROL"))
    # ta32 在 (M=128,N=128) 是 dh0 的实际形状，本轮 dh0 实测 2.4e-3 ⇒ 也当对照（预期 PASS）
    ctl.append(ta_like(gdn_v3_ta32, C, 128, 128, "CONTROL", True))
    if not all(ctl):
        print("WIDE verdict=CONTROL_FAILED（先修量具，下面的数不采信）", flush=True)
        print("WIDE-DONE", flush=True)
        return

    # ---- 待判定的格 ----
    mm(C, 128, 128)                      # A：唯一没测过的双宽格
    ta_like(gdn_v3_ta, C, 64, 128, "WIDE", False)    # B：fp16 输出、宽 128
    ta_like(gdn_v3_ta32, C, 64, 128, "WIDE", True)   # B 的对照：同格 fp32 输出
    ta_like(gdn_v3_ta, C, 128, 128, "WIDE", False)
    ta_like(gdn_v3_ta32, C, 128, 128, "WIDE", True)
    tb(C, 64, 128, "WIDE")                           # dg 的 qh
    tb(C, 128, 128, "WIDE")
    raw(128, "WIDE")                                 # K=DK=128
    print("WIDE-DONE", flush=True)


if __name__ == "__main__":
    main()

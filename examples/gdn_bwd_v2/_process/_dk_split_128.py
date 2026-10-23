# _dk_split_128.py -- 在最差配置上把 dk 拆成四项，看谁把余量吃掉
#
# 依据（T2，带 CONTROL）：H=1、DK=DV=128 下 dk 的 rel 非单调——
#   T=128（2 chunk）8.049e-03 FAIL / T=512（8 chunk）1.321e-03 / T=2048 3.316e-03 / T=8192 4.111e-03
# 且 dbeta 在 H=1/T=8192 也翻过 5e-3（6.116e-03）。两个量都是“过线但余量薄”，
# 不是 10% 级别的系统错——所以要看的是**哪一项在贡献**，而不是“还坏不坏”。
#
# 用 relay 自带的 `GDN_V3_DK_SPLIT=1`：它让 relay 多返回一个 dict，含 dk 的四项
#   q1 = dqkt^T Q        （chunk 内）
#   q2 = (beta E) .* t2  （chunk 内，经 dw = dvnew @ H_in）
#   q3 = th .* (vnew Δ') （跨 chunk）
#   q4 = A_b(dxs, K)     （chunk 内，经 A 的导数）
# 并打印 `DKPART sum_vs_full`（拆分完备性）与每项的 abs_max。
#
# 判读规则（先写死）：
#   * sum_vs_full 明显非 0 ⇒ 拆分不完备，本支作废；
#   * 某一项的 abs_max 与 dk 的 abs_max 同量级 ⇒ 它就是主要来源，去查那一项；
#   * 四项都很小而 dk 大 ⇒ dk 的误差不在四项里（在累加/落盘），转查写入路径。
#
# 打印：DKSPLIT T=.. rel(dk)=.. 以及 relay 自己的 DKPART 行
import os

os.environ.setdefault("GDN_V3_DK_SPLIT", "1")

import torch_npu  # noqa: F401,E402  （不需要 torch 本体；run_relay 内部自带）

from test_gdn_bwd_v3 import run_relay  # noqa: E402


def main():
    print("DKSPLIT start（GDN_V3_DK_SPLIT=1，dk 四项分解）", flush=True)
    # CONTROL：DK=DV=64、T=128 —— 这个配置 dk 的 rel 已知 ~2.4e-3（v3t 实测）。
    # 它在这里跑出同样的量级，才说明“拆分 + 打印”这条链本身是准的。
    for T_len, dim in ((128, 64), (128, 128), (512, 128)):
        tag = "CONTROL" if dim == 64 else "DKSPLIT"
        print(f"{tag} begin T={T_len} DK=DV={dim}", flush=True)
        try:
            run_relay(1, 1, T_len, dim, dim, 64, gate=("dk",), rel_gate=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"{tag} T={T_len} DK={dim} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
    print("DKSPLIT-DONE", flush=True)


if __name__ == "__main__":
    main()

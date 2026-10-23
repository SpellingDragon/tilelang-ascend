# _golden_xcheck.py -- 纯 CPU 交叉核对两套 fp64 golden（不碰 NPU，秒级）
#
# 缘起：仓库里同时存在两套反向 golden，而且不同载体用的是不同那套——
#   A) `_np_chunk_fwd` + `_np_chunk_bwd`：`example_gdn_bwd.py` 内联的那套；
#      我的 `_realshape_acc/_relay_attr/_ws_cmp` 都用它当分母。
#   B) `_np_fwd_exact` + `_np_bwd_adjoint(order="ub")`：`_case_inputs` 内部用的那套；
#      `test_gdn_bwd_v3.run_relay`（也就是 `v3t` 的 ~1e-3）用它当分母。
#
# 两套不一致的话，"卡上 10% 误差"可能只是分母不同，跟 kernel 无关。
# 四支函数都在 `test_gdn_bwd_v2` 里，所以同进程、同输入就能比，不需要卡。
#
# 用法：python3 _process/_golden_xcheck.py [T] [C]
# 打印：GX T=.. C=.. item=.. rel=.. max_abs=..   rel 以 B 的 |g|max 为分母
import sys

import numpy as np

sys.path.insert(0, ".")
from test_gdn_bwd_v2 import (  # noqa: E402
    _np_bwd_adjoint, _np_chunk_bwd, _np_chunk_fwd, _np_fwd_exact, _np_sample,
)

ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    H, DK, DV = 1, 64, 64
    Q, K, V, BETA, G, H0, dO = _np_sample(H, T, DK, DV, 1000)

    a = []
    for h in range(H):
        _o, cache = _np_chunk_fwd(T, DK, DV, Q[h], K[h], V[h], BETA[h], G[h], H0[h], C)
        a.append(_np_chunk_bwd(T, DK, DV, cache, dO[h]))

    _f, HIST, _l = _np_fwd_exact(H, T, DK, DV, Q, K, V, BETA, G, H0)
    b = _np_bwd_adjoint(H, T, DK, DV, Q, K, V, BETA, G, HIST, dO, order="ub")

    print(f"GX T={T} C={C} H={H} DK={DK} DV={DV} chunks={T // C}", flush=True)
    worst = 0.0
    for i, item in enumerate(ITEMS):
        num = 0.0
        den = 0.0
        for h in range(H):
            # `_np_bwd_adjoint` 返回 (dqa,dka,dva,dba,dga,dh0)，每项是 [H,T,*]；
            # 而 `_np_chunk_bwd` 是**单 head**，返回六项。两者索引顺序相反。
            ref = np.asarray(b[i][h], np.float64)
            got = np.asarray(a[h][i], np.float64)
            num = max(num, float(np.abs(got - ref).max()))
            den = max(den, float(np.abs(ref).max()))
        rel = num / (den + 1e-30)
        worst = max(worst, rel)
        print(f"GX T={T} item={item:<5} rel={rel:.3e} max_abs={num:.3e} "
              f"verdict={'SAME' if rel < 5e-3 else 'DIFFER'}", flush=True)
    print(f"GX T={T} worst={worst:.3e} "
          f"verdict={'GOLDENS_AGREE' if worst < 5e-3 else 'GOLDENS_DISAGREE'}",
          flush=True)


if __name__ == "__main__":
    main()

# w11 fix compile gate — 修复版 5 kernel + 真形档 scan1 编译与形态扫描
# 判据：
#   onelem_ub2ub == 0：`copy_ub_to_ub<..., 1>` 全是 1-elem UB->UB DataCopy 形态
#     （w11 P3 实证 507015 触发面；修复后应清零——含 reduce_sum 输出槽等同型排查）
#   SetValue > 0：标量替换生效（scan1/scan2a/scan2b）
#   SetValue/GetValue 在 solve_a 本就存在（broadcast 标量读），非本次判据
# 顺带核对每 kernel 的 bytes 以对位历史 cache（w11b/w11d）。
import re
import sys

sys.path.insert(0, "/tmp")
import torch  # noqa: E402

try:
    import torch_npu  # noqa: E402,F401
except ImportError:  # device-less container: shim npu so the factories can compile
    _orig_device = torch.device

    def _device(s=None, *a, **k):
        if isinstance(s, str) and s.startswith("npu"):
            return _orig_device("cpu")
        return _orig_device(s, *a, **k)

    torch.device = _device

    class _NpuShim:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def current_stream():
            class _S:
                npu_stream = 0

            return _S()

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def set_device(*a, **k):
            pass

    torch.npu = _NpuShim

from gdn_bwd_v2 import (  # noqa: E402
    gdn_bwd_v2_cumsum,
    gdn_bwd_v2_scan1,
    gdn_bwd_v2_scan2a,
    gdn_bwd_v2_scan2b,
    gdn_bwd_v2_solve_a,
)

B, H, T_len, DK, DV = 1, 1, 64, 64, 64
C, BV, BK = 64, 64, 64


def gate(name, make):
    for dt in ("float16", "float32"):
        try:
            src = make(dt).get_kernel_source()
        except Exception as e:  # noqa: BLE001
            print(f"GATE-FAIL {name} {dt} {type(e).__name__}: {str(e)[:140]}")
            continue
        onelems = re.findall(r"copy_ub_to_ub<[^>]*,\s*1>\([^;]*\)", src)
        n_sv = src.count("SetValue")
        n_gv = src.count("GetValue")
        print(f"GATE {name} {dt}: bytes={len(src)} onelem_ub2ub={len(onelems)} "
              f"SetValue={n_sv} GetValue={n_gv}")
        for o in onelems:
            print(f"  ONELEM {name} {dt}: {o.strip()[:130]}")


gate("cumsum", lambda dt: gdn_bwd_v2_cumsum(B, H, T_len, C))
gate("solve_a", lambda dt: gdn_bwd_v2_solve_a(B, H, T_len, DK, C, BK, dtype=dt))
gate("scan1", lambda dt: gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, BV, BK, dtype=dt))
gate("scan2a", lambda dt: gdn_bwd_v2_scan2a(B, H, T_len, DK, DV, C, BV, BK, dtype=dt))
gate("scan2b", lambda dt: gdn_bwd_v2_scan2b(B, H, T_len, DK, DV, C, BV, BK, dtype=dt))
gate("scan1_true", lambda dt: gdn_bwd_v2_scan1(1, 16, 8192, 128, 128, 64, 64, 64, dtype=dt))
print("W11FIX-GATE-DONE")

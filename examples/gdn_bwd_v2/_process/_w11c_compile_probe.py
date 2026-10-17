# W11c compile gate probe with mask-order scan (process file; docker cp to /tmp).
#   docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh
#     >/dev/null 2>&1; export PYTHONPATH=/tlasc:$PYTHONPATH; cd /tmp;
#     timeout 1800 python3 /tmp/w11c_compile_probe.py'
# (bash is mandatory: sh=dash silently skips set_env.sh and fakes bisheng misses.)
#
# Gate 1: all five kernels compile for both dtypes at L0 + scaled true-shape.
# Gate 2 (the 507015 fix): scan the generated .asc for "count-mode context +
# elementwise repeat>1" -- the shape that killed scan1 on-card (Sub inheriting
# count=4096 with repeat=64). All known-good codegen (solve_a/cumsum, and every
# legal count-form site) uses repeat=1 under count; expect bad=0 everywhere once
# tl.ascend_vector_mask_reuse is off.
import re
import sys

sys.path.insert(0, "/tmp")
import torch as _torch

_orig_device = _torch.device


def _device(s=None, *a, **k):
    if isinstance(s, str) and s.startswith("npu"):
        return _orig_device("cpu")
    return _orig_device(s, *a, **k)


_torch.device = _device
if not hasattr(_torch, "npu"):

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

        @staticmethod
        def get_device_name():
            return "w11c-shim"

    _torch.npu = _NpuShim

import gdn_bwd_v2 as m


def mask_scan(src):
    """Return [(line, text)] for elementwise ops run under count-mode mask
    with repeat>1 (the 507015 shape)."""
    mode = None
    bad = []
    for i, l in enumerate(src.splitlines(), 1):
        if "SetMaskCount()" in l:
            mode = "count"
        elif "SetMaskNorm()" in l:
            mode = "norm"
        mm = re.search(r"(Sub|Add|Muls?|Subs?)<\w+, false>\(.*MASK_PLACEHOLDER,\s*(\d+),", l)
        if mm and mode == "count" and int(mm.group(2)) > 1:
            bad.append((i, l.strip()[:100]))
    return bad


C, BV, BK = 64, 64, 64
CONFIGS = [
    ("L0", 1, 2, 256, 64, 64),
    ("TSC", 1, 16, 256, 128, 128),  # scaled true-shape (H16 DK128 DV128)
]

failed = False
for name, B, H, L, DK, DV in CONFIGS:
    for dt in ("float16", "float32"):
        kernels = [
            ("cumsum", lambda: m.gdn_bwd_v2_cumsum(B, H, L, C)),
            ("solve_a", lambda: m.gdn_bwd_v2_solve_a(B, H, L, DK, C, BK, dtype=dt)),
            ("scan1", lambda: m.gdn_bwd_v2_scan1(B, H, L, DK, DV, C, BV, BK, dtype=dt)),
            ("scan2a", lambda: m.gdn_bwd_v2_scan2a(B, H, L, DK, DV, C, BV, BK, dtype=dt)),
            ("scan2b", lambda: m.gdn_bwd_v2_scan2b(B, H, L, DK, DV, C, BV, BK, dtype=dt)),
        ]
        for kn, build in kernels:
            try:
                src = build().get_kernel_source()
                bad = mask_scan(src)
                mark = "OK" if not bad else f"MASK-BAD({len(bad)})"
                print(f"W11C-COMPILE-{name}-{dt}-{kn} {mark} src={len(src)}")
                if bad:
                    failed = True
                    for ln, txt in bad[:3]:
                        print(f"W11C-MASKBAD {kn} L{ln}: {txt}")
            except Exception as e:  # noqa: BLE001 - report and keep the matrix rolling
                failed = True
                print(f"W11C-COMPILE-{name}-{dt}-{kn} FAIL {type(e).__name__}: {e}")

print("W11C-ALL-OK" if not failed else "W11C-FAIL")
sys.exit(1 if failed else 0)

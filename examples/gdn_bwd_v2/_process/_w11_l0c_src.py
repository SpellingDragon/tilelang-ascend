# _w11_l0c_src.py — actual L0C allocation in scan2a's generated code.
#   The budget script reports a static no-reuse bound (196KB); what matters is
#   what the codegen actually emits. Print every emit line mentioning L0C so we
#   can sum the real offsets against the 128KB (131072) A2 limit.
import sys

sys.path.insert(0, "/tmp")
from gdn_bwd_v2 import gdn_bwd_v2_scan2a  # noqa: E402

ker = gdn_bwd_v2_scan2a(1, 1, 64, 64, 64, 64, 64, 64, dtype="float16")
src = ker.get_kernel_source()
lines = src.splitlines()
print(f"SRC lines={len(lines)}")
for i, ln in enumerate(lines):
    low = ln.lower()
    if "l0c" in low or "co1" in low:
        print(f"L{i}: {ln.strip()[:160]}")
print("L0CSRC-DONE")

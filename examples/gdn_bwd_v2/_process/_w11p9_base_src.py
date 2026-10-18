# _w11p9_base_src.py — 可运行基线（9fc72a7b, plain-2D）scan2a 源码 dump
#   用途：与 plan-9 的生成码做跨核 flag 序列对照，判定死锁是否由 cv-sync 引入。
import sys

sys.path.insert(0, "/tmp")
from gdn_bwd_v2_base import gdn_bwd_v2_scan2a  # noqa: E402

ker = gdn_bwd_v2_scan2a(1, 1, 64, 64, 64, 64, 64, 64, dtype="float16")
src = ker.get_kernel_source()
with open("/tmp/p9_base_f16.asc", "w") as f:
    f.write(src)
print(f"BASE lines={len(src.splitlines())}")
print("BASE-DONE")

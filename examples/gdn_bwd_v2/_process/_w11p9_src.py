# _w11p9_src.py — plan-9 生成码审计：T.tile.broadcast 在 scan2a 里的落法
#   输出 /tmp/p9_scan2a_<tag>.asc 全量源码；打印每个 broadcast 命中点 ±8 行、
#   sync/flag 普查、以及 spilling（workspace_ 在 UB 段被引用）的可疑行。
import sys

sys.path.insert(0, "/tmp")
from gdn_bwd_v2 import gdn_bwd_v2_scan2a  # noqa: E402


def dump(dtype, tag):
    ker = gdn_bwd_v2_scan2a(1, 1, 64, 64, 64, 64, 64, 64, dtype=dtype)
    src = ker.get_kernel_source()
    lines = src.splitlines()
    with open(f"/tmp/p9_scan2a_{tag}.asc", "w") as f:
        f.write(src)
    print(f"SRC {tag} lines={len(lines)}")
    idx = [i for i, l in enumerate(lines) if "broadcast" in l.lower()]
    print(f"BCAST {tag} hits={len(idx)}")
    for i in idx:
        print(f"--- {tag} L{i}")
        for j in range(max(0, i - 8), min(len(lines), i + 9)):
            print(f"{tag} {j}: {lines[j].strip()[:200]}")
    for pat in ("set_flag", "wait_flag", "SyncAll", "cross", "asc_broadcast", "copy_ub_to_ub"):
        print(f"CENSUS {tag} {pat}=", sum(1 for l in lines if pat.lower() in l.lower()))


def main():
    dtypes = [a for a in sys.argv[1:] if a in ("float16", "float32")] or ["float16"]
    for d in dtypes:
        dump(d, "f16" if d == "float16" else "f32")
    print("P9SRC-DONE")


if __name__ == "__main__":
    main()

# S7c: simulate AutoInsertCrossCoreSync pairing audit on /tmp/ir_pre_combinecv.txt.
# (process file; docker cp to container /tmp, run with container python3)
# Replicates ascend_combinecv.cc semantics:
#   - sync point = call_extern matching GM_COPY_CFG_INFOS with a workspace* buffer arg
#   - side: copy_gm_to_l1 / copy_l0c_to_gm / atomic_add_l0c_to_gm -> cube,
#           copy_gm_to_ub / copy_ub_to_gm / atomic_add_ub_to_gm -> vec
#   - per workspace: cube_sps[i] must pair vec_sps[i] with opposite is_write
# Prints every mispair (the real pass FATALs on the first).
import re

CFG = [
    ("copy_gm_to_l1", False, "cube"),
    ("copy_l0c_to_gm", True, "cube"),
    ("copy_gm_to_ub", False, "vec"),
    ("copy_ub_to_gm", True, "vec"),
    ("atomic_add_ub_to_gm", True, "vec"),
    ("atomic_add_l0c_to_gm", True, "cube"),
]

sp = {}  # ws -> {"cube": [...], "vec": [...]}
with open("/tmp/ir_pre_combinecv.txt") as fh:
    for lineno, line in enumerate(fh, 1):
        m = re.search(r'call_extern\("handle", "tl::ascend::([A-Za-z0-9_]+)<', line)
        if not m:
            continue
        fn = m.group(1)
        hit = None
        for name, is_write, side in CFG:
            if fn.find(name) != -1:
                hit = (name, is_write, side)
                break
        if hit is None:
            continue
        ws = re.search(r"(workspace_\w+)\.data", line)
        if ws is None:
            continue
        bufs = re.findall(r'tvm_access_ptr\(T\.type_annotation\("[^"]+"\), (\w+)', line)
        sp.setdefault(ws.group(1), {"cube": [], "vec": []})[hit[2]].append(
            (lineno, hit[0], hit[1], "->".join(bufs[:2]))
        )

for ws in sorted(sp):
    for side in ("cube", "vec"):
        for ln, fn, w, bufs in sp[ws][side]:
            print(f"CALL {ws} {side} L{ln} {fn} write={int(w)} {bufs}")

bad = 0
for ws in sorted(sp):
    c, v = sp[ws]["cube"], sp[ws]["vec"]
    if len(c) != len(v):
        bad += 1
        print(f"MISPAIR {ws}: cube {len(c)} vs vec {len(v)}")
        continue
    for i, (cs, vs) in enumerate(zip(c, v)):
        if cs[2] == vs[2]:
            bad += 1
            print(f"MISPAIR {ws}[{i}]: both write={cs[2]}")
print(f"S7C-AUDIT-SIM total_workspaces={len(sp)} mispairs={bad}")

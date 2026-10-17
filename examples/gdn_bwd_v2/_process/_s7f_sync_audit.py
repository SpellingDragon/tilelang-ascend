# S7f sync-point audit (frozen at the fix-2 state, for the S7f verdict):
# CombineCV AutoInsertCrossCoreSync (ascend_combinecv.cc:380) requires each GM
# workspace's access statements to pair up cube-side vs vec-side.  This tool
# statically classifies every T.copy touching a workspace_* buffer:
#   L0C->GM = cube W,  GM->L1 = cube R,  UB->GM = vec W,  GM->UB = vec R
# and reports the per-box statement sequence + strict W-R alternation verdict
# (scan1's one-write-one-read discipline passes; the S7f-E half-block if/else
# double dumps and multi-read boxes do not -- see the S7f battle report).
# Run: python3 examples/gdn_bwd_v2/_s7f_sync_audit.py
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gdn_bwd_v2.py")

tree = ast.parse(open(SRC).read())
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
KERNELS = (
    "gdn_bwd_v2_cumsum",
    "gdn_bwd_v2_solve_a",
    "gdn_bwd_v2_scan1",
    "gdn_bwd_v2_scan2a",
    "gdn_bwd_v2_scan2b",
)


def base_of(node):
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def alloc_scopes(main):
    """buffer name -> scope ('ub' / 'l1' / 'l0c') for on-chip allocs."""
    out = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr.startswith("alloc_"):
                tgt = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                if tgt:
                    out[tgt] = call.func.attr.split("_", 1)[1].lower()
    return out


fail = False
for kname in KERNELS:
    fn = funcs.get(kname)
    if fn is None:
        continue
    main = next(s for s in fn.body if isinstance(s, ast.FunctionDef) and s.name == "main")
    gm = {a.arg for a in main.args.args}
    chip = alloc_scopes(main)
    seqs = {}  # workspace -> [(line, access)]
    for node in ast.walk(main):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "copy" or len(node.args) < 2:
            continue
        s, d = node.args[0], node.args[1]
        sb, db = base_of(s), base_of(d)
        s_in, d_in = sb in gm, db in gm
        ws = None
        if s_in and sb.startswith("workspace_"):
            ws = sb
        elif d_in and db.startswith("workspace_"):
            ws = db
        if ws is None:
            continue
        # classify: side = which engine moves the data
        if d_in and not s_in:  # on-chip -> GM
            side = "cube" if chip.get(sb) == "l0c" else "vec"
            acc = "W"
        elif s_in and not d_in:  # GM -> on-chip
            side = "cube" if chip.get(db) in ("l1", "l0c") else "vec"
            acc = "R"
        else:
            side, acc = "?", "?"
        seqs.setdefault(ws, []).append((node.lineno, side, acc))
    for ws in sorted(seqs):
        seq = sorted(seqs[ws])
        toks = [f"{s[0].upper()}{a}" for _, s, a in seq]  # CW/VR/...
        alt = all(toks[i][1] != toks[i + 1][1] for i in range(len(toks) - 1))
        pair = len(toks) % 2 == 0 and alt and toks[0][1] == "W"
        ncw = sum(1 for t in toks if t == "CW")
        nvr = sum(1 for t in toks if t == "VR")
        nvw = sum(1 for t in toks if t == "VW")
        ncr = sum(1 for t in toks if t == "CR")
        status = "OK" if pair else "MISMATCH"
        if not pair:
            fail = True
        print(
            f"SYNC-AUDIT {kname}::{ws} {status} seq={toks} "
            f"cube(W={ncw},R={ncr}) vec(W={nvw},R={nvr})"
        )

print("SYNC-AUDIT-ALL-OK" if not fail else "SYNC-AUDIT-FAIL")
sys.exit(1 if fail else 0)

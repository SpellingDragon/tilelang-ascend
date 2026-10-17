# S7e process artifact: H1 hard-constraint assertion for gdn_bwd_v2.py.
# H1: every j-contraction goes through T.gemm_v0 (incl. ones-GEMM row/colsum
# rewrites); T.dot / einsum / raw broadcast reductions are forbidden in kernels.
# Prints the gemm_v0 census + the ones-GEMM exemption list.
# Run: python3 examples/gdn_bwd_v2/_s7e_h1_check.py
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gdn_bwd_v2.py")

src = open(SRC).read()
lines = src.splitlines()
fail = False

# 1. forbidden constructs anywhere in the file
for pat, tag in ((r"\bT\.dot\s*\(", "T.dot"), (r"\beinsum\b", "einsum")):
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(lines) if re.search(pat, ln)]
    if hits:
        fail = True
        print(f"H1-CHECK forbidden {tag} hits={hits}")
    else:
        print(f"H1-CHECK forbidden {tag} count=0 OK")

# 2. gemm_v0 census (line numbers, per kernel)
tree = ast.parse(src)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
total = 0
for kname in ("gdn_bwd_v2_cumsum", "gdn_bwd_v2_solve_a", "gdn_bwd_v2_scan1", "gdn_bwd_v2_scan2a", "gdn_bwd_v2_scan2b"):
    fn = funcs.get(kname)
    if fn is None:
        print(f"H1-CHECK gemm_v0 census {kname} MISSING")
        fail = True
        continue
    hits = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "gemm_v0"
        ):
            hits.append(node.lineno)
    total += len(hits)
    print(f"H1-CHECK gemm_v0 census {kname} count={len(hits)} lines={sorted(hits)}")
print(f"H1-CHECK gemm_v0 census TOTAL={total}")

# 3. ones-GEMM exemption list: gemm_v0 calls whose A or B operand is an
#    ones_* buffer (row/colsum rewrite, design.md Sec 5.1)
exempt = []
for node in ast.walk(tree):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "gemm_v0"
        and node.args
    ):
        ops = []
        for a in node.args[:2]:
            if isinstance(a, ast.Name):
                ops.append(a.id)
        if any(o.startswith("ones") for o in ops):
            exempt.append((node.lineno, ops))
print(f"H1-CHECK ones-GEMM exemptions count={len(exempt)} detail={sorted(exempt)}")

print("H1-ALL-OK" if not fail else "H1-FAIL")
sys.exit(1 if fail else 0)

# S7d diagnosis probe -- error B root cause: faithful Python port of
# /tlasc/src/transform/ascend_memory_planning.cc (LivenessAnalysis +
# ReorderKillPoints + ExtendKillIndex + LinearScanAllocator) run on the REAL
# pre-planning IR (captured by monkeypatching tilelang.transform.
# AscendMemoryPlanning with a passthrough recorder).
# Two models compared per case:
#   modelA = faithful   (touch-based interval + kill reorder/extension)
#   modelB = stmt-level (interval = [first touch, last touch], no reorder/extend)
# Calibration check: modelA must reproduce the exact compile-gate FATAL line
# "Memory allocation failed for: <buf> required: N, new memory available: M".
# (process file; docker cp to container /tmp. sandbox forbids host /tmp writes.)
# Usage (bash mandatory):
#   docker cp _s7d_mp_model.py cann910b-l:/tmp/s7d_mp_model.py
#   docker cp gdn_bwd_v2.py cann910b-l:/tmp/gdn_bwd_v2.py
#   docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/
#     set_env.sh; PYTHONPATH=/tlasc python3 /tmp/s7d_mp_model.py <kernel> <case>'
# <kernel> in {scan, scan1, scan2}; <case> in {L0-fp16, TSC-fp16, TSC-fp32}.
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
            return "s7d-shim"

    _torch.npu = _NpuShim

import gdn_bwd_v2 as m
import tvm
import tilelang
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.utils.target import determine_platform

KIND = sys.argv[1] if len(sys.argv) > 1 else "scan"
CASE = sys.argv[2] if len(sys.argv) > 2 else "TSC-fp16"
CASES = {
    "L0-fp16": (1, 2, 256, 64, 64, "float16"),
    "TSC-fp16": (1, 16, 256, 128, 128, "float16"),
    "TSC-fp32": (1, 16, 256, 128, 128, "float32"),
}
B, H, L, DK, DV, DT = CASES[CASE]
C, BV, BK = 64, 64, 64
MEM_LIMITS = {
    "shared.l1": 524032,
    "wmma.matrix_a": 65536,
    "wmma.matrix_b": 65536,
    "wmma.accumulator": 131072,
    "shared.ub": 196352,
}

# ---------------------------------------------------------------- capture ---
captured = {}


def _fake_amp():
    def _apply(mod):
        captured["mod"] = mod
        return mod

    return _apply


tilelang.transform.AscendMemoryPlanning = _fake_amp

impl = getattr(m, f"gdn_bwd_v2_{KIND}").__jit_impl__
prim = impl.func(B, H, L, DK, DV, C, BV, BK, dtype=DT, accum_dtype="float")
platform = determine_platform("auto")
target = tvm.target.Target({"kind": "llvm", "model": "auto"})
mod = tvm.IRModule({prim.attrs["global_symbol"]: prim})
for gvar, f in mod.functions_items():
    mod[gvar] = f.with_attr("npu_platform", platform)
try:
    with tilelang.transform.PassContext(config=dict(impl.pass_configs)):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target, platform)
except Exception as e:  # noqa: BLE001 - later stages may need address_map
    print(f"S7D-MPMODEL-{KIND}-{CASE} pipeline-note {type(e).__name__}: {e}")

mod_c = captured.get("mod")
func = list(mod_c.functions.values())[0] if mod_c is not None else None
if func is None:
    print(f"S7D-MPMODEL-{KIND}-{CASE} NO-IR-CAPTURED")
    sys.exit(2)
print(f"S7D-MPMODEL-{KIND}-{CASE} captured-ir ok body={type(func.body).__name__}")

# ------------------------------------------------------- faithful port ------
from tvm.tir import stmt_functor  # noqa: E402


class Entry:
    __slots__ = ("key", "node", "touched", "offset", "level", "kind")

    def __init__(self, key, node, kind, level):
        self.key = key
        self.node = node
        self.kind = kind
        self.level = level
        self.touched = []
        self.offset = 0


class Port:
    def __init__(self, func):
        self.alloc = {}  # name -> {"scope","size","node"}
        self.all_scopes = {}  # name -> raw scope string (diagnostic)
        self.by_var = {}  # id(var) -> name
        self.linear = []
        self.scope_stack = []
        self.first_use = {}
        self.scope_level = 0
        self.walk_stmt(func.body)

    # -- helpers -------------------------------------------------------------
    def var_name(self, node):
        name = getattr(node, "name", None)
        return name if name and name in self.alloc else None

    def track(self, node):
        name = self.var_name(node)
        if name is None:
            return
        if self.scope_stack:
            self.scope_stack[-1].append(name)
        self.first_use.setdefault(name, len(self.linear))

    def touches_of(self, node):
        acc = []

        def f(n):
            if isinstance(n, tvm.tir.Var):
                nm = self.var_name(n)
                if nm:
                    acc.append(nm)
            elif isinstance(n, tvm.tir.BufferLoad):
                nm = self.var_name(n.buffer.data)
                if nm:
                    acc.append(nm)

        stmt_functor.post_order_visit(node, f)
        return acc

    # -- statements ----------------------------------------------------------
    def walk_stmt(self, s):
        t = type(s)
        if t is tvm.tir.Allocate or t is tvm.tir.AllocateConst:
            name = s.buffer_var.name
            scope = getattr(s.buffer_var.type_annotation, "storage_scope", "")
            self.all_scopes[name] = scope
            if scope in MEM_LIMITS:
                self.alloc[name] = {
                    "scope": scope,
                    "size": self.calc_size(s),
                    "node": s.buffer_var,
                }
            self.by_var[id(s.buffer_var)] = name
            self.walk_stmt(s.body)
        elif t is tvm.tir.BufferStore or t is tvm.tir.Evaluate:
            self.scope_stack.append([])
            acc = self.touches_of(s)
            # C++ TrackBufferTouch: first_use at current linear size (entry not
            # pushed yet) + touches accumulate in the open store scope only
            for nm in acc:
                self.first_use.setdefault(nm, len(self.linear))
            e = Entry(id(s), s, "store" if t is tvm.tir.BufferStore else "eval",
                      self.scope_level)
            e.touched = acc
            self.scope_stack.pop()
            if e.touched:
                self.linear.append(e)
        elif t in (tvm.tir.For, tvm.tir.While, tvm.tir.IfThenElse,
                   tvm.tir.AttrStmt, tvm.tir.LetStmt, tvm.tir.AssertStmt):
            self.visit_new_scope(s)
        elif t is tvm.tir.SeqStmt:
            for x in s.seq:
                self.walk_stmt(x)
        elif t is tvm.tir.DeclBuffer:
            self.walk_stmt(s.body)
        else:
            raise RuntimeError(f"unhandled stmt {t}")

    def visit_new_scope(self, s):
        if isinstance(s, tvm.tir.For):
            self.scope_level += 1
            level = self.scope_level
        else:
            level = self.scope_level
        self.scope_stack.append([])
        begin = Entry(id(s), s, "marker", level)
        bi = len(self.linear)
        self.linear.append(begin)
        # expr fields first (C++ default visitors), then body
        for field in self.expr_fields(s):
            for nm in self.touches_of(field):
                self.scope_stack[-1].append(nm)
                self.first_use.setdefault(nm, len(self.linear))
        self.walk_stmt(s.body) if hasattr(s, "body") else None
        touched = self.scope_stack.pop()
        end = Entry(id(s), s, "marker", level)
        ei = len(self.linear)
        end.touched = touched
        end.offset = bi - ei
        self.linear.append(end)
        self.linear[bi].offset = ei - bi
        if isinstance(s, tvm.tir.For):
            self.scope_level -= 1

    @staticmethod
    def expr_fields(s):
        out = []
        for f in ("min", "extent", "condition", "value", "node"):
            v = getattr(s, f, None)
            if isinstance(v, tvm.tir.PrimExpr):
                out.append(v)
        return out

    def calc_size(self, s):
        n = 1
        for e in s.extents:
            n *= int(e)
        dt = tvm.runtime.DataType(s.dtype)
        size = n * (int(dt.bits) // 8) * int(dt.lanes)
        return (size + 31) // 32 * 32


p = Port(func)

# -- scope diagnostic (S7d: raw storage_scope strings seen at Allocate) ----
from collections import Counter  # noqa: E402

_sc = Counter(p.all_scopes.values())
print(f"S7D-MPMODEL-{KIND}-{CASE} scope-distribution {dict(_sc)} n_alloc={len(p.all_scopes)}")
print(
    f"S7D-MPMODEL-{KIND}-{CASE} walk-stats n_alloc_tracked={len(p.alloc)} "
    f"n_linear={len(p.linear)} n_first_use={len(p.first_use)} "
    f"n_touch_names={len(set(p.first_use) | {b for e in p.linear for b in e.touched})}"
)

# -- liveness (LivenessAnalysis + ReorderKillPoints) -------------------------
from collections import defaultdict  # noqa: E402

event = defaultdict(lambda: {"gen": [], "kill": []})  # entry index -> events
seen = set()
for i in range(len(p.linear) - 1, -1, -1):
    for b in p.linear[i].touched:
        if b not in seen:
            seen.add(b)
            event[i]["kill"].append(b)
for i, e in enumerate(p.linear):
    for b in e.touched:
        if p.first_use.get(b) == i:
            event[i]["gen"].append(b)


def entry_level(idx):
    return p.linear[idx].level


gen_of = {}
for k, ev in event.items():
    for b in ev["gen"]:
        gen_of[b] = k

# snapshot BEFORE ReorderKillPoints: modelB must see original kill points
from copy import deepcopy  # noqa: E402

event_raw = deepcopy({k: dict(v) for k, v in event.items()})

# ReorderKillPoints (faithful incl. gen_level-1 fallthrough to seq end)
displaced = {}  # buf -> (old_idx, new_idx, gen_level, kill_level)
gen_kill_seq = [i for i in range(len(p.linear))
                if event[i]["gen"] or event[i]["kill"]]
for k, ev in list(event.items()):
    if not ev["kill"]:
        continue
    kill_level = entry_level(k)
    visited = set()
    kept = []
    for b in ev["kill"]:
        gk = gen_of.get(b)
        if gk is not None and kill_level > entry_level(gk) and b not in visited:
            gen_level = entry_level(gk)
            pos = gen_kill_seq.index(k)
            target = None
            for j in range(pos, len(gen_kill_seq)):
                nxt = (gen_kill_seq[j + 1]
                       if j + 1 < len(gen_kill_seq) else None)
                if nxt is None or entry_level(nxt) == gen_level - 1:
                    target = gen_kill_seq[j]
                    break
            if target is not None:
                event[target]["kill"].append(b)
                visited.add(b)
                displaced[b] = (k, target, gen_level, kill_level)
            else:
                kept.append(b)
        else:
            kept.append(b)
    ev["kill"] = kept

# -- intervals (FindEventIndex + ExtendKillIndex) ---------------------------
loop_intervals = []
for i, e in enumerate(p.linear):
    if e.offset > 0 and isinstance(e.node, (tvm.tir.For, tvm.tir.While)):
        loop_intervals.append((i, i + e.offset))
touch_idx = defaultdict(set)
for i, e in enumerate(p.linear):
    for b in e.touched:
        touch_idx[b].add(i)


def find_event(buf, use_gen, evmap=None):
    evmap = event if evmap is None else evmap
    for k, ev in evmap.items():
        vec = ev["gen"] if use_gen else ev["kill"]
        if buf in vec:
            return k
    return -1


def extend_kill(buf, g, k):
    if k < 0:
        return k
    ext = k
    for lo, hi in loop_intervals:
        if g < lo and lo <= k < hi:
            ts = sorted(t for t in touch_idx[buf] if t > lo)
            if ts and ts[0] < hi:
                ext = max(ext, hi)
    return ext


def build(model_faithful):
    out = {}
    evmap = event if model_faithful else event_raw
    for name, info in p.alloc.items():
        g = find_event(name, True, evmap)
        k = find_event(name, False, evmap)
        if g < 0 or k < 0:
            continue
        if model_faithful:
            k = extend_kill(name, g, k)
        out[name] = (g, k, info["size"], info["scope"])
    return out


def run_alloc(model_faithful, scope="shared.ub"):
    ivs = {n: v for n, v in build(model_faithful).items() if v[3] == scope}
    items = sorted(ivs.items(), key=lambda kv: kv[1][0])
    limit = MEM_LIMITS[scope]
    nxt = 0
    active = []  # (end, offset, size, name)
    frees = []
    allocs = {}
    for name, (g, k, sz, _) in items:
        active = [a for a in active]
        frees2 = []
        exp = [a for a in active if a[0] < g]
        active = [a for a in active if a[0] >= g]
        frees2 = [(a[1], a[2]) for a in exp]
        frees.extend(frees2)
        frees.sort()
        merged = []
        for blk in frees:
            if merged and merged[-1][0] + merged[-1][1] >= blk[0]:
                merged[-1] = (merged[-1][0], max(merged[-1][0] + merged[-1][1],
                                                blk[0] + blk[1]) - merged[-1][0])
            else:
                merged.append(blk)
        frees = merged
        new_off = (nxt + 31) // 32 * 32
        if new_off + sz <= limit:
            off = new_off
            nxt = new_off + sz
        else:
            off = -1
            for blk in frees:
                ao = (blk[0] + 31) // 32 * 32
                if blk[1] - (ao - blk[0]) >= sz:
                    off = ao
                    break
            if off < 0 and frees and frees[-1][0] + frees[-1][1] == nxt:
                nxt = limit
                frees[-1] = (frees[-1][0], limit - frees[-1][0])
                ao = (frees[-1][0] + 31) // 32 * 32
                if frees[-1][1] - (ao - frees[-1][0]) >= sz:
                    off = ao
            if off < 0:
                return (False, name, sz, limit - nxt, nxt, allocs, items)
            keep = []
            for blk in frees:
                if blk[0] <= off < blk[0] + blk[1]:
                    if off > blk[0]:
                        keep.append((blk[0], off - blk[0]))
                    if off + sz < blk[0] + blk[1]:
                        keep.append((off + sz, blk[0] + blk[1] - off - sz))
                else:
                    keep.append(blk)
            frees = keep
        allocs[name] = off
        active.append((k, off, sz, name))
    return (True, None, None, None, nxt, allocs, items)


for tag, faithful in (("modelA-faithful", True), ("modelB-stmt-level", False)):
    ok, name, sz, avail, nxt, allocs, items = run_alloc(faithful)
    if ok:
        print(f"S7D-MPMODEL-{KIND}-{CASE} {tag} OK usage={nxt}/{MEM_LIMITS['shared.ub']}")
    else:
        print(f"S7D-MPMODEL-{KIND}-{CASE} {tag} "
              f"Memory allocation failed for: {name} required: {sz}, "
              f"new memory available: {avail}")

# -- interval table + statement-reach evidence ------------------------------
print(f"S7D-MPMODEL-{KIND}-{CASE} ---- UB interval table (modelA) ----")
fa = build(True)
for name, (g, k, sz, sc) in sorted(fa.items(), key=lambda kv: kv[1][0]):
    if sc != "shared.ub":
        continue
    reach = (min(touch_idx[name]), max(touch_idx[name]))
    marks = []
    if name in displaced:
        old, new, gl, kl = displaced[name]
        marks.append(f"KILL-DISPLACED old={old} new={new} genL={gl} killL={kl}")
    if k > reach[1]:
        marks.append(f"claimed-kill {k} > stmt-reach {reach[1]}")
    print(f"  {name}: [{g}, {k}], size={sz}, stmt-reach=[{reach[0]}, {reach[1]}] "
          + " ".join(marks))

print(f"S7D-MPMODEL-{KIND}-{CASE} ---- IR excerpt: displaced buffers' touches ----")
for name in displaced:
    if name not in fa or fa[name][3] != "shared.ub":
        continue
    idxs = sorted(touch_idx[name])
    for i in (idxs[0], idxs[-1]):
        snip = str(p.linear[i].node).splitlines()[0][:96]
        print(f"  {name} touch@{i}: {snip}")
print(f"S7D-MPMODEL-{KIND}-{CASE} DONE")

# S7c process file: host-side simulator of ascend_memory_planning.cc
# LinearScanAllocator (memory_auto_plan path) -- faithful port of:
#   /tlasc/src/transform/ascend_memory_planning.cc lines 667-910
# Purpose: reproduce round-2 compile-gate OOM errors exactly (calibration),
# then search a buffer-split set that passes all 4 probe configs offline,
# so the kernel file is edited once with a validated plan.
# Usage: python3 _s7c_mp_sim.py


def align_up(x, a=32):
    return (x + a - 1) // a * a


class Sim:
    """One scope (UB / L0C / ...). Port of LinearScanAllocator.allocate()."""

    def __init__(self, limit):
        self.limit = limit
        self.next_new = 0
        self.free = []  # [offset, size]

    def merge(self):
        if not self.free:
            return
        self.free.sort()
        merged = [self.free[0][:]]
        for b in self.free[1:]:
            last = merged[-1]
            if last[0] + last[1] >= b[0]:
                end = max(last[0] + last[1], b[0] + b[1])
                last[1] = end - last[0]
            else:
                merged.append(b[:])
        self.free = merged

    def find_reusable(self, size):
        self.free.sort()
        for b in self.free:
            if b[1] >= size:
                aligned = align_up(b[0])
                avail = b[1] - (aligned - b[0])
                if avail >= size:
                    return aligned
        if self.free:
            last = self.free[-1]
            if last[0] + last[1] == self.next_new:
                self.next_new = self.limit  # tail extension (mutates!)
                last[1] = self.limit - last[0]
                aligned = align_up(last[0])
                avail = last[1] - (aligned - last[0])
                if avail >= size:
                    return aligned
        return -1

    def remove_from_free(self, offset, size):
        for i, b in enumerate(self.free):
            if b[0] <= offset < b[0] + b[1]:
                bs, bz = b
                del self.free[i]
                if offset > bs:
                    self.free.append([bs, offset - bs])
                end = offset + size
                if end < bs + bz:
                    self.free.append([end, bs + bz - end])
                return

    def allocate(self, intervals):
        """intervals: [(name, gen, kill, size)] -> (ok, fail_name, available, allocs)"""
        ivs = sorted(intervals, key=lambda t: t[1])
        active = []  # (kill, name, off, size)
        allocs = {}
        for name, gen, kill, size in ivs:
            while active:
                a = min(active, key=lambda t: t[0])
                if a[0] < gen:
                    active.remove(a)
                    self.free.append([a[2], a[3]])
                else:
                    break
            self.merge()
            new_off = align_up(self.next_new)
            if new_off + size <= self.limit:
                off = new_off
                self.next_new = new_off + size
            else:
                off = self.find_reusable(size)
                if off == -1:
                    return False, name, self.limit - self.next_new, allocs
                self.remove_from_free(off, size)
            allocs[name] = off
            active.append((kill, name, off, size))
        return True, None, None, allocs


# ---------------------------------------------------------------------------
# buffer span tables.  span = (gen_stmt, kill_stmt) on the gdn_bwd_v2.py
# statement timeline (line numbers used as statement ids, stable enough).
# ---------------------------------------------------------------------------

# L0C spans -- round-2 state (post the 5 L0C splits dw1/u01/wh1/dp1/rs2)
L0C_ROUND2 = [
    ("dw1", 416, 417, "CDK"), ("u01", 433, 434, "CBV"), ("wh1", 443, 444, "CBV"),
    ("hout", 452, 455, "BVDK"),
    ("dw", 508, 664, "CDK"), ("kk", 517, 519, "CC"), ("dp1", 520, 524, "CC"),
    ("u0", 566, 568, "CBV"), ("wh", 570, 572, "CBV"), ("qh", 579, 581, "CBV"),
    ("rs", 587, 656, "C16"), ("dvnew", 596, 599, "CBV"), ("dp", 601, 698, "CC"),
    ("dq", 608, 819, "CDK"), ("dh", 609, 618, "BVDK"), ("dkt", 622, 791, "CDK"),
    ("t1", 642, 643, "CBV"), ("da", 661, 716, "CC"), ("t2", 672, 681, "CDK"),
    ("rs2", 687, 788, "C16"), ("dk", 714, 823, "CDK"), ("dsum", 721, 722, "CC"),
    ("dx", 722, 724, "CC"),
]

# UB spans -- round-2 state
UB_ROUND2 = [
    ("hin", 366, 625, "BVDK"), ("io_hv", 367, 639, "io_hv"),
    ("g", 373, 488, "C1"), ("be", 376, 745, "C1"), ("e", 378, 811, "C1"),
    ("glast", 381, 633, "S8"), ("th", 386, 799, "C1"), ("bee", 394, 676, "C1"),
    ("xk", 398, 826, "CDK"), ("io_xk", 399, 828, "io_xk"),
    ("bq", 425, 606, "CBV"), ("io_bq", 426, 660, "io_bq"),
    ("wh", 445, 648, "CBV"), ("dh", 455, 639, "BVDK"),
    ("gm", 487, 760, "CC"), ("k2", 493, 782, "CDK"), ("kk", 519, 747, "CC"),
    ("pk", 522, 706, "CC"), ("mul", 524, 770, "CC"), ("msk", 525, 729, "CC"),
    ("io_cc", 530, 771, "io_cc"),
    ("dE", 534, 811, "C1"), ("dgam", 535, 817, "C1"), ("tmp1", 536, 829, "C1"),
    ("dbe", 537, 696, "C1"), ("dq", 538, 821, "CDK"), ("dk", 539, 827, "CDK"),
    ("dhp", 551, 625, "BVDK"), ("t1", 569, 659, "CBV"), ("rs", 589, 790, "C16"),
    ("dvnew", 598, 599, "CBV"),
    ("red1", 626, 629, "BV1"), ("rowb", 627, 628, "R1"), ("scal", 629, 808, "S8"),
    ("rowc", 802, 804, "R1"), ("dgam2", 705, 769, "CC"), ("da", 702, 735, "CC"),
    ("dx", 724, 753, "CC"), ("tmp2", 789, 831, "C1"), ("io_1d", 829, 831, "io_1d"),
]

# shape class -> size per (config, dtype).  config: "L0" (DK=DV=64), "TSC" (DK=DV=128)
def sz(cls, cfg, fp32):
    dk = 64 if cfg == "L0" else 128
    c, bv = 64, 64
    f, h = (4, 2) if fp32 else (4, 2)  # accum float=4B; dtype fp32=4B fp16=2B
    dt = 4 if fp32 else 2
    return {
        "CC": c * c * f, "CBV": c * bv * f, "CDK": c * dk * f,
        "BVDK": bv * dk * f, "C16": c * 16 * f, "C1": c * f, "S8": 8 * f,
        "BV1": bv * f, "R1": c * f,
        "io_cc": c * c * dt, "io_bq": c * bv * dt, "io_xk": c * dk * dt,
        "io_hv": bv * dk * dt, "io_1d": c * dt,
    }[cls]


# ---------------------------------------------------------------------------
# R3 candidate splits (planned edits to gdn_bwd_v2.py, all renames + 2 GM
# reloads; spans taken from the current file's line numbers).
# L0C: rs -> rs[587,591]+rs3[652,656]; rs2 -> rs4[687,688]+rs2[738,788];
#      dw -> dw_w[508,509]+dw[615,664].
# UB:  xk 5-way, io_xk 6-way, k2+k2b (GM reload at 679-680), t1+t1b,
#      io_hv 3-way, dh 2-way, wh 2-way, bq 2-way, io_bq 2-way,
#      io_cc 2-way, pk 2-way, msk 3-way, da 3-way, mul 3-way,
#      dead fills dropped (dq gen 820; dk gen 677 with += -> = at 678).
# ---------------------------------------------------------------------------

L0C_R3 = [
    ("dw1", 416, 417, "CDK"), ("u01", 433, 434, "CBV"), ("wh1", 443, 444, "CBV"),
    ("hout", 452, 455, "BVDK"),
    ("dw_w", 508, 509, "CDK"), ("kk", 517, 519, "CC"), ("dp1", 520, 524, "CC"),
    ("u0", 566, 568, "CBV"), ("wh", 570, 572, "CBV"), ("qh", 579, 581, "CBV"),
    ("rs", 587, 591, "C16"), ("dvnew", 596, 599, "CBV"), ("dp", 601, 698, "CC"),
    ("dq", 608, 819, "CDK"), ("dh", 609, 618, "BVDK"), ("dw", 615, 664, "CDK"),
    ("dkt", 622, 791, "CDK"), ("t1", 642, 643, "CBV"), ("rs3", 652, 656, "C16"),
    ("da", 661, 716, "CC"), ("t2", 672, 681, "CDK"),
    ("rs4", 687, 688, "C16"), ("rs2", 738, 788, "C16"),
    ("dk", 714, 823, "CDK"), ("dsum", 721, 722, "CC"), ("dx", 722, 724, "CC"),
]

UB_R3 = [
    ("hin", 366, 625, "BVDK"), ("io_hv1", 367, 460, "io_hv"),
    ("g", 373, 488, "C1"), ("be", 376, 745, "C1"), ("e", 378, 811, "C1"),
    ("glast", 381, 633, "S8"), ("th", 386, 799, "C1"), ("bee", 394, 676, "C1"),
    ("xk1", 398, 421, "CDK"), ("io_xk1", 399, 420, "io_xk"),
    ("bq1", 425, 447, "CBV"), ("io_bq1", 426, 450, "io_bq"),
    ("wh1", 445, 451, "CBV"), ("dh1", 455, 458, "BVDK"),
    ("gm", 487, 760, "CC"), ("k2", 493, 503, "CDK"), ("kk", 519, 747, "CC"),
    ("io_xk", 492, 515, "io_xk"), ("xk", 498, 515, "CDK"),
    ("pk", 522, 532, "CC"), ("mul", 524, 708, "CC"), ("msk", 525, 529, "CC"),
    ("io_cc1", 530, 531, "io_cc"),
    ("dE", 534, 811, "C1"), ("dgam", 535, 817, "C1"), ("tmp1", 536, 829, "C1"),
    ("dbe", 537, 696, "C1"),
    ("bq", 542, 606, "CBV"), ("io_bq", 543, 613, "io_bq"),
    ("io_hv", 546, 555, "io_hv"), ("dhp", 551, 625, "BVDK"),
    ("wh", 558, 576, "CBV"), ("wh2", 645, 649, "CBV"), ("t1", 569, 586, "CBV"),
    ("rs", 589, 790, "C16"), ("dvnew", 598, 612, "CBV"),
    ("red1", 626, 629, "BV1"), ("rowb", 627, 628, "R1"), ("scal", 629, 808, "S8"),
    ("dh", 618, 639, "BVDK"), ("io_hv2", 635, 639, "io_hv"),
    ("t1b", 644, 659, "CBV"), ("io_bq2", 645, 660, "io_bq"),
    ("xk2a", 665, 668, "CDK"), ("io_xk2a", 668, 668, "io_xk"),
    ("xk2b", 673, 678, "CDK"),
    ("dk", 677, 827, "CDK"), ("xk2c", 680, 684, "CDK"),
    ("io_xk2b", 684, 685, "io_xk"), ("k2b", 679, 683, "CDK"),
    ("pk2", 698, 706, "CC"), ("msk2", 700, 706, "CC"),
    ("da", 702, 712, "CC"), ("dgam2", 705, 769, "CC"),
    ("io_cc", 710, 771, "io_cc"), ("da_b", 716, 720, "CC"),
    ("msk3", 727, 729, "CC"), ("da2", 731, 737, "CC"), ("dx", 724, 754, "CC"),
    ("mul2", 744, 752, "CC"), ("mul3", 769, 770, "CC"),
    ("xk3a", 780, 783, "CDK"), ("io_xk3", 783, 784, "io_xk"),
    ("k2c", 779, 782, "CDK"),
    ("xk3b", 791, 796, "CDK"),
    ("tmp2", 789, 831, "C1"), ("rowc", 802, 804, "R1"),
    ("dq", 820, 821, "CDK"), ("io_xk4", 821, 822, "io_xk"),
    ("xk4", 824, 826, "CDK"), ("io_xk5", 827, 828, "io_xk"),
    ("io_1d", 829, 831, "io_1d"),
]


def run(l0c, ub, cfg, fp32, label):
    ivs_l0c = [(n, g, k, sz(s, cfg, fp32)) for n, g, k, s in l0c]
    ivs_ub = [(n, g, k, sz(s, cfg, fp32)) for n, g, k, s in ub]
    out = []
    for scope, ivs, lim in (("L0C", ivs_l0c, 131072), ("UB", ivs_ub, 196352)):
        ok, name, avail, _ = Sim(lim).allocate(ivs)
        if ok:
            out.append(f"{label} {cfg} fp{'32' if fp32 else '16'} {scope}: OK")
        else:
            out.append(
                f"{label} {cfg} fp{'32' if fp32 else '16'} {scope}: FAIL "
                f"at {name} required: {[i[3] for i in ivs if i[0] == name][0]}, "
                f"available: {avail}")
    return out


if __name__ == "__main__":
    print("=== round-2 state (calibration against observed errors) ===")
    for cfg in ("L0", "TSC"):
        for fp32 in (False, True):
            for line in run(L0C_ROUND2, UB_ROUND2, cfg, fp32, "R2"):
                print(line)
    print("=== round-3 candidate ===")
    for cfg in ("L0", "TSC"):
        for fp32 in (False, True):
            for line in run(L0C_R3, UB_R3, cfg, fp32, "R3"):
                print(line)

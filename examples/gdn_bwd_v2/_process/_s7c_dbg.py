# S7c process file: allocation-trace debugger for _s7c_mp_sim.py
# prints per-buffer offset/how (bump vs reuse) and the active/free state at failure
# Usage: python3 _s7c_dbg.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _s7c_mp_sim as m


class Dbg(m.Sim):
    def allocate(self, intervals):
        ivs = sorted(intervals, key=lambda t: t[1])
        active = []
        for name, gen, kill, size in ivs:
            while active:
                a = min(active, key=lambda t: t[0])
                if a[0] < gen:
                    active.remove(a)
                    self.free.append([a[2], a[3]])
                else:
                    break
            self.merge()
            new_off = m.align_up(self.next_new)
            if new_off + size <= self.limit:
                off = new_off
                self.next_new = new_off + size
                how = "bump"
            else:
                off = self.find_reusable(size)
                if off == -1:
                    print(f"FAIL {name} gen={gen} kill={kill} size={size}")
                    print("  active:", sorted([(a[1], a[0], a[2], a[3]) for a in active]))
                    print("  free:", sorted(self.free), "next_new:", self.next_new)
                    return False
                self.remove_from_free(off, size)
                how = "reuse"
            print(f"{gen:5d} {name:8s} kill={kill:5d} sz={size:6d} off={off:6d} {how}")
            active.append((kill, name, off, size))
        return True


ivs = [(n, g, k, m.sz(s, "TSC", True)) for n, g, k, s in m.UB_R3]
print("=== TSC fp32 UB trace ===")
Dbg(196352).allocate(ivs)

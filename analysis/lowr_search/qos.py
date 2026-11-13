"""Keep this harness's solver processes off Windows' background (efficiency-core only) scheduling.

On a hybrid CPU Windows may run processes it regards as background on the efficiency cores only; on
2026-11-13 the 20 evaluator worker processes shared the 8 E-cores while the 12 P-core threads sat idle
(2.5x slower). This opts the *descendants of a given Python process* out of power throttling
(SetProcessInformation / ProcessPowerThrottling, execution speed = unthrottled). It changes no system
setting and touches no process outside that tree.

    python qos.py <root_pid> [--watch]     # --watch: repeat every 5 s until the root process exits
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
k32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
k32.CloseHandle.argtypes = [wt.HANDLE]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wt.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


class POWER_THROTTLING(ctypes.Structure):
    _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]


k32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]


def processes():
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    e = PROCESSENTRY32W(); e.dwSize = ctypes.sizeof(e)
    out = []
    ok = k32.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        out.append((e.th32ProcessID, e.th32ParentProcessID, e.szExeFile))
        ok = k32.Process32NextW(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    return out


def descendants(root: int):
    procs = processes()
    kids, frontier = {}, {root}
    while frontier:
        nxt = {pid for pid, ppid, _ in procs if ppid in frontier and pid not in kids and pid != root}
        for pid, ppid, name in procs:
            if pid in nxt:
                kids[pid] = name
        frontier = nxt
    return kids, any(pid == root for pid, _, _ in procs)


def unthrottle(pid: int) -> bool:
    h = k32.OpenProcess(0x0200, False, pid)            # PROCESS_SET_INFORMATION
    if not h:
        return False
    s = POWER_THROTTLING(1, 0x1, 0)                    # control EXECUTION_SPEED, state 0 = never throttle
    ok = k32.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))   # 4 = ProcessPowerThrottling
    k32.CloseHandle(h)
    return bool(ok)


def sweep(root: int, done: set) -> tuple[int, bool]:
    kids, alive = descendants(root)
    n = 0
    for pid, name in kids.items():
        if pid not in done and (name.lower().startswith("lab6solver") or name.lower() in ("solver.exe", "python.exe")):
            if unthrottle(pid):
                done.add(pid); n += 1
    return n, alive


if __name__ == "__main__":
    root = int(sys.argv[1]); done: set = set()
    unthrottle(root)
    n, alive = sweep(root, done)
    print(f"unthrottled {n} descendants of {root}", flush=True)
    while "--watch" in sys.argv and alive:
        time.sleep(5)
        n, alive = sweep(root, done)

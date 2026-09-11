import gc
import datetime
import json
import math
import os
import pathlib
import sys


class GpuMemoryTracker:
    def __init__(self):
        self.reset(enabled=True)

    def reset(self, enabled=True):
        self.enabled = bool(enabled)
        self.samples = 0
        self.device_id = None
        self.device_total = 0
        self.current_device_used = 0
        self.current_device_free = 0
        self.current_pool_used = 0
        self.current_pool_total = 0
        self.peak_device_used = 0
        self.peak_device_label = None
        self.peak_pool_used = 0
        self.peak_pool_used_label = None
        self.peak_pool_total = 0
        self.peak_pool_total_label = None
        self.last_error = None

    def sample(self, label, include_device=False):
        if not self.enabled:
            return

        try:
            import cupy as cp

            self.samples += 1
            self.device_id = int(cp.cuda.runtime.getDevice())

            mempool = cp.get_default_memory_pool()
            pool_used = int(mempool.used_bytes())
            pool_total = int(mempool.total_bytes())
            self.current_pool_used = pool_used
            self.current_pool_total = pool_total

            if pool_used > self.peak_pool_used:
                self.peak_pool_used = pool_used
                self.peak_pool_used_label = label
            if pool_total > self.peak_pool_total:
                self.peak_pool_total = pool_total
                self.peak_pool_total_label = label

            if include_device:
                free, total = cp.cuda.runtime.memGetInfo()
                free = int(free)
                total = int(total)
                used = total - free
                self.device_total = total
                self.current_device_free = free
                self.current_device_used = used
                if used > self.peak_device_used:
                    self.peak_device_used = used
                    self.peak_device_label = label
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def print_summary(self):
        if not self.enabled:
            return

        self.sample("summary", include_device=True)
        print("\n[mem] GPU memory summary:")
        print(f"  samples: {self.samples}")
        if self.device_id is not None:
            print(f"  CUDA device: {self.device_id}")
        if self.device_total:
            print(f"  device total:       {_format_bytes(self.device_total)}")
            print(
                f"  device used now:    {_format_bytes(self.current_device_used)}  "
                f"free={_format_bytes(self.current_device_free)}"
            )
            print(
                f"  device used peak:   {_format_bytes(self.peak_device_used)}  "
                f"at {self.peak_device_label}"
            )
        print(
            f"  CuPy pool used peak:  {_format_bytes(self.peak_pool_used)}  "
            f"at {self.peak_pool_used_label}"
        )
        print(
            f"  CuPy pool total peak: {_format_bytes(self.peak_pool_total)}  "
            f"at {self.peak_pool_total_label}"
        )
        print(f"  CuPy pool used now:   {_format_bytes(self.current_pool_used)}")
        print(f"  CuPy pool total now:  {_format_bytes(self.current_pool_total)}")
        if self.last_error:
            print(f"  last sample error: {self.last_error}")
        print("  note: device usage is sampled; CuPy pool peaks catch cached CuPy allocations between samples.")


def _format_bytes(value):
    value = int(value or 0)
    if value <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = min(int(math.log(value, 1024)), len(units) - 1)
    scaled = value / (1024 ** idx)
    return f"{scaled:.2f} {units[idx]}"


gpu_memory_tracker = GpuMemoryTracker()


def _registry_dir():
    return pathlib.Path(__file__).resolve().parents[1] / "logs" / "gpu_processes"


def _current_cmdline():
    proc_cmdline = f"/proc/{os.getpid()}/cmdline"
    if os.path.exists(proc_cmdline):
        try:
            with open(proc_cmdline, "rb") as f:
                parts = [part.decode(errors="replace") for part in f.read().split(b"\0") if part]
            return " ".join(parts)
        except OSError:
            pass
    return " ".join(sys.argv)


def register_gpu_process(kind, metadata=None):
    out_dir = _registry_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pid_{os.getpid()}.json"
    payload = {
        "pid": os.getpid(),
        "kind": kind,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cmdline": _current_cmdline(),
        "cuda_device": os.environ.get("CUDA_DEVICE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "metadata": metadata or {},
    }
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    return str(path)


def unregister_gpu_process(path):
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[mem] GPU process registry cleanup warning: {type(exc).__name__}: {exc}")


def list_registered_gpu_processes():
    out_dir = _registry_dir()
    if not out_dir.exists():
        return []
    records = []
    for path in out_dir.glob("pid_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["_path"] = str(path)
            records.append(record)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def cleanup_cupy_memory(label="cleanup", include_device=False, synchronize=True, collect=True):
    errors = []

    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        return [f"cupy import failed: {type(exc).__name__}: {exc}"]

    if synchronize:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stream sync failed: {type(exc).__name__}: {exc}")
            try:
                cp.cuda.runtime.deviceSynchronize()
            except Exception as sync_exc:  # noqa: BLE001
                errors.append(f"device sync failed: {type(sync_exc).__name__}: {sync_exc}")

    if collect:
        gc.collect()

    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"default pool cleanup failed: {type(exc).__name__}: {exc}")

    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pinned pool cleanup failed: {type(exc).__name__}: {exc}")

    if collect:
        gc.collect()

    gpu_memory_tracker.sample(label, include_device=include_device)
    if errors:
        print("[mem] CuPy cleanup warning: " + "; ".join(errors))
    return errors


def manual_gpu_cleanup(label="manual cleanup"):
    return cleanup_cupy_memory(label=label, include_device=True, synchronize=True, collect=True)

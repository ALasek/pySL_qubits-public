"""
CLI README

Usage:
  python pySL_gpu_cleanup.py --list
  python pySL_gpu_cleanup.py --self-clean [--gpu-id GPU_ID]
  python pySL_gpu_cleanup.py --kill-owned-pysl [--force]

Purpose:
  Diagnose and clean up pySL GPU usage without restarting a shared machine.

Notes:
  --self-clean frees CuPy pools only in this Python process. It cannot free VRAM
  held by another live process.
  --kill-owned-pysl sends SIGTERM only to GPU processes owned by the current user
  whose command line looks like pySL.py or pySLbatch.py. Use --force to follow
  with SIGKILL if they do not exit.
"""

import argparse
import csv
import io
import os
import signal
import subprocess
import time
from dataclasses import dataclass


PYSL_PROCESS_MARKERS = ("pysl.py", "pyslbatch.py")


@dataclass
class GpuProcess:
    pid: int
    process_name: str
    used_memory: str
    cmdline: str
    owned_by_user: bool | None
    registered_pysl: bool = False


def _read_proc_cmdline(pid):
    proc_cmdline = f"/proc/{pid}/cmdline"
    if os.path.exists(proc_cmdline):
        try:
            with open(proc_cmdline, "rb") as f:
                parts = [part.decode(errors="replace") for part in f.read().split(b"\0") if part]
            return " ".join(parts)
        except OSError:
            return ""

    if os.name == "nt":
        command = (
            f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return completed.stdout.strip()

    return ""


def _owned_by_current_user(pid):
    if hasattr(os, "getuid"):
        try:
            return os.stat(f"/proc/{pid}").st_uid == os.getuid()
        except OSError:
            return None
    if os.name == "nt":
        return True
    return None


def query_gpu_compute_processes():
    from src.gpu_memory import list_registered_gpu_processes

    registry = {
        int(record["pid"]): record
        for record in list_registered_gpu_processes()
        if str(record.get("pid", "")).isdigit()
    }
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except FileNotFoundError:
        return [], "nvidia-smi was not found on PATH"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi query timed out"

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return [], f"nvidia-smi query failed: {message}"

    processes = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) < 3:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        cmdline = _read_proc_cmdline(pid)
        record = registry.get(pid)
        registered_pysl = False
        if record:
            record_cmdline = str(record.get("cmdline", ""))
            registered_pysl = not record_cmdline or not cmdline or record_cmdline == cmdline
        processes.append(
            GpuProcess(
                pid=pid,
                process_name=row[1].strip(),
                used_memory=row[2].strip(),
                cmdline=cmdline,
                owned_by_user=_owned_by_current_user(pid),
                registered_pysl=registered_pysl,
            )
        )
    return processes, None


def is_owned_pysl_process(process):
    if process.pid == os.getpid() or process.owned_by_user is not True:
        return False
    if process.registered_pysl:
        return True
    command_text = f"{process.process_name} {process.cmdline}".lower().replace("\\", "/")
    if "pysl_gpu_cleanup.py" in command_text:
        return False
    return any(marker in command_text for marker in PYSL_PROCESS_MARKERS)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _print_processes(processes):
    if not processes:
        print("No GPU compute processes reported by nvidia-smi.")
        return

    print("GPU compute processes:")
    print(f"{'pid':>8}  {'owned':>7}  {'pySL':>5}  {'mem MiB':>8}  process  command")
    for process in processes:
        owned = "yes" if process.owned_by_user else "no" if process.owned_by_user is False else "unknown"
        pysl = "yes" if is_owned_pysl_process(process) else "no"
        cmdline = process.cmdline or "(command line unavailable)"
        print(
            f"{process.pid:>8}  {owned:>7}  {pysl:>5}  {process.used_memory:>8}  "
            f"{process.process_name}  {cmdline}"
        )


def terminate_owned_pysl_processes(processes, wait_seconds=10.0, force=False):
    targets = [process for process in processes if is_owned_pysl_process(process)]
    if not targets:
        print("No owned pySL GPU processes found.")
        return []

    for process in targets:
        print(f"Sending SIGTERM to pySL GPU process pid={process.pid}")
        os.kill(process.pid, signal.SIGTERM)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        remaining = [process for process in targets if _pid_alive(process.pid)]
        if not remaining:
            return []
        time.sleep(0.25)

    remaining = [process for process in targets if _pid_alive(process.pid)]
    if remaining and force and hasattr(signal, "SIGKILL"):
        for process in remaining:
            print(f"Sending SIGKILL to stubborn pySL GPU process pid={process.pid}")
            os.kill(process.pid, signal.SIGKILL)
        time.sleep(0.25)
        remaining = [process for process in targets if _pid_alive(process.pid)]

    return remaining


def main():
    parser = argparse.ArgumentParser(description="Inspect and clean up pySL GPU process state.")
    parser.add_argument("--list", action="store_true", help="List GPU compute processes reported by nvidia-smi.")
    parser.add_argument("--self-clean", action="store_true", help="Free CuPy pools in this Python process.")
    parser.add_argument("--gpu-id", help="CUDA device id for --self-clean; sets CUDA_VISIBLE_DEVICES first.")
    parser.add_argument("--kill-owned-pysl", action="store_true", help="Terminate current user's pySL GPU processes.")
    parser.add_argument("--force", action="store_true", help="After SIGTERM, use SIGKILL for remaining pySL processes.")
    parser.add_argument("--wait-seconds", type=float, default=10.0, help="Seconds to wait after SIGTERM.")
    args = parser.parse_args()

    if not (args.list or args.self_clean or args.kill_owned_pysl):
        args.list = True

    if args.gpu_id is not None:
        os.environ["CUDA_DEVICE"] = str(args.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.self_clean:
        from src.gpu_memory import manual_gpu_cleanup

        errors = manual_gpu_cleanup()
        if errors:
            print("Self-clean completed with warnings.")
        else:
            print("Self-clean completed.")

    if args.list or args.kill_owned_pysl:
        processes, error = query_gpu_compute_processes()
        if error:
            print(error)
            return 1
        _print_processes(processes)

        if args.kill_owned_pysl:
            remaining = terminate_owned_pysl_processes(
                processes,
                wait_seconds=args.wait_seconds,
                force=args.force,
            )
            if remaining:
                print("Some pySL GPU processes are still alive:")
                _print_processes(remaining)
                return 2

            processes_after, error_after = query_gpu_compute_processes()
            if error_after:
                print(error_after)
                return 1
            print("\nAfter cleanup:")
            _print_processes(processes_after)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

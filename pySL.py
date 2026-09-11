"""
CLI README

Usage:
  python pySL.py [TARGET_GPU_ID] [--params PARAMS_FILE]

Arguments:
  TARGET_GPU_ID          Optional positional CUDA device id. If present, it must
                         come before flags; pySL stores it in CUDA_DEVICE and
                         CUDA_VISIBLE_DEVICES before importing CuPy.
  --params PARAMS_FILE   JSON file under run_params/ to load. Defaults to
                         params.json. If missing, built-in defaults are used.
"""

import datetime
import json
import os
import pathlib
import re
import shutil
import signal
import sys
import time
import tracemalloc


class _Tee:
    """Mirror writes to multiple streams (e.g. stdout + log file)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()


if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    print("\nTarget CUDA device: ", str(sys.argv[1]))
    os.environ["CUDA_DEVICE"] = str(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])


def _request_graceful_shutdown(signum, _frame):
    raise KeyboardInterrupt(f"received signal {signum}")


if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _request_graceful_shutdown)


import cupy as cp
import matplotlib.pyplot as plt

import src.get_gpu_info as gpuinfo
from src.export.path_utils import DEFAULT_RUN_SUBDIR, get_data_dir
from src.export.runDataSave import runDataSave
from src.export.run_store import compute_run_id
from src.gpu_memory import (
    cleanup_cupy_memory,
    gpu_memory_tracker,
    register_gpu_process,
    unregister_gpu_process,
)
from src.input.hamiltonian import hamiltonian
from src.input.default_params import build_default_params
from src.input.param_expressions import resolve_parameter_expressions
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import derive_time_grid, validate_params
from src.input.wavefunction import wavefunction
from src.plot.animate1D import animate1D
from src.plot.plot1Dtrace import plot1Dtrace


_RUN_PARAMS_DIR = pathlib.Path(__file__).parent / "run_params"


def _load_params_from_json(params_path):
    with open(params_path, "r") as f:
        return resolve_parameter_expressions(json.load(f))


def _safe_figure_name(text):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return cleaned[:80] or "figure"


class FigureSaver:
    def __init__(self, mode, subdir, params):
        self.mode = mode
        self.count = 0
        self.saved = set()
        self.dirs = []

        if mode in ("latest", "both"):
            latest_dir = os.path.join(get_data_dir(subdir), "latest_figs")
            if os.path.exists(latest_dir):
                shutil.rmtree(latest_dir)
            os.makedirs(latest_dir, exist_ok=True)
            self.dirs.append(latest_dir)

        if mode in ("run", "both"):
            run_dir = os.path.join(get_data_dir(subdir), "runs", compute_run_id(params), "figs")
            os.makedirs(run_dir, exist_ok=True)
            self.dirs.append(run_dir)

    def save_open_figures(self, force=False):
        if self.mode == "off":
            return

        for num in plt.get_fignums():
            if not force and num in self.saved:
                continue
            fig = plt.figure(num)
            title = fig._suptitle.get_text() if fig._suptitle is not None else ""
            if not title and fig.axes:
                title = fig.axes[0].get_title()
            self.count += 1
            filename = f"{self.count:03d}_{_safe_figure_name(title)}.png"
            for out_dir in self.dirs:
                fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=300)
            if not force:
                self.saved.add(num)


def run_pySL(readParams=1, subdir=DEFAULT_RUN_SUBDIR, params=None, params_path=None):
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.max_open_warning"] = 0
    original_show = plt.show

    log_dir = _RUN_PARAMS_DIR.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = str(log_dir / f"run_{log_stamp}_pid{os.getpid()}.log")
    _logfile = open(log_path, 'w', encoding='utf-8')
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(sys.__stdout__, _logfile)
    memory_summary_printed = False
    params_for_cleanup = None
    psi = None
    H = None
    _pane = None
    psi_cleared = False
    registry_path = None
    try:
        try:
            registry_path = register_gpu_process(
                "run_pySL",
                {
                    "subdir": subdir,
                    "params_path": params_path,
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[mem] GPU process registry warning: {type(exc).__name__}: {exc}")
        print("\n----------------------------------------------------------------")
        print("pySLQubits version 0.1.0")
        print("----------------------------------------------------------------")
        t0_full = time.time()

        tracemalloc.start()
        gpuinfo.print_gpu_info()

        redundancy_slope_mult = 0.5
        printMem = False
        evolver = "SL"

        print("Evolver = " + evolver)

        if params is None:
            resolved = params_path if params_path is not None else str(_RUN_PARAMS_DIR / "params.json")
            print(f"[PARAMS] Loading parameters from {resolved}")
            params = _load_params_from_json(resolved) if readParams else build_default_params()
        else:
            print("[PARAMS] Using provided in-memory parameters")

        assert params is not None
        validate_params(params)
        params = dict(params)
        halfN = params.get("fragment_half_only", False)
        params_for_cleanup = params
        gpu_memory_tracker.reset(enabled=params.get("log_gpu_memory", True))
        gpu_memory_tracker.sample("after params loaded", include_device=True)
        seed_plan = resolve_run_seed_plan_from_params(params)
        params.update(seed_plan)
        print(
            "[SEED] "
            f"input={seed_plan['seed_input']!r} base={seed_plan['seed_base']} "
            f"run_seeds={seed_plan['run_seeds']}"
        )
        time_grid = derive_time_grid(params)
        staged_protocol = params.get("evolution_protocol", "simultaneous") == "staged_write_store"
        write_step = (
            int(round(float(params.get("staged_write_time", 0)) / float(params["dT"])))
            if staged_protocol
            else None
        )
        correlator_reference_step = int(
            round(float(params.get("correlator_reference_time", 0)) / float(params["dT"]))
        )
        print(
            "[PARAMS] "
            f"T={params['T']} printT={params['printT']} dT={params['dT']} "
            f"samples={time_grid['print_sample_count']} "
            f"H_SE_Special={params['H_SE_Special']} "
            f"store_correlators={params.get('store_correlators', True)} "
            f"compute_discord={params.get('compute_discord', True)} "
            f"profile_evolution={params.get('profile_evolution', False)} "
            f"profile_evolution_max_steps={params.get('profile_evolution_max_steps', 200)} "
            f"log_gpu_memory={params.get('log_gpu_memory', True)}"
        )
        plot_mode = params.get("plot_mode", "auto")
        save_figures = params.get("save_figures")
        if save_figures is None:
            save_figures = "latest" if subdir == DEFAULT_RUN_SUBDIR else "off"
        figure_saver = FigureSaver(save_figures, subdir, params)

        if plot_mode == "external_nonblocking":
            def _nonblocking_show(*args, **kwargs):
                kwargs.pop("block", None)
                figure_saver.save_open_figures()
                original_show(*args, block=False, **kwargs)
                figure = plt.gcf()
                canvas = getattr(figure, "canvas", None)
                if canvas is not None:
                    try:
                        canvas.draw_idle()
                        canvas.flush_events()
                    except Exception:
                        pass

            plt.show = _nonblocking_show

        elif plot_mode == "plot_pane":
            plt.switch_backend("Agg")
            from src.plot.plot_pane import PlotPane
            _pane = PlotPane()

            def _pane_show(*__a, **__kw):  # noqa: ARG001
                figure_saver.save_open_figures(force=True)
                for num in plt.get_fignums():
                    _pane.push(plt.figure(num))
                plt.close("all")

            plt.show = _pane_show
        elif plot_mode == "none":
            plt.switch_backend("Agg")

            def _silent_show(*__a, **__kw):  # noqa: ARG001
                figure_saver.save_open_figures(force=True)
                plt.close("all")

            plt.show = _silent_show
        else:
            def _saving_show(*args, **kwargs):
                figure_saver.save_open_figures()
                original_show(*args, **kwargs)

            plt.show = _saving_show

        dT = params["dT"]
        printTi = time_grid["print_interval_steps"]
        noiseTi = time_grid["noise_interval_steps"]

        psi = wavefunction(params, 0)

        for rn, run_seed in enumerate(params["run_seeds"]):
            print("Doing run N=" + str(rn))

            t1 = time.time()
            psi.start_run(rn, run_seed)
            psi.buildPsi()
            if not psi.store_correlators or correlator_reference_step == 0:
                psi.buildPhi()
            t2 = time.time()
            print(f"Psi init: {t2 - t1:.4f} seconds")
            gpu_memory_tracker.sample(f"run {rn} after psi init", include_device=True)

            t1 = time.time()
            H = hamiltonian(params, seed=psi.seed)
            if params["p_noise_Gate"] > 0:
                H.run_noiseU_Gate()
            if params["noiseT"] > 0:
                H.run_noiseU_Rand()
            t2 = time.time()
            print(f"H init: {t2 - t1:.4f} seconds")
            gpu_memory_tracker.sample(f"run {rn} after H init", include_device=True)

            t1 = time.time()
            H_store_matrix = None
            if staged_protocol:
                psi.precomputeH_cpspCombine(H.component_view(include_SE=True, include_E=False, include_EE=False))
                H_write_matrix = psi.H_all
                psi.precomputeH_cpspCombine(H.component_view(include_SE=False, include_E=True, include_EE=True))
                H_store_matrix = psi.H_all
                psi.H_all = H_write_matrix
                print(
                    "[PROTOCOL] staged write/store: "
                    f"write until t={params['staged_write_time']}, then evolve with E fields + E-E only"
                )
            else:
                psi.precomputeH_cpspCombine(H)
            t2 = time.time()
            print(f"precomputeH_cpsp: {t2 - t1:.4f} seconds")
            gpu_memory_tracker.sample(f"run {rn} after H precompute", include_device=True)

            t1 = time.time()
            t2 = time.time()
            print(f"precomputeH_combine: {t2 - t1:.4f} seconds")

            printCount = 0
            t1 = time.time()

            for ti in range(time_grid["total_steps"] + 1):
                t = ti * dT

                if staged_protocol and ti == write_step:
                    psi.record_branch_energy_diagnostics(H_store_matrix)
                    psi.H_all = H_store_matrix
                    print(
                        "[PROTOCOL] switched to storage Hamiltonian: "
                        f"t={t:.6g} delta_epsilon={psi.branch_energy_density_gap[rn]:.6e}"
                    )

                if psi.store_correlators and ti == correlator_reference_step and not psi.correlators_active:
                    psi.buildPhi()
                    print(
                        "[CORRELATOR] initialized local perturbations: "
                        f"t={t:.6g} axis={psi.correlator_axis} sources={psi.correlator_sources}"
                    )

                if ti % printTi == 0:
                    t2 = time.time()
                    print(f"evolve segment: {t2 - t1:.4f} seconds")

                    t1 = time.time()
                    if evolver == "SL":
                        psi.print_norm(H_matrix=psi.H_all, dt=dT)
                        psi.psiparts()
                    else:
                        psi.print_norm()
                    if psi.store_psi:
                        psi.store_psi_snapshot(printCount)
                    if psi.store_correlators:
                        psi.print_correlators(printCount)
                    fragment_samples = params.get("fragment_sample_count") or params["nqubits_E"]
                    psi.print_vn(printCount, fragment_samples, halfN=halfN)
                    psi.printTnow += 1
                    t2 = time.time()
                    print(f"print : {t2 - t1:.4f} seconds")
                    t1 = time.time()

                    print(f"print : time= {t:.4f} seconds")
                    printCount += 1

                    if printMem:
                        free, total = cp.cuda.runtime.memGetInfo()
                        used = total - free
                        print("GPU Memory Usage (CUDA runtime):")
                        print(f"  Used memory: {used / (1024 ** 2):.2f} MB")
                        print(f"  Free memory: {free / (1024 ** 2):.2f} MB")
                        print(f"  Total memory: {total / (1024 ** 2):.2f} MB")

                if ti == time_grid["total_steps"]:
                    continue

                if evolver == "Euler":
                    psi.evolve_euler(H, dT)
                elif evolver == "SL":
                    if params["noiseT"] > 0:
                        if ti % noiseTi == 0:
                            print("Start noise Rand")
                            H.run_noise_times()
                            H.start_noise_H()
                        else:
                            H.update_noise_H(t)

                    if params["p_noise_Gate"] > 0 and (ti % params["noiseT_Gate"]) == 0:
                        print("Apply noise Gate")
                        psi.apply_noise_Gate(H, params)

                    psi.evolve_step(H, dT)
                    psi.Tnow = (ti + 1) * dT

            psi.plot_norm()
            psi.finish_run_metrics()
            psi.persist_psi_snapshots_current_run(run_seed)
            psi.incRunN()
            gpu_memory_tracker.sample(f"run {rn} complete", include_device=True)

        psi.doRunAvs()
        psi.max_redundancy_slope(redundancy_slope_mult)
        psi.print_ISEerrorbarQDslope()
        psi.plot_I_S()

        for frag_size in range(1, params["QREmaxFragSize"] + 1):
            psi.plot_SBSq(frag_size)

        psi.plot_RhoSoffdiagDecay()
        if psi.store_correlators:
            psi.plot_correlators()
        psi.plot_rhoS_Bloch(True)
        psi.plot_rhoS_Bloch(False)
        psi.plot_rhoS_Evals()

        if int(round(float(params["AverageOverRunsN"]))) == 1:
            overlapUD = psi.plot_Psi_E_init()
            print(overlapUD)
        else:
            print("[AVG] Skipping initial environment-state plot; realizations use different seeds.")
        H.decompose_H_SE()

        t_print = time.time() - t0_full
        print("\nTotal time elapsed = ", t_print)
        psi.print_perf_summary()
        gpu_memory_tracker.print_summary()
        memory_summary_printed = True

        runDataSave(psi, params, subdir)
        psi.clearMem()
        psi_cleared = True

        if plot_mode == "external_nonblocking":
            original_show()
        elif _pane is not None:
            _pane.wait()

        return psi, H, params
    except BaseException:
        if not memory_summary_printed:
            gpu_memory_tracker.sample("exception", include_device=True)
            gpu_memory_tracker.print_summary()
        raise
    finally:
        try:
            if psi is not None and not psi_cleared:
                psi.clearMem()
                psi_cleared = True
        except Exception as exc:  # noqa: BLE001
            print(f"[mem] wavefunction cleanup warning: {type(exc).__name__}: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass
        include_device = True
        if params_for_cleanup is not None:
            include_device = params_for_cleanup.get("log_gpu_memory", True)
        cleanup_cupy_memory("run_pySL finally", include_device=include_device)
        unregister_gpu_process(registry_path)
        plt.show = original_show
        sys.stdout = _orig_stdout
        _logfile.close()
        print(f"[LOG] saved to {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json", help="Filename in run_params/ to load (default: params.json)")
    args, _ = parser.parse_known_args()
    _params_path = _RUN_PARAMS_DIR / args.params
    if _params_path.exists():
        [psi, H, params] = run_pySL(readParams=1, params_path=str(_params_path))
    else:
        print(f"[INFO] {_params_path} not found, using build_default_params()")
        [psi, H, params] = run_pySL(0)

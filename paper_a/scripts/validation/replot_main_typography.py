"""Regenerate the five approved main figures with the shared LaTeX typography."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import runpy
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from matplotlib.figure import Figure

from paper_plot_style import apply_style, save_figure


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "analysis" / "figure_typography_2026_09_09" / "main_sources.json"
DEFAULT_DATA_ROOT = ROOT.parent / "Papers" / "output" / "recalculations" / "paper-a-2026-09-05" / "data"
DEFAULT_SUMMARY_ROOT = ROOT.parent / "QD-summary-Imperfect-CNOT" / "scripts"
DEFAULT_SIMULATION_ROOT = ROOT.parent / "pySL_qubits" / "analysis_scripts"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "paper_a"
PNG_PLACEHOLDER = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0dIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def styled_saves():
    original = Figure.savefig

    def savefig(fig: Figure, path, *args, **kwargs):
        if Path(path).suffix.lower() == ".png":
            # Upstream manifests hash preview PNGs; this typography-only run installs PDFs only.
            Path(path).write_bytes(PNG_PLACEHOLDER)
            return
        Figure.savefig = original
        try:
            save_figure(fig, path, *args, **kwargs)
        finally:
            Figure.savefig = savefig

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = original


def run_plotter(script: Path, argv: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(script)
    previous_argv = sys.argv
    previous_path = sys.path[:]
    sys.argv = [str(script), *argv]
    sys.path[:0] = [str(script.parent), str(script.parent.parent)]
    apply_style()
    try:
        try:
            with styled_saves():
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def run_exact_fragment_plotter(script: Path, argv: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(script)
    module_name = "_main_figure_typography_matched_sweep"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    original_quantile_array = module.quantile_array

    def quantiles_without_empty_fragment(*args, **kwargs):
        values = original_quantile_array(*args, **kwargs).copy()
        values[0] = float("nan")
        return values

    module.quantile_array = quantiles_without_empty_fragment
    previous_argv = sys.argv
    sys.argv = [str(script), *argv]
    apply_style()
    try:
        with styled_saves():
            result = module.main()
        if result not in (None, 0):
            raise RuntimeError(f"{script} returned {result}")
    finally:
        sys.argv = previous_argv
        del sys.modules[module_name]


def verify_tables(work_dir: Path, expected: dict[str, str]) -> None:
    for relative, expected_hash in expected.items():
        actual_path = work_dir / relative
        if not actual_path.is_file():
            raise FileNotFoundError(f"plotter did not write {actual_path}")
        actual_hash = sha256(actual_path)
        if actual_hash != expected_hash:
            raise ValueError(f"derived table changed: {relative}: {actual_hash} != {expected_hash}")


def validate_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty PDF: {path}")
    content = path.read_bytes()
    if b"/Type /Page" not in content:
        raise ValueError(f"invalid PDF page structure: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--summary-script-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--simulation-script-root", type=Path, default=DEFAULT_SIMULATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    data_root = args.data_root.resolve()
    summary_root = args.summary_script_root.resolve()
    simulation_root = args.simulation_script_root.resolve()
    output_dir = args.output_dir.resolve()
    for directory in (data_root, summary_root, simulation_root):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    with tempfile.TemporaryDirectory(prefix="paper-a-main-typography-") as temporary:
        work_dir = Path(temporary)
        run_plotter(
            summary_root / "plot_holevo_plateau_overview.py",
            ["--batch-dir", str(data_root / "Paper_A_final_HolevoPlateauFullFragments"), "--output-dir", str(work_dir / "plateau")],
        )
        run_exact_fragment_plotter(
            summary_root / "plot_matched_lambda_fragment_sweep.py",
            [str(data_root / "Paper_A_final_SubmissionMatchedLambda"), str(work_dir / "matched_sweep")],
        )
        run_plotter(
            simulation_root / "paper_a_field_profile_control.py",
            ["--batch-dir", str(data_root / "Paper_A_final_FieldProfileControl"), "--output-dir", str(work_dir / "field"), "--bootstrap", "2000"],
        )
        run_plotter(
            simulation_root / "paper_a_submission_panels.py",
            ["--data-root", str(data_root), "--output-dir", str(work_dir / "submission"), "--families", "stability", "--strict-missing"],
        )
        verify_tables(work_dir, sources["table_hashes_sha256"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for generated, target in sources["targets"].items():
            source = work_dir / generated
            validate_pdf(source)
            destination = output_dir / target
            shutil.copy2(source, destination)
            validate_pdf(destination)
            print(f"wrote {destination} {sha256(destination)}")


if __name__ == "__main__":
    main()

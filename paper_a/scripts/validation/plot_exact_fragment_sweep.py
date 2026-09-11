from pathlib import Path
import importlib.util
import sys

source = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location('exact_fragment_plotter', source)
plotter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plotter
spec.loader.exec_module(plotter)
original = plotter.quantile_array


def quantiles_without_empty_fragment(*args, **kwargs):
    values = original(*args, **kwargs).copy()
    # The source plotter identifies sampled sizes by finite quantiles; m=0 is analytic.
    values[0] = float('nan')
    return values


plotter.quantile_array = quantiles_without_empty_fragment
sys.argv[0] = str(source)
raise SystemExit(plotter.main())

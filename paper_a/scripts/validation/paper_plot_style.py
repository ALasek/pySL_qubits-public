from pathlib import Path

import matplotlib as mpl
from matplotlib.text import Text


def apply_style():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}",
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, path, **kwargs):
    apply_style()
    printed_width = {"holevo_plateau_overview.pdf": 3.4,
                     "pointer_basis_expanded.pdf": 4.97,
                     "higher_n_validation.pdf": 6.39}.get(Path(path).name, 7.1)
    minimum_size = 6.5 * fig.get_figwidth() / printed_width
    for label in fig.findobj(Text):
        label.set_fontfamily("serif")
        label.set_usetex(True)
        label.set_fontsize(max(label.get_fontsize(), minimum_size))
        label.set_text(label.get_text().replace("H_Z(S)", r"H_Z(\mathcal{S})")
                       .replace("N_E", r"N_{\mathcal{E}}"))
    fig.savefig(path, **kwargs)

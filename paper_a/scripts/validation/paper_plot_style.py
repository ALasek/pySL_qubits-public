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


# Figure-internal titles must use the main-text terminology.
TITLE_REPLACEMENTS = {
    r"Mean pointer information, $\Lambda=3/4$": r"Mean Holevo information, $\Lambda=3/4$",
    "Conservative pointer information": "10th-percentile Holevo information",
    "Conservative discord-like remainder": "90th-percentile pointer-basis remainder",
    r"Aligned recorder ($p=1/2$, $\theta=0$), $N_E=16$": r"Blind preparation ($p=1/2$, $\theta=0$), $N_E=16$",
    r"Aligned-recorder fragment scaling ($p=1/2$, $\theta=0$)": r"Fragment-size dependence at the blind preparation ($p=1/2$, $\theta=0$)",
}


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
        text = TITLE_REPLACEMENTS.get(label.get_text(), label.get_text())
        label.set_text(text.replace("H_Z(S)", r"H_Z(\mathcal{S})")
                       .replace("N_E", r"N_{\mathcal{E}}"))
    fig.savefig(path, **kwargs)

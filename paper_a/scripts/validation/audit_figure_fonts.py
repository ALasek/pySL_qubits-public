import hashlib
import json
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[2]


def main():
    records = []
    for section in sorted((ROOT / "sections").glob("*.tex")):
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", section.read_text()):
            path = ROOT / "figures" / name
            with fitz.open(path) as document:
                fonts = sorted({font[3] for page in document for font in page.get_fonts()})
                assert fonts and all(font.split("+")[-1].startswith("LM") for font in fonts), (name, fonts)
                assert all(font[1] != "n/a" for page in document for font in page.get_fonts()), name
                assert all("\ufffd" not in page.get_text() for page in document), name
            records.append({"file": str(path.relative_to(ROOT)), "fonts": fonts,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    style = ROOT / "scripts/validation/paper_plot_style.py"
    report = {"style_sha256": hashlib.sha256(style.read_bytes()).hexdigest(),
              "figures": records, "schematic": "Figure 1 is TikZ compiled with the manuscript fonts."}
    target = ROOT / "analysis/figure_typography_2026_09_09/font_audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(records)} included plot PDFs use embedded Latin Modern fonts.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper for loading pySL run data.

The loader prefers v2 JSON/NPZ runs when an index is available and falls back to
legacy `.rundata` files. The public `matchdata` shape remains compatible with
existing callers.
"""

import os

from src.export.path_utils import DEFAULT_RUN_SUBDIR, get_data_dir
from src.export.run_store import load_runs_matching


class runDataLoad:
    def __init__(self, reqParams, subdir=DEFAULT_RUN_SUBDIR, dataOld=False):
        self.matchdata = []
        self.outDir = get_data_dir(subdir)

        if not os.path.exists(self.outDir):
            print("OUTDIR doesn't exist")
            return

        self.matchdata = load_runs_matching(
            reqParams,
            subdir=subdir,
            allow_legacy=True,
            old_data=dataOld,
        )

    def plotData(self):
        for data in self.matchdata:
            psidummy = data[1]
            if hasattr(psidummy, "max_redundancy_slope"):
                psidummy.max_redundancy_slope(0.5)
            for method_name, args in [
                ("print_ISEerrorbarQDslope", ()),
                ("plot_I_S", ()),
                ("plot_rhoS_Bloch", (True,)),
                ("plot_rhoS_Bloch", (False,)),
                ("plot_rhoS_Evals", ()),
            ]:
                if hasattr(psidummy, method_name):
                    getattr(psidummy, method_name)(*args)

    def plotDataOld(self):
        from src.input.wavefunction import wavefunction

        for data in self.matchdata:
            psidummy = wavefunction(data[0], 0)
            psidummy.I_S_Ef_fractionsT_runAv = data[1]
            psidummy.I_S_Ef_fractionsT_STD_runAv = data[2]
            psidummy.S_vn_runAv = data[3]
            psidummy.max_redundancy_slope(0.5)
            psidummy.print_ISEerrorbarQDslope()
            psidummy.plot_I_S()

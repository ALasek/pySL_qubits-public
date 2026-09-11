#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper for saving pySL run data.

New runs are saved in the v2 JSON/NPZ store. Legacy `.rundata` pickles are
opt-in because they pickle the full live simulation object and can exceed RAM
for large runs.
"""

from src.export.path_utils import DEFAULT_RUN_SUBDIR, get_data_dir
from src.export.run_store import save_run


class runDataSave:
    def __init__(self, psi, params, subdir=DEFAULT_RUN_SUBDIR):
        self.outDir = get_data_dir(subdir)
        write_legacy = bool(params.get("save_legacy_rundata", False))
        self.save_result = save_run(psi, params, subdir=subdir, write_legacy=write_legacy)


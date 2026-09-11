#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI README

Usage:
  python pySL_batchAnalyzeTheta.py [--config CONFIG] [--subdir SUBDIR] [options]

Configuration:
  CLI flags override optional local defaults from
  analysis_configs/batch_analyze_theta.local.json. Copy
  analysis_configs/batch_analyze_theta.example.json as a starting point.

Common flags:
  --local-config FILE, --config CONFIG, --subdir SUBDIR, --x PARAM, --y PARAM,
  --Tsample TIME, --setMinTime TIME, --redundancyminfraction X, --old-data,
  --plot-data BOOL

Output plots are saved under data/figs/analysis1 and shown in the plot pane.

Created on Tue Nov  4 16:56:26 2025

@author: ALasek
"""

import argparse
import sys
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt

# Import project modules
import src.get_gpu_info as gpuinfo
from src.get_ccbin import get_ccbin
from src.input.parameters import getparams
from src.input.wavefunction import wavefunction
from src.input.hamiltonian import hamiltonian
from src.input.operator import operator

from src.export.wavefunctionOut import wavefunctionOut
from src.export.exportTrace import exportTrace

from src.export.runDataSave import runDataSave

from src.export.makepath import makepath
from src.export.exportParameters import exportParameters
from src.export.runDataLoad import runDataLoad
# Built-in plotting
from src.plot.animate1D import animate1D
from src.plot.plot1Dtrace import plot1Dtrace

import itertools

from pySL_print import pySL_print
from src.input.wavefunction import wavefunction
from src.analysis.local_config import apply_config_defaults, load_local_config_with_keys
from src.batch_utils import load_batch_config

from scipy.signal import savgol_filter
import scipy.ndimage as ndi

import plotly.graph_objects as go


ANALYSIS_FIG_DIR = os.path.join("data", "figs", "analysis1")
_original_show = plt.show
_plot_pane = None


def _save_analysis_fig(title):
    os.makedirs(ANALYSIS_FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(ANALYSIS_FIG_DIR, title + ".jpg"), dpi=300)


def _save_plotly_fig(fig, title):
    os.makedirs(ANALYSIS_FIG_DIR, exist_ok=True)
    fig.write_html(os.path.join(ANALYSIS_FIG_DIR, title + ".html"))


def _push_open_figures_to_pane(*__a, **__kw):  # noqa: ARG001
    if _plot_pane is None:
        _original_show()
        return
    for num in plt.get_fignums():
        _plot_pane.push(plt.figure(num))
    plt.close("all")


def _setup_plot_pane():
    global _plot_pane
    plt.switch_backend("Agg")
    from src.plot.plot_pane import PlotPane

    _plot_pane = PlotPane(title="Theta Batch Analysis Plots")
    plt.show = _push_open_figures_to_pane


def _wait_for_plot_pane():
    if _plot_pane is not None:
        _plot_pane.wait()


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Legacy theta batch analyzer.")
    parser.add_argument(
        "--local-config",
        default="analysis_configs/batch_analyze_theta.local.json",
        help="Ignored JSON config file for local analysis defaults.",
    )
    parser.add_argument("--subdir", help="Data subdirectory under data/.")
    parser.add_argument("--config", help="Python batch config module or .py path.")
    parser.add_argument("--x", help="Sweep parameter to use as the x axis.")
    parser.add_argument("--y", help="Sweep parameter to use as the y axis.")
    parser.add_argument("--Tsample", type=float, help="Sample time; use -1 for best-time scan.")
    parser.add_argument("--setMinTime", type=float, help="Minimum physical time for redundancy metrics.")
    parser.add_argument("--redundancyminfraction", type=float, help="Minimum redundancy fraction threshold.")
    parser.add_argument("--old-data", action="store_true", default=None, help="Load legacy old-data pickle layout.")
    parser.add_argument("--plot-data", type=_parse_bool, default=None, help="Plot each matched run before analysis.")
    parser.add_argument("--smooth", type=_parse_bool, default=None, help="Smooth loaded I(S:E_f) data before metrics.")
    parser.add_argument("--nqubits", type=int, help="Total qubit count used for filtering and plot titles.")
    return parser.parse_args()


def _apply_cli_overrides(config, args):
    if args.config is not None:
        config["config"] = args.config
    if args.subdir is not None:
        config["subdir"] = args.subdir
    if args.x is not None:
        config["x"] = args.x
    if args.y is not None:
        config["y"] = args.y
    if args.Tsample is not None:
        config["Tsample"] = args.Tsample
    if args.setMinTime is not None:
        config["setMinTime"] = args.setMinTime
    if args.redundancyminfraction is not None:
        config["redundancyminfraction"] = args.redundancyminfraction
    if args.old_data is not None:
        config["dataOld"] = args.old_data
    if args.plot_data is not None:
        config["plotData"] = args.plot_data
    if args.smooth is not None:
        config["smooth"] = args.smooth
    if args.nqubits is not None:
        config["nqubits"] = args.nqubits
    return config


def _cli_override_keys(args):
    keys = set()
    if args.config is not None:
        keys.add("config")
    if args.subdir is not None:
        keys.add("subdir")
    if args.x is not None:
        keys.add("x")
    if args.y is not None:
        keys.add("y")
    if args.Tsample is not None:
        keys.add("Tsample")
    if args.setMinTime is not None:
        keys.add("setMinTime")
    if args.redundancyminfraction is not None:
        keys.add("redundancyminfraction")
    if args.old_data is not None:
        keys.add("dataOld")
    if args.plot_data is not None:
        keys.add("plotData")
    if args.smooth is not None:
        keys.add("smooth")
    if args.nqubits is not None:
        keys.add("nqubits")
    return keys


def _axis_values(config, batch_params, axis_name, override_name):
    if config.get(override_name) is not None:
        return np.asarray(config[override_name])
    return np.asarray(batch_params[axis_name])


def _axis_index(values, value):
    matches = np.where(np.isclose(values.astype(float), float(value)))[0]
    if len(matches) == 0:
        matches = np.where(values == value)[0]
    return matches


def _add_fixed_filter(batch_params, params_default, names):
    for name in names:
        if name in params_default and name not in batch_params:
            batch_params[name] = [params_default[name]]
    return batch_params


def _resolve_qubit_filter(analysis_config, params_default):
    nqubits_s = int(params_default.get("nqubits_S", 1))
    if analysis_config.get("nqubits") is not None:
        nqubits = int(analysis_config["nqubits"])
        return nqubits, nqubits - nqubits_s
    if "nqubits_E" in params_default:
        nqubits_e = int(params_default["nqubits_E"])
        return nqubits_s + nqubits_e, nqubits_e
    return 16, 15


def _first_match_or_raise(filter_dict, subdir, plot_data, data_old):
    matches = pySL_print(filter_dict, subdir, plot_data, data_old)
    if matches:
        return matches[0]
    filter_text = ", ".join(f"{key}={value!r}" for key, value in sorted(filter_dict.items()))
    raise RuntimeError(
        f"No runs matched in data/{subdir}. Check the selected batch config, "
        f"--nqubits, and completed run data. Filter: {filter_text}"
    )



def smooth2D_sagov(data2D,window_length=7, polyorder=3, axis=0):
    smoothed = savgol_filter(data2D, window_length, polyorder, axis)
    # smoothed = savgol_filter(smoothed, window_length=7, polyorder=3, axis=1)

    return smoothed

def smooth2D_ndi(data2D,sigma=2):

    smoothed = ndi.gaussian_filter(data2D, sigma)      

    return smoothed

BATCH_ANALYSIS_CONFIG_KEYS = (
    "x",
    "y",
    "Tsample",
    "nqubits",
    "setMinTime",
    "redundancyminfraction",
    "plotData",
)

DEFAULT_ANALYSIS_CONFIG = {
    "config": "batch_configs.mironowicz_theta_sweep",
    "x": "Mironowicz_theta",
    "y": "Mironowicz_alpha2",
    "x_arr": None,
    "y_arr": None,
    "Tsample": -1,
    "nqubits": None,
    "redundancyminfraction": 0.4,
    "setMinTime": 30,
    "plotData": True,
    "smooth": False,
    "subdir": None,
    "dataOld": False,
}


args = _parse_cli_args()
analysis_config, local_config_keys = load_local_config_with_keys(
    DEFAULT_ANALYSIS_CONFIG,
    args.local_config,
    "theta analysis",
)
analysis_config = _apply_cli_overrides(analysis_config, args)
cli_config_keys = _cli_override_keys(args)

batch_config = load_batch_config(analysis_config["config"])
analysis_config = apply_config_defaults(
    analysis_config,
    batch_config.get("analysis", {}),
    protected_keys=local_config_keys | cli_config_keys,
    keys=BATCH_ANALYSIS_CONFIG_KEYS,
)

Tsample = analysis_config["Tsample"]
params_default = batch_config.get("params_default", {})
nqubits, nqubits_e = _resolve_qubit_filter(analysis_config, params_default)
redundancyminfraction = analysis_config["redundancyminfraction"]
setMinTime = analysis_config["setMinTime"]
plotData = analysis_config["plotData"]
smooth = analysis_config["smooth"]
dataOld = analysis_config["dataOld"]

_setup_plot_pane()


batchParams = batch_config["sweep"].copy()
subdir = analysis_config["subdir"] or batch_config["subdir"]
x_axis = analysis_config["x"]
y_axis = analysis_config["y"]

x_arr = _axis_values(analysis_config, batchParams, x_axis, "x_arr")
y_arr = _axis_values(analysis_config, batchParams, y_axis, "y_arr")

Plotdata2D=np.zeros([len(x_arr), len(y_arr)])

BestTimes2D=np.zeros([len(x_arr), len(y_arr)])

Plotdata2D_fracScore=np.zeros([len(x_arr), len(y_arr)])

Plotdata2D_badval=np.zeros([len(x_arr), len(y_arr)])

if "nqubits_E" not in batchParams:
    batchParams["nqubits_E"] = [nqubits_e]
batchParams[x_axis] = x_arr
batchParams[y_axis] = y_arr
batchParams = _add_fixed_filter(
    batchParams,
    params_default,
    [
        "nqubits_S",
        "psi_bias",
        "Mironowicz_H_E",
        "H_SE_Special",
        "Mironowicz_epsilon",
        "Mironowicz_epsilon2",
        "T",
        "printT",
    ],
)


# Get keys and value lists
keys = list(batchParams.keys())
value_lists = [batchParams[key] for key in keys]

# Iterate over all combinations
for combo in itertools.product(*value_lists):
    
    filterDict={}
    x_val = None
    y_val = None
    
    for key, value in zip(keys, combo):
        # print(key)
        # print(value)
        
        filterDict[key]=value
        if key == x_axis:
            x_val=(value)
        elif key == y_axis:
            y_val=value
    
    x_ind = _axis_index(x_arr, x_val)
    y_ind = _axis_index(y_arr, y_val)
    
    if dataOld:
        matchdata=_first_match_or_raise(filterDict,subdir,plotData,dataOld)
        psidummy=wavefunction(matchdata[0],0)
        psidummy.I_S_Ef_fractionsT_runAv=matchdata[1]
        psidummy.I_S_Ef_fractionsT_STD_runAv=matchdata[2]
        
        psiT=matchdata[4]
        psidT=matchdata[5]
        psiprintT=matchdata[6]
    if not dataOld:
        matchdata=_first_match_or_raise(filterDict,subdir,plotData,dataOld)
        psidummy=matchdata[1]
        
        psiT=psidummy.T
        psidT=psidummy.dT
        psiprintT=psidummy.printT
        
    #smoothing
    if smooth:
        psidummy.I_S_Ef_fractionsT_runAv=smooth2D_sagov(psidummy.I_S_Ef_fractionsT_runAv,window_length=20, polyorder=4, axis=0)
        
        # matchdata[1]=smooth2D_ndi(matchdata[1],sigma=0.01)
    
    
    if plotData:
        psidummy.plot_I_S()
        
       


    setMinTime_proper=setMinTime/psiprintT #check if float or int is required in target funct

    if Tsample>0:
        indexofT=int(Tsample/psiprintT)
    else:
        indexofT=-1

    slopeData= psidummy.redundancy_slope_at_T(indexofT,0.5,minTime=setMinTime_proper)
    
    fracScoreData=psidummy.redundancy_fractionScore_at_T(indexofT,cutoff=0.5,minfraction=redundancyminfraction,minTime=setMinTime_proper)
    Plotdata2D_fracScore[x_ind,y_ind]   = 1/fracScoreData[0]
    
    if not (slopeData is None):
        
        Plotdata2D[x_ind,y_ind]=slopeData[0]
        BestTimes2D[x_ind,y_ind]=slopeData[1]
        
    else:
        print(f"bad data at [{x_axis}, {y_axis}] = " + str([x_val,y_val]))
        Plotdata2D_badval[x_ind,y_ind]=1

    if not (slopeData is None):
        if slopeData[0]==-1:
            print("-1 found")
            slopeData[0]=0
            Plotdata2D[x_ind,y_ind]=slopeData[0]





X, Y = np.meshgrid(x_arr, y_arr)

# Create 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Plotdata2D.T, cmap='viridis', edgecolor='none')
title='ISE_3Dplot_N='+str(nqubits)+" T="+str(Tsample)
# Add color bar and labels
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(title)
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_zlabel("Normalized slope at T= "+ str(Tsample) +" (0=QD)")
_save_analysis_fig(title)
plt.show()




# Create 2D heatmap
fig, ax = plt.subplots()
c = ax.pcolormesh(x_arr, y_arr, Plotdata2D.T, cmap='viridis', shading='auto')

# Add colorbar and labels
fig.colorbar(c, ax=ax, label='Z Value')


# Fill selected cell with solid red

badcells=np.where(Plotdata2D_badval==1)

for i in range(len(badcells[0])):
    x0 = x_arr[badcells[0][i]]
    x1 = x_arr[badcells[0][i] + 1] if badcells[0][i] + 1 < len(x_arr) else x_arr[badcells[0][i]]
    y0 = y_arr[badcells[1][i]]
    y1 = y_arr[badcells[1][i] + 1] if badcells[1][i] + 1 < len(y_arr) else y_arr[badcells[1][i]]
    
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor='red', facecolor='red', alpha=0.6)
    ax.add_patch(rect)

title='ISE_2Dplot_N='+str(nqubits) +" T="+str(Tsample)
plt.title(title)

ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_label("Normalized slope at T= "+ str(Tsample) +" (0=QD)")
_save_analysis_fig(title)
plt.show()



# Create surface plot
plotly_title='ISE_surface_N='+str(nqubits)+" T="+str(Tsample)
fig = go.Figure(data=[go.Surface(z=Plotdata2D.T, x=x_arr, y=y_arr)])
fig.update_layout(title=plotly_title, scene=dict(
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    zaxis_title='Normalized slope'
))
_save_plotly_fig(fig, plotly_title)

#Plot fracScore


# Create 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Plotdata2D_fracScore.T, cmap='viridis', edgecolor='none')
title='fracScore_3Dplot_N='+str(nqubits)+" T="+str(Tsample)
# Add color bar and labels
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(title)
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_zlabel("fracScore at T= "+ str(Tsample) +" (0=QD)")
_save_analysis_fig(title)
plt.show()



# Create 2D heatmap
fig, ax = plt.subplots()
c = ax.pcolormesh(x_arr, y_arr, Plotdata2D_fracScore.T, cmap='viridis', shading='auto')

# Add colorbar and labels
fig.colorbar(c, ax=ax, label='Redundancy')


title='fracScore_2Dplot_N='+str(nqubits) +" T="+str(Tsample)
plt.title(title)

ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_label("fracScore at T= "+ str(Tsample) +" (0=QD)")
_save_analysis_fig(title)
plt.show()



# Create surface plot
plotly_title='fracScore_surface_N='+str(nqubits)+" T="+str(Tsample)
fig = go.Figure(data=[go.Surface(z=Plotdata2D_fracScore.T, x=x_arr, y=y_arr)])
fig.update_layout(title=plotly_title, scene=dict(
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    zaxis_title='fracScore'
))
_save_plotly_fig(fig, plotly_title)

# Create 2D heatmap for BestTImes
BestTimes2D=BestTimes2D*psiprintT
fig, ax = plt.subplots()
c = ax.pcolormesh(x_arr, y_arr, BestTimes2D.T, cmap='viridis', shading='auto')

# Add colorbar and labels
fig.colorbar(c, ax=ax, label='Time')


title='bestTimes_2Dplot_N='+str(nqubits) +" T="+str(Tsample)
plt.title(title)

ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_label("BestTimes at T= "+ str(Tsample) +" (0=QD)")
_save_analysis_fig(title)
plt.show()



# Create surface plot
plotly_title='bestTimes_surface_N='+str(nqubits)+" T="+str(Tsample)
fig = go.Figure(data=[go.Surface(z=BestTimes2D.T, x=x_arr, y=y_arr)])
fig.update_layout(title=plotly_title, scene=dict(
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    zaxis_title='Time'
))
_save_plotly_fig(fig, plotly_title)


_wait_for_plot_pane()

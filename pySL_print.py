#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:56:26 2025

@author: ALasek
"""

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


def pySL_print(filterDict={},subdir="", plot=False, oldData=False):

    # if len(filterDict)==0:
    #     filterDict={
    #     "psi_bias": 0.5,
    #     "Mironowicz_H_E": "Z",
    #     "Mironowicz_theta": np.pi/4
    #     }
        
    
    ld=runDataLoad(filterDict,subdir,oldData)
    
    for md in ld.matchdata:
        print(md[0])
        
    print("Matched "+str(len(ld.matchdata)) +' runs')
    
    if plot and oldData:
        ld.plotDataOld()
    elif plot and not oldData:
        ld.plotData()
    
    return ld.matchdata
    
if __name__ == "__main__":
    pySL_print()
    # pySL_print({},"Mironowicz_EE_alpha3sweep",True)
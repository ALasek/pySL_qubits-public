# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:40:35 2025

@author: Olek
"""

import numpy as np
import cupy as cp
import cuquantum
from cuquantum import custatevec



def apply_H(params,psi,H,pairs):
    # ---------------- PARAMETERS ----------------
    N = params["nqubits_S"]  +params["nqubits_E"]  # number of qubits
    dtypepbits = params.get("dtypepbits")
    if dtypepbits == 32:
        dtype = cuquantum.cudaDataType.CUDA_C_64F
    elif dtypepbits == 64:
        dtype = cuquantum.cudaDataType.CUDA_C_128F
    else:
        raise ValueError(f"Unsupported dtypepbits {dtypepbits!r}")
    compute_type = cuquantum.ComputeType.COMPUTE_DEFAULT
    layout = custatevec.MatrixLayout.COL
    # --------------------------------------------
    
    # Create a cuStateVec handle
    handle = custatevec.create()
    
    try:
        # Initialize GPU statevector |000...0>
        # dim = 1 << N
        # psi = cp.zeros(dim, dtype=cp.complex128)
        # psi[0] = 1.0
    
        # Build a simple 2-qubit gate (for example, exp(-i * XX * theta))
        # theta = 0.3
        # X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        # XX = np.kron(X, X)
        # U = np.cos(theta) * np.eye(4) - 1j * np.sin(theta) * XX
        # U_dev = cp.asarray(U)
        
        # H_dev=H.asarray(H)#?
        
        # Define disjoint pairs (even bonds for a 1D chain)
        # pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]
    
        # Create a CUDA stream per gate
        streams = [cp.cuda.Stream(non_blocking=True) for _ in pairs]
    
        # Apply each gate in its own stream
        for s, (q0, q1) in zip(streams, pairs):
            with s:
                custatevec.apply_matrix(
                    handle,
                    int(psi.data.ptr),
                    dtype,
                    N,
                    int(H.data.ptr),
                    dtype,
                    layout,
                    0,  # adjoint = False
                    [np.int32(q0), np.int32(q1)],
                    2,  # two targets
                    None, None, 0,  # no controls
                    compute_type,
                    0, 0  # no workspace
                )
    
        # Synchronize all streams (wait for all gates)
        for s in streams:
            s.synchronize()
    
        print("Appliped Trotter layer with", len(pairs), "disjoint gates.")
        print("Norm of si:", float(cp.linalg.norm(psi)))
    
    finally:
        custatevec.destroy(handle)

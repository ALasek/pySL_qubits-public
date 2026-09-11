#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:20:00 2025

@author: ALasek
"""
import math
import numpy as np
import pycuda.driver as cuda


import time

import cupy as cp
import random
import cupyx.scipy.sparse as cpsp
import warnings

from src.gpu_memory import gpu_memory_tracker

from scipy.linalg import expm,sqrtm, logm

def build_state_ud(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    state = cp.array([1], dtype=self.dtypec)  # start with scalar 1
    for n in range(self.nqubitsE):
        state = cp.kron(state, up if n%2==0 else down)
    return state
   

def build_state_mm(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    state = cp.array([1], dtype=self.dtypec)  # start with scalar 1
    for n in range(self.nqubitsE):
        state = cp.kron(state, (up + down) / cp.sqrt(2))
    return state


def build_state_bias(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    
    p=self.psi_bias
    
    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    state = cp.array([1], dtype=self.dtypec)  # start with scalar 1
    for n in range(self.nqubitsE):
        state = cp.kron(state, (np.sqrt(p)*up + np.sqrt(1-p)*down))
        state=state/cp.linalg.norm(state)
    return state

def build_state_Sspec(self):
    
    spec = self.psi_S_spec
    
    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    
    eye=cp.array([[1,0],[0,1]], dtype=self.dtypec)
    z = cp.array([[1,0],[0,-1]], dtype=self.dtypec)
    x = cp.array([[0,1],[1,0]], dtype=self.dtypec)
    y = cp.array([[0,-1j],[1j,0]], dtype=self.dtypec)

    specials=["Bell+","Bell-"] #etc

    psi_S = cp.array([1], dtype=self.dtypec)

    if spec in specials:
        match spec:
            case "Bell+":
                psi_S=cp.kron(up,down) + cp.kron(down,up)
            case "Bell-":
                psi_S=cp.kron(up,down) - cp.kron(down,up)
   
    else:
        
        for i in range(self.nqubitsS):
            Qspec=spec[2*i:2*i+2]
            
            
            match Qspec:
               case "x+":
                   psi_Q= (up + down) / cp.sqrt(2)
               case "x-":
                   psi_Q = (up - down) / cp.sqrt(2)
               case "y+":
                   psi_Q= (up + 1j*down) / cp.sqrt(2)
               case "y-":
                   psi_Q = (up - 1j*down) / cp.sqrt(2)
               case "z+":
                   psi_Q= up
               case "z-":
                   psi_Q= down
                 
            psi_S=cp.kron(psi_S,psi_Q)
        
    psi_S=psi_S.astype(self.dtypec)
    psi_S/=cp.linalg.norm(psi_S)
    return psi_S
       
def build_state_costumprod(self):
    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    state = cp.array([1], dtype=self.dtypec)  # start with scalar 1
    
    match self.psi_S_spec:
       case "x+":
           psi_q= (up + down) / cp.sqrt(2)
       case "x-":
           psi_q = (up - down) / cp.sqrt(2)
       case "y+":
           psi_q= (up + 1j*down) / cp.sqrt(2)
       case "y-":
           psi_q = (up - 1j*down) / cp.sqrt(2)
       case "z+":
           psi_q= up
       case "z-":
           psi_q= down
    
    for n in range(self.nqubitsE):
        state = cp.kron(state, psi_q)
        state=state/cp.linalg.norm(state)
    return state
    
def build_state_rand(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    rng = np.random.default_rng(int(self.seed))

    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    state = cp.array([1], dtype=self.dtypec)  # start with scalar 1
    for n in range(self.nqubitsE):
        theta = np.arccos(2 * rng.random() - 1)   # theta in [0, pi]
        phi   = 2 * np.pi * rng.random()           # phi in [0, 2*pi)
        state = cp.kron(state, np.cos(theta/2)*up + np.exp(1j*phi)*np.sin(theta/2)*down)
        state = state / cp.linalg.norm(state)
    return state

def build_state_UDrand(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    random.seed(int(self.seed))
    
   
   
    
    up = cp.array([1, 0], dtype=self.dtypec)
    down = cp.array([0, 1], dtype=self.dtypec)
    state = cp.array([1], dtype=self.dtypec)  # start with scalar 1
    for n in range(self.nqubitsE):
        p = random.uniform(0, 1)
        state = cp.kron(state, np.sqrt(p)*up + np.sqrt(1-p)*down)
        state=state/cp.linalg.norm(state)
    return state

def partial_trace_cp(psi, keep, dims):
    psi = cp.asarray(psi)
    total_qubits = len(dims)
    traced_out = [i for i in range(total_qubits) if i not in keep]

    reshaped_dims = [2] * total_qubits
    psi_tensor = cp.reshape(psi, reshaped_dims)

    perm = keep + traced_out
    psi_perm = cp.transpose(psi_tensor, axes=perm)

    dim_keep = 2 ** len(keep)
    dim_trace = 2 ** len(traced_out)

    psi_matrix = cp.reshape(psi_perm, (dim_keep, dim_trace))
        
    rho_reduced = psi_matrix @ psi_matrix.conj().T
    return rho_reduced

def vn_entropy_cp(rho):
    evals = cp.linalg.eigvalsh(rho)
    evals = evals[evals > 0]  # Avoid log(0)
    entropy = -cp.sum(evals * cp.log(evals))
    return entropy


def vn_entropy_np(rho):
    evals = np.linalg.eigvalsh(np.asarray(rho))
    evals = np.real(evals)
    evals = evals[evals > 0]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log(evals)))



def von_neumann_entropy_direct(psi, keep_qubits, total_qubits):
    all_qubits = list(range(total_qubits))
    keep_qubits = list(dict.fromkeys(keep_qubits))
    keep_set = set(keep_qubits)

    if len(keep_qubits) > total_qubits // 2:
        keep_qubits = [q for q in all_qubits if q not in keep_set]

    # Step 1: reshape to tensor
    psi_tensor = psi.reshape([2] * total_qubits)

    # Step 2: permute axes
    trace_qubits = [q for q in all_qubits if q not in keep_qubits]
    perm = keep_qubits + trace_qubits
    psi_tensor = cp.transpose(psi_tensor, perm)

    # Step 3: reshape to matrix
    psi_matrix = psi_tensor.reshape((2**len(keep_qubits), -1))

    # Diagonalize rho on the smaller bipartition; SVD needs much larger cuSOLVER workspace.
    rho = psi_matrix @ psi_matrix.conj().T
    gpu_memory_tracker.sample("entropy reduced density matrix")
    evals = cp.linalg.eigvalsh(rho)
    gpu_memory_tracker.sample("entropy eigvalsh")
    evals = cp.maximum(cp.real(evals), 0.0)
    evals = evals[evals > 1e-12]

    entropy = -cp.sum(evals * cp.log(evals))
    return entropy.item()

def mutual_information_direct(psi, A,B,N,skipAB=False):

    if skipAB:
        return max(0,von_neumann_entropy_direct(psi,A,N)+von_neumann_entropy_direct(psi,B,N) )
    else:
        return max(0,von_neumann_entropy_direct(psi,A,N)+von_neumann_entropy_direct(psi,B,N) - von_neumann_entropy_direct(psi,A+B,N))


def trace_distance_fractions(psi,A):
    SBS0_psi=psi.pointerProj0 @ psi.psi
    SBS1_psi=psi.pointerProj1 @ psi.psi
    
    SBS0_psi/=cp.linalg.norm(SBS0_psi)
    SBS1_psi/=cp.linalg.norm(SBS1_psi)
    
    rho_E_i_0=partial_trace_cp(SBS0_psi,A,[2]*psi.nqubitsE).get()
    rho_E_i_1=partial_trace_cp(SBS1_psi,A,[2]*psi.nqubitsE).get()
    
    td=trace_distance(rho_E_i_0, rho_E_i_1)
    
    return td
       

def mutual_information(psi, A,B,dims):
    psi_A=partial_trace_cp(psi, A, dims)
    psi_B=partial_trace_cp(psi, B, dims)
    psi_AB=partial_trace_cp(psi, A+B, dims)
    return vn_entropy_cp(psi_A)+vn_entropy_cp(psi_B) - vn_entropy_cp(psi_AB)

def apply_two_qubit_gate(G, i, j, num_qubits):
    # Ensure i < j
    if i > j:
        i, j = j, i

    dim = 2 ** num_qubits

    # Create full operator by embedding G
    # Step 1: reshape state space to tensor form
    G_full = np.eye(dim, dtype=complex).reshape([2] * num_qubits * 2)

    # Step 2: apply G to qubits i and j
    for idx in range(dim):
        for jdx in range(dim):
            # Convert indices to binary strings
            idx_bits = list(np.binary_repr(idx, width=num_qubits))
            jdx_bits = list(np.binary_repr(jdx, width=num_qubits))

            # Extract bits for qubits i and j
            idx_sub = int(idx_bits[i] + idx_bits[j], 2)
            jdx_sub = int(jdx_bits[i] + jdx_bits[j], 2)

            # Replace bits with G[idx_sub, jdx_sub]
            if idx_bits[:i] + idx_bits[i+1:j] + idx_bits[j+1:] == jdx_bits[:i] + jdx_bits[i+1:j] + jdx_bits[j+1:]:
                G_full[tuple(map(int, idx_bits + jdx_bits))] = G[idx_sub, jdx_sub]

    # Step 3: reshape back to matrix form
    H = G_full.reshape((dim, dim))
    return H

def embedded_two_qubit_entries_cp(G, a, b, num_qubits, dtypecp=cp.complex64):
    if not (0 <= a < b < num_qubits):
        raise ValueError(f"Expected qubit indices 0 <= a < b < {num_qubits}, got a={a}, b={b}")

    G = cp.asarray(G, dtype=dtypecp)
    nz_rows, nz_cols = cp.nonzero(G)
    if nz_rows.size == 0:
        empty = cp.empty(0, dtype=cp.int64)
        return empty, empty, cp.empty(0, dtype=dtypecp)

    dim = 2 ** num_qubits
    cols_all = cp.arange(dim, dtype=cp.int64)
    bit_a = num_qubits - 1 - a
    bit_b = num_qubits - 1 - b
    mask_a = cp.int64(1 << bit_a)
    mask_b = cp.int64(1 << bit_b)
    local_cols_all = (((cols_all >> bit_a) & 1) << 1) | ((cols_all >> bit_b) & 1)

    row_parts = []
    col_parts = []
    data_parts = []
    for k in range(int(nz_rows.size)):
        local_row = int(nz_rows[k].item())
        local_col = int(nz_cols[k].item())
        cols = cols_all[local_cols_all == local_col]
        row_a = (local_row >> 1) & 1
        row_b = local_row & 1
        rows = cols & ~mask_a & ~mask_b
        if row_a:
            rows = rows | mask_a
        if row_b:
            rows = rows | mask_b
        row_parts.append(rows)
        col_parts.append(cols)
        data_parts.append(cp.full(cols.shape, G[local_row, local_col], dtype=dtypecp))

    return cp.concatenate(row_parts), cp.concatenate(col_parts), cp.concatenate(data_parts)


def _numpy_dtype_for_cupy(dtypecp):
    dtype = cp.dtype(dtypecp)
    if dtype == cp.dtype(cp.complex64):
        return np.complex64
    if dtype == cp.dtype(cp.complex128):
        return np.complex128
    if dtype == cp.dtype(cp.float32):
        return np.float32
    if dtype == cp.dtype(cp.float64):
        return np.float64
    return np.dtype(dtype.name)


def _as_numpy_matrix(matrix, dtype):
    if isinstance(matrix, cp.ndarray):
        matrix = cp.asnumpy(matrix)
    return np.asarray(matrix, dtype=dtype)


def embedded_two_qubit_entries_np(G, a, b, num_qubits, dtypecp=cp.complex64):
    if not (0 <= a < b < num_qubits):
        raise ValueError(f"Expected qubit indices 0 <= a < b < {num_qubits}, got a={a}, b={b}")

    dtype = _numpy_dtype_for_cupy(dtypecp)
    G = _as_numpy_matrix(G, dtype)
    nz_rows, nz_cols = np.nonzero(G)
    if nz_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=dtype)

    dim = 2 ** num_qubits
    cols_all = np.arange(dim, dtype=np.int64)
    bit_a = num_qubits - 1 - a
    bit_b = num_qubits - 1 - b
    mask_a = np.int64(1 << bit_a)
    mask_b = np.int64(1 << bit_b)
    local_cols_all = (((cols_all >> bit_a) & 1) << 1) | ((cols_all >> bit_b) & 1)

    row_parts = []
    col_parts = []
    data_parts = []
    for local_row, local_col in zip(nz_rows, nz_cols):
        cols = cols_all[local_cols_all == local_col]
        row_a = (int(local_row) >> 1) & 1
        row_b = int(local_row) & 1
        rows = cols & ~mask_a & ~mask_b
        if row_a:
            rows = rows | mask_a
        if row_b:
            rows = rows | mask_b
        row_parts.append(rows)
        col_parts.append(cols)
        data_parts.append(np.full(cols.shape, G[local_row, local_col], dtype=dtype))

    return np.concatenate(row_parts), np.concatenate(col_parts), np.concatenate(data_parts)


def apply_two_qubit_gate_cp(G, a, b, num_qubits, dtypecp=cp.complex64):
    rows, cols, data = embedded_two_qubit_entries_cp(G, a, b, num_qubits, dtypecp=dtypecp)
    return cpsp.coo_matrix((data, (rows, cols)), shape=(2**num_qubits, 2**num_qubits), dtype=dtypecp).tocsr()


def embedded_one_qubit_entries_cp(G, a, num_qubits, dtypecp=cp.complex64):
    if not (0 <= a < num_qubits):
        raise ValueError(f"Expected qubit index 0 <= a < {num_qubits}, got a={a}")

    G = cp.asarray(G, dtype=dtypecp)
    nz_rows, nz_cols = cp.nonzero(G)
    if nz_rows.size == 0:
        empty = cp.empty(0, dtype=cp.int64)
        return empty, empty, cp.empty(0, dtype=dtypecp)

    dim = 2 ** num_qubits
    cols_all = cp.arange(dim, dtype=cp.int64)
    bit = num_qubits - 1 - a
    mask = cp.int64(1 << bit)
    local_cols_all = (cols_all >> bit) & 1

    row_parts = []
    col_parts = []
    data_parts = []
    for k in range(int(nz_rows.size)):
        local_row = int(nz_rows[k].item())
        local_col = int(nz_cols[k].item())
        cols = cols_all[local_cols_all == local_col]
        rows = cols & ~mask
        if local_row:
            rows = rows | mask
        row_parts.append(rows)
        col_parts.append(cols)
        data_parts.append(cp.full(cols.shape, G[local_row, local_col], dtype=dtypecp))

    return cp.concatenate(row_parts), cp.concatenate(col_parts), cp.concatenate(data_parts)


def embedded_one_qubit_entries_np(G, a, num_qubits, dtypecp=cp.complex64):
    if not (0 <= a < num_qubits):
        raise ValueError(f"Expected qubit index 0 <= a < {num_qubits}, got a={a}")

    dtype = _numpy_dtype_for_cupy(dtypecp)
    G = _as_numpy_matrix(G, dtype)
    nz_rows, nz_cols = np.nonzero(G)
    if nz_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=dtype)

    dim = 2 ** num_qubits
    cols_all = np.arange(dim, dtype=np.int64)
    bit = num_qubits - 1 - a
    mask = np.int64(1 << bit)
    local_cols_all = (cols_all >> bit) & 1

    row_parts = []
    col_parts = []
    data_parts = []
    for local_row, local_col in zip(nz_rows, nz_cols):
        cols = cols_all[local_cols_all == local_col]
        rows = cols & ~mask
        if int(local_row):
            rows = rows | mask
        row_parts.append(rows)
        col_parts.append(cols)
        data_parts.append(np.full(cols.shape, G[local_row, local_col], dtype=dtype))

    return np.concatenate(row_parts), np.concatenate(col_parts), np.concatenate(data_parts)


def apply_one_qubit_gate_cp(G, a, num_qubits, dtypecp=cp.complex64):
    rows, cols, data = embedded_one_qubit_entries_cp(G, a, num_qubits, dtypecp=dtypecp)
    return cpsp.coo_matrix((data, (rows, cols)), shape=(2**num_qubits, 2**num_qubits), dtype=dtypecp).tocsr()


def kron_n(ops):
    """Kronecker product of a list of operators"""
    result = ops[0]
    for op in ops[1:]:
        result = cpsp.kron(result, op, format='csr')
    return result
def embed_two_qubit_op(op1, op2, N, i, j,dtype=cp.complex64):
    I = cpsp.eye(2, format='csr',dtype=dtype)
    """Embed a two-qubit operator into N-qubit space"""
    ops = []
    for k in range(N):
        if k == i:
            ops.append(op1)
        elif k == j:
            ops.append(op2)
        else:
            ops.append(I)
    return kron_n(ops)

def swap_gate(N, i, j,dtype=cp.complex64):
    """Construct SWAP gate for N qubits acting on qubits i and j"""
    if i == j:
        return cpsp.identity(2 ** N, dtype=dtype, format='csr')

    # Define Pauli matrices as sparse
    I = cpsp.eye(2, format='csr',dtype=cp.complex64)
    X = cpsp.csr_matrix(cp.array([[0, 1], [1, 0]],dtype=dtype))
    Y = cpsp.csr_matrix(cp.array([[0, -1j], [1j, 0]],dtype=dtype))
    Z = cpsp.csr_matrix(cp.array([[1, 0], [0, -1]],dtype=dtype))

    swap = (
        embed_two_qubit_op(I, I, N, i, j,dtype) +
        embed_two_qubit_op(X, X, N, i, j,dtype) +
        embed_two_qubit_op(Y, Y, N, i, j,dtype) +
        embed_two_qubit_op(Z, Z, N, i, j,dtype)
    ) * 0.5
    return swap

def trace_distance(rho, sigma):
    """
    Compute the trace distance: 1 * || rho - sigma ||_1
    where ||A||_1 = Tr( sqrt(A A) ).
    """
    delta = rho - sigma
    # Singular values of delta = eigenvalues of sqrt(delta delta)
    singular_vals = np.linalg.svd(delta, compute_uv=False)
    return _unit_interval(0.5 * np.sum(np.abs(singular_vals)))


def quantum_relative_entropy(rho, sigma):
    """
    Compute the quantum relative entropy:
    S(rho || sigma) = Tr[rho (log rho - log sigma)]
    
    Assumes full-rank sigma; otherwise divergence may occur.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            log_rho = logm(rho)
            log_sigma = logm(sigma)
            value = np.real(np.trace(rho @ (log_rho - log_sigma)))
        except (RuntimeWarning, ValueError, np.linalg.LinAlgError):
            return np.nan
    if not np.isfinite(value):
        return np.nan
    return value

def fidelity(rho, sigma):
    """
    Compute the quantum fidelity between two density matrices:
    F(rho, sigma) = ( Tr[ sqrt( sqrt(rho) * sigma * sqrt(rho) ) ] )^2
    """
    # sqrt(rho)
    sqrt_rho = sqrtm(rho)

    # A = sqrt(rho) * sigma * sqrt(rho)
    A = sqrt_rho @ sigma @ sqrt_rho

    # sqrt(A)
    sqrt_A = sqrtm(A)

    # Fidelity = (Tr sqrt(A))^2, real part only
    value = np.real((np.trace(sqrt_A))**2)
    return _unit_interval(value)


def _unit_interval(value):
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0.0, 1.0))

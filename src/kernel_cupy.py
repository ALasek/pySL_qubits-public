# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:05:31 2025

@author: Olek
"""

import cupy as cp
import numpy as np

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

def apply_two_body(psi, h_ij, i, j):
    """
    Apply 2-qubit operator h_ij (4x4) to sites (i,j) of N-qubit state psi.
    psi: cupy array, shape (2,)*N
    h_ij: cupy array, shape (4,4)
    i, j: qubit indices (0=leftmost)
    """
    # reshape h_ij into 2x2x2x2 tensor
    h_tensor = h_ij.reshape(2,2,2,2)

    # move qubits i,j to front
    perm = (i,j) + tuple(k for k in range(psi.ndim) if k not in (i,j))
    psi_perm = psi.transpose(perm)

    # shape (2,2,dim_rest)
    dim_rest = psi_perm.size // 4
    psi_block = psi_perm.reshape(2,2,dim_rest)

    # contract h_tensor with psi_block
    out_block = cp.tensordot(h_tensor, psi_block, axes=([2,3],[0,1]))  # shape (2,2,dim_rest)

    # reshape back
    out_perm = out_block.reshape(psi_perm.shape).transpose(cp.argsort(cp.array(perm)))

    return out_perm

def apply_H(psi, terms):
    # terms is list of (h_ij, i, j)
    result = cp.zeros_like(psi)
    for h_ij, i, j in terms:
        result += apply_two_body(psi, h_ij, i, j)
    return result


def leapfrog_evolve(psi0, dt, n_steps, terms):
    u = psi0.real.astype(cp.complex128)
    v = psi0.imag.astype(cp.complex128)
    for _ in range(n_steps):
        v -= 0.5 * dt * apply_H(u, terms)
        u += dt * apply_H(v, terms)
        v -= 0.5 * dt * apply_H(u, terms)

    return u + 1j * v


def evolve_euler(psi, H, dt):
    return psi - 1j * H @ psi * dt

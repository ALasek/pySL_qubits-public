# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:05:31 2025

@author: Olek
"""

import cupy as cp

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



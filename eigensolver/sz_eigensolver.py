"""Block-diagonalization eigensolver for Hamiltonians with Sz conservation.

Adapted for the Mironowicz_rand_EE model (and variants) from pySL_qubits2.
CPU-only (numpy / scipy sparse) — no GPU or CuPy required.

Approach
--------
For a Hamiltonian that commutes with total Sz = (1/2) Σ_i Z_i, the Hilbert
space decomposes into sectors labelled by the number of down-spins k (Hamming
weight of the basis-state index).  Sector k has dimension C(n, k) and total
Sz = n/2 - k.  Basis states are just the computational-basis states grouped by
Hamming weight — no Clebsch-Gordan coefficients needed.

Sz conservation is tested by checking the commutator [h_2body, Sz_2q] on
every 2-body interaction template.  For nearest-neighbour models this is
sufficient: if the local 2-body term conserves Sz then the full embedded
operator commutes with total Sz automatically.

Qubit convention (MSB-first, same as apply_two_qubit_gate_cp in Qutils.py):
  basis index idx  →  qubit k = (idx >> (n-1-k)) & 1
  qubit 0 is the most-significant bit.
"""

import random
import numpy as np
from scipy import sparse as sp


# --------------------------------------------------------------------------- #
# Pauli matrices (complex128)
# --------------------------------------------------------------------------- #

_I2 = np.eye(2, dtype=np.complex128)
_Z  = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
_X  = np.array([[0., 1.], [1.,  0.]], dtype=np.complex128)
_Y  = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)

_SIGMAS = {'I': _I2, 'X': _X, 'Y': _Y, 'Z': _Z}

_Sz_1q = 0.5 * _Z
_Sz_2q = np.kron(_Sz_1q, _I2) + np.kron(_I2, _Sz_1q)


# --------------------------------------------------------------------------- #
# Sparse embedding utilities
# --------------------------------------------------------------------------- #

def _kron_chain(ops):
    result = ops[0]
    for op in ops[1:]:
        result = sp.kron(result, op, format='csr')
    return result


def embed_one(op_2x2, k, n):
    """Embed a 2×2 operator on qubit k in the n-qubit Hilbert space.

    Returns a (2^n, 2^n) scipy sparse CSR matrix.
    """
    I = sp.eye(2, format='csr', dtype=np.complex128)
    ops = [sp.csr_matrix(op_2x2.astype(np.complex128)) if i == k else I
           for i in range(n)]
    return _kron_chain(ops)


def embed_two(op_4x4, a, b, n):
    """Embed a 4×4 operator on qubits a, b (a < b) in the n-qubit space.

    Row/column indexing of op_4x4: index = 2 * bit_a + bit_b
    (same convention as apply_two_qubit_gate_cp in Qutils.py).

    Returns a (2^n, 2^n) scipy sparse CSR matrix.
    """
    assert 0 <= a < b < n, f"Need 0 <= a < b < n, got a={a}, b={b}, n={n}"
    dim   = 2 ** n
    sa    = n - 1 - a   # bit-shift amount for qubit a (MSB-first)
    sb    = n - 1 - b   # bit-shift amount for qubit b

    idx   = np.arange(dim, dtype=np.intp)
    bit_a = (idx >> sa) & 1
    bit_b = (idx >> sb) & 1
    rest  = idx & ~((1 << sa) | (1 << sb))   # index with bits a, b zeroed

    rows_list, cols_list, data_list = [], [], []

    for ra in range(2):
        for rb in range(2):
            row_mask   = (bit_a == ra) & (bit_b == rb)
            row_states = idx[row_mask]
            row_rest   = rest[row_mask]
            r_order    = np.argsort(row_rest)
            row_states = row_states[r_order]   # sorted by rest bits

            for ca in range(2):
                for cb in range(2):
                    g = op_4x4[ra * 2 + rb, ca * 2 + cb]
                    if abs(g) < 1e-15:
                        continue

                    col_mask   = (bit_a == ca) & (bit_b == cb)
                    col_states = idx[col_mask]
                    col_rest   = rest[col_mask]
                    c_order    = np.argsort(col_rest)
                    col_states = col_states[c_order]

                    # row_states[k] and col_states[k] share the same rest bits
                    rows_list.append(row_states)
                    cols_list.append(col_states)
                    data_list.append(np.full(len(row_states), g, dtype=np.complex128))

    if not data_list:
        return sp.csr_matrix((dim, dim), dtype=np.complex128)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))


# --------------------------------------------------------------------------- #
# Sz conservation check
# --------------------------------------------------------------------------- #

def _commutator_max(A, B):
    """Max |[A, B]| = max |AB - BA| (for small dense matrices)."""
    return np.max(np.abs(A @ B - B @ A))


def check_sz_conservation(params, tol=1e-10):
    """Check whether every interaction term commutes with total Sz.

    Tests the 2-body (and 1-body) templates — sufficient for all-to-all or
    nearest-neighbour models because if [h_ij, Sz_i + Sz_j] = 0 then
    [embed(h_ij), Sz_total] = 0.

    Returns
    -------
    conserved : bool
    reason    : str  (explanation / which term fails)
    """
    special = params.get("H_SE_Special", 0)

    # --- SE interaction template ---
    if special in ("Mironowicz", "Mironowicz_rand", "Mironowicz_rand_EE",
                   "Mironowicz_rand_EE_transverse", "Mironowicz_rand_EE_ZX",
                   "Mironowicz_rand_EE_ZZ_to_ZX",
                   "Mironowicz_rand_epsilon", "Mironowicz_rand_mod_epsilon"):
        theta = params.get("Mironowicz_theta", 0.0)
        c_th  = np.cos(theta)
        s_th  = np.sin(theta)

        H_SE = np.zeros((4, 4), dtype=np.complex128)
        H_SE[2, 2] = (np.pi / 2) * (1 - s_th)
        H_SE[3, 3] = (np.pi / 2) * (1 + s_th)
        H_SE[2, 3] = -(np.pi / 2) * c_th
        H_SE[3, 2] = -(np.pi / 2) * c_th

        if special in ("Mironowicz_rand_epsilon", "Mironowicz_rand_mod_epsilon"):
            epsilon = params.get("Mironowicz_epsilon", 0.0)
            H_SE += epsilon * (
                np.kron(_X, c_th * _Z) - np.kron(_X, s_th * _X)
            )
        if special == "Mironowicz_rand_mod_epsilon":
            epsilon2 = params.get("Mironowicz_epsilon2", 0.0)
            H_SE += epsilon2 * np.kron(_Y, _Y)

        if _commutator_max(H_SE, _Sz_2q) > tol:
            return (False,
                    f"H_SE (Mironowicz, theta={theta:.4f}) breaks Sz "
                    f"[cos(theta)={c_th:.4g} must be 0 for Sz conservation]")

    elif special == "ZZ_rand":
        # H_SE = Z⊗Z, diagonal → always conserves Sz
        pass

    else:
        # Generic model: build H_SE from H_SE parameter list
        H_SE_terms = params.get("H_SE", [])
        H_SE = sum(c * np.kron(_SIGMAS[s1], _SIGMAS[s2])
                   for c, s1, s2 in H_SE_terms) if H_SE_terms else np.zeros((4, 4))
        if _commutator_max(H_SE, _Sz_2q) > tol:
            return False, "H_SE (generic) breaks Sz"

    # --- H_EE template ---
    if special == "Mironowicz_rand_EE_transverse":
        H_EE = 0.5 * (np.kron(_Z, _I2) + np.kron(_Z, _X) + np.kron(_Z, _Y))
        if _commutator_max(H_EE, _Sz_2q) > tol:
            return False, "H_EE transverse term breaks Sz"
    elif special == "Mironowicz_rand_EE_ZX":
        H_EE = np.kron(_Z, _X)
        if _commutator_max(H_EE, _Sz_2q) > tol:
            return False, "H_EE ZX term breaks Sz"
    elif special == "Mironowicz_rand_EE_ZZ_to_ZX":
        epsilon = params.get("Mironowicz_ZZ_to_ZX_epsilon", 0.0)
        H_EE = np.kron(_Z, np.cos(epsilon) * _Z + np.sin(epsilon) * _X)
        if _commutator_max(H_EE, _Sz_2q) > tol:
            return False, (
                "H_EE ZZ_to_ZX term breaks Sz "
                f"[sin(epsilon)={np.sin(epsilon):.4g} must be 0]"
            )
    else:
        H_EE_terms = params.get("H_EE", [])
        H_EE = sum(c * np.kron(_SIGMAS[s1], _SIGMAS[s2])
                   for c, s1, s2 in H_EE_terms) if H_EE_terms else None
    if H_EE is not None:
        if _commutator_max(H_EE, _Sz_2q) > tol:
            return False, "H_EE breaks Sz"

    # --- Single-qubit E term ---
    miro_H_E = params.get("Mironowicz_H_E")
    if miro_H_E is not None:
        if _commutator_max(_SIGMAS[miro_H_E], _Sz_1q) > tol:
            return False, f"Mironowicz_H_E = '{miro_H_E}' breaks Sz (only 'Z' is safe)"
    else:
        H_E_terms = params.get("H_E", [])
        if H_E_terms:
            H_E = sum(c * _SIGMAS[s] for c, s in H_E_terms)
            if _commutator_max(H_E, _Sz_1q) > tol:
                return False, "H_E (generic) breaks Sz"

    return True, "All interaction terms commute with Sz"


# --------------------------------------------------------------------------- #
# Hamiltonian builder (CPU, scipy sparse, no GPU)
# --------------------------------------------------------------------------- #

def _draw_J(distribution, sigma):
    """Draw a coupling constant matching hamiltonian.py conventions."""
    if distribution == "Norm":
        return random.gauss(0, sigma)
    else:  # "Const"
        return sigma


def build_H_numpy(params, seed=None):
    """Assemble the full Hamiltonian as a scipy sparse CSR matrix.

    Replicates the logic of hamiltonian.__init__ + precomputeH_cpspCombine
    in pure numpy/scipy, with the same random-draw sequence as the GPU code.

    Supports: Mironowicz, Mironowicz_rand, Mironowicz_rand_EE,
              Mironowicz_rand_EE_transverse, Mironowicz_rand_EE_ZX,
              Mironowicz_rand_EE_ZZ_to_ZX, Mironowicz_rand_epsilon,
              Mironowicz_rand_mod_epsilon, ZZ_rand, and the generic model.
    Assumes nqubits_S = 1 for Mironowicz variants (as in all current configs).

    Parameters
    ----------
    params : dict   Parameter dict (same schema as pySL.py / batch_configs).
    seed   : int    Random seed; if None, uses params["seed"].

    Returns
    -------
    H : scipy sparse CSR matrix, shape (2^n, 2^n), dtype complex128
    """
    special = params.get("H_SE_Special", 0)
    nS      = params["nqubits_S"]
    nE      = params["nqubits_E"]
    n       = nS + nE
    dim     = 2 ** n

    if seed is None:
        seed = params.get("seed", 0)
    random.seed(seed)

    H = sp.csr_matrix((dim, dim), dtype=np.complex128)

    # ------------------------------------------------------------------ #
    # Mironowicz family
    # ------------------------------------------------------------------ #
    if special in ("Mironowicz", "Mironowicz_rand", "Mironowicz_rand_EE",
                   "Mironowicz_rand_EE_transverse", "Mironowicz_rand_EE_ZX",
                   "Mironowicz_rand_EE_ZZ_to_ZX",
                   "Mironowicz_rand_epsilon", "Mironowicz_rand_mod_epsilon"):

        theta   = params["Mironowicz_theta"]
        H_SE_J  = params["H_SE_J"]
        alpha2  = params["Mironowicz_alpha2"]
        h0      = params.get("Mironowicz_h0", 0.0)

        # Build 2-qubit SE template
        c_th, s_th = np.cos(theta), np.sin(theta)
        H_SE_2q = np.zeros((4, 4), dtype=np.complex128)
        H_SE_2q[2, 2] = (np.pi / 2) * (1 - s_th)
        H_SE_2q[3, 3] = (np.pi / 2) * (1 + s_th)
        H_SE_2q[2, 3] = -(np.pi / 2) * c_th
        H_SE_2q[3, 2] = -(np.pi / 2) * c_th

        if special in ("Mironowicz_rand_epsilon", "Mironowicz_rand_mod_epsilon"):
            eps = params.get("Mironowicz_epsilon", 0.0)
            H_SE_2q += eps * (np.kron(_X, c_th * _Z) - np.kron(_X, s_th * _X))
        if special == "Mironowicz_rand_mod_epsilon":
            eps2 = params.get("Mironowicz_epsilon2", 0.0)
            H_SE_2q += eps2 * np.kron(_Y, _Y)

        # SE bonds: [ns, ne + nS] for all (ns, ne) pairs
        # random sequence matches _build_mironowicz_rand
        for ns in range(nS):
            for ne in range(nE):
                J = H_SE_J if special == "Mironowicz" else random.gauss(0, H_SE_J)
                H += J * embed_two(H_SE_2q, ns, ne + nS, n)

        # Single-qubit E self-energy
        # hamiltonian.py reseeds before drawing J_E values
        random.seed(seed)
        H_E_key = params.get("Mironowicz_H_E", "Z")
        H_E_1q  = _SIGMAS[H_E_key]
        for i in range(nE):
            J = alpha2 if special == "Mironowicz" else h0 + random.gauss(0, alpha2)
            if J != 0:
                H += J * embed_one(H_E_1q, i + nS, n)

        # EE interactions (rand_EE variant only; J_EE drawn after J_E)
        if special == "Mironowicz_rand_EE":
            H = _add_EE(H, params, n, nS, nE)
        elif special == "Mironowicz_rand_EE_transverse":
            H = _add_EE_transverse(H, params, n, nS, nE)
        elif special == "Mironowicz_rand_EE_ZX":
            H = _add_EE_ZX(H, params, n, nS, nE)
        elif special == "Mironowicz_rand_EE_ZZ_to_ZX":
            H = _add_EE_ZZ_to_ZX(H, params, n, nS, nE)

    # ------------------------------------------------------------------ #
    # Pure ZZ S-E coupling
    # ------------------------------------------------------------------ #
    elif special == "ZZ_rand":
        H_SE_J  = params["H_SE_J"]
        H_SE_2q = np.kron(_Z, _Z)
        for ns in range(nS):
            for ne in range(nE):
                J = random.gauss(0, H_SE_J)
                H += J * embed_two(H_SE_2q, ns, ne + nS, n)

    # ------------------------------------------------------------------ #
    # Generic model (falls through to EE via _add_EE)
    # ------------------------------------------------------------------ #
    else:
        H_SE_bonds = params.get("H_SE_bonds", "S_to_all")
        H_SE_J     = params.get("H_SE_J", 0.0)
        H_SE_p     = params.get("H_SE_p", "Norm")
        H_SE_terms = params.get("H_SE", [])
        H_SE_2q    = sum(c * np.kron(_SIGMAS[s1], _SIGMAS[s2])
                         for c, s1, s2 in H_SE_terms)

        bonds = ([(ns, ne + nS) for ns in range(nS) for ne in range(nE)]
                 if H_SE_bonds == "S_to_all" else
                 [(b[0], b[1]) for b in H_SE_bonds])

        for a, b in bonds:
            J = _draw_J(H_SE_p, H_SE_J)
            H += J * embed_two(H_SE_2q, a, b, n)

        H_E_terms = params.get("H_E", [])
        if H_E_terms:
            H_E_J = params.get("H_E_J", 0.0)
            H_E_p = params.get("H_E_p", "Norm")
            H_E_1q = sum(c * _SIGMAS[s] for c, s in H_E_terms)
            for i in range(nE):
                J = _draw_J(H_E_p, H_E_J)
                if J != 0:
                    H += J * embed_one(H_E_1q, i + nS, n)

        H = _add_EE(H, params, n, nS, nE)

    return H


def _add_EE(H, params, n, nS, nE):
    """Add nearest-neighbour E-E interactions (mirrors _add_EE_interactions)."""
    H_EE_bonds = params.get("H_EE_bonds", "1DNN")
    H_EE_J     = params.get("H_EE_J", 0.0)
    H_EE_p     = params.get("H_EE_p", "Norm")
    H_EE_terms = params.get("H_EE", [])

    if not H_EE_terms:
        return H

    H_EE_2q = sum(c * np.kron(_SIGMAS[s1], _SIGMAS[s2])
                  for c, s1, s2 in H_EE_terms)

    if H_EE_bonds == "1DNN":
        for ne in range(nE - 1):
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, ne + nS, ne + nS + 1, n)
    else:
        for bond in H_EE_bonds:
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, bond[0], bond[1], n)

    return H


def _add_EE_transverse(H, params, n, nS, nE):
    H_EE_bonds = params.get("H_EE_bonds", "1DNN")
    H_EE_J     = params.get("H_EE_J", 0.0)
    H_EE_p     = params.get("H_EE_p", "Norm")
    H_EE_2q    = 0.5 * (np.kron(_Z, _I2) + np.kron(_Z, _X) + np.kron(_Z, _Y))

    if H_EE_bonds == "1DNN":
        for ne in range(nE - 1):
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, ne + nS, ne + nS + 1, n)
    else:
        for bond in H_EE_bonds:
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, bond[0], bond[1], n)

    return H


def _add_EE_ZX(H, params, n, nS, nE):
    H_EE_bonds = params.get("H_EE_bonds", "1DNN")
    H_EE_J     = params.get("H_EE_J", 0.0)
    H_EE_p     = params.get("H_EE_p", "Norm")
    H_EE_2q    = np.kron(_Z, _X)

    if H_EE_bonds == "1DNN":
        for ne in range(nE - 1):
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, ne + nS, ne + nS + 1, n)
    else:
        for bond in H_EE_bonds:
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, bond[0], bond[1], n)

    return H


def _add_EE_ZZ_to_ZX(H, params, n, nS, nE):
    H_EE_bonds = params.get("H_EE_bonds", "1DNN")
    H_EE_J     = params.get("H_EE_J", 0.0)
    H_EE_p     = params.get("H_EE_p", "Norm")
    epsilon    = params.get("Mironowicz_ZZ_to_ZX_epsilon", 0.0)
    H_EE_2q    = np.kron(_Z, np.cos(epsilon) * _Z + np.sin(epsilon) * _X)

    if H_EE_bonds == "1DNN":
        for ne in range(nE - 1):
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, ne + nS, ne + nS + 1, n)
    else:
        for bond in H_EE_bonds:
            J = _draw_J(H_EE_p, H_EE_J)
            H += J * embed_two(H_EE_2q, bond[0], bond[1], n)

    return H


# --------------------------------------------------------------------------- #
# Sz sector decomposition
# --------------------------------------------------------------------------- #

def sz_sectors(n):
    """Group computational basis states by Hamming weight (number of down-spins).

    Returns
    -------
    dict  k (int) -> np.ndarray of basis indices with exactly k ones.
          k = 0 is all-up (total Sz = +n/2), k = n is all-down (Sz = -n/2).
          Entries are sorted by basis index.
    """
    sectors = {}
    for idx in range(2 ** n):
        k = bin(idx).count('1')
        sectors.setdefault(k, []).append(idx)
    return {k: np.array(v, dtype=np.intp) for k, v in sorted(sectors.items())}


# --------------------------------------------------------------------------- #
# Block eigensolver (main entry point)
# --------------------------------------------------------------------------- #

def block_eigensolver(params, seed=None, tol=1e-10, verbose=True):
    """Build the Hamiltonian, check Sz conservation, and diagonalize.

    If Sz is conserved (to tolerance tol): diagonalizes each Sz sector
    independently using dense numpy.linalg.eigh.

    If Sz is NOT conserved: falls back to a full diagonalization of H.
    Warns for n > 12 qubits where the full dense matrix is large.

    Parameters
    ----------
    params  : dict   pySL parameter dict.
    seed    : int    Random seed (defaults to params["seed"]).
    tol     : float  Commutator threshold for Sz conservation test.
    verbose : bool   Print conservation status and sector sizes.

    Returns
    -------
    dict with keys:
      "eigenvalues" : 1-D array, all eigenvalues sorted ascending
      "sz_labels"   : 1-D array of total Sz per eigenvalue (None if not conserved)
      "conserved"   : bool
      "n_sectors"   : int or None
    """
    n = params["nqubits_S"] + params["nqubits_E"]

    conserved, msg = check_sz_conservation(params, tol)
    if verbose:
        tag = "CONSERVED" if conserved else "NOT CONSERVED"
        print(f"[Sz] {tag}: {msg}")

    H = build_H_numpy(params, seed)

    if conserved:
        sectors   = sz_sectors(n)
        all_evals = []
        all_sz    = []

        for k, indices in sectors.items():
            m_z   = n / 2 - k
            block = H[indices[:, None], indices[None, :]].toarray()

            if not np.allclose(block, block.conj().T, atol=max(tol, 1e-8)):
                raise ValueError(f"Sz block k={k} is not Hermitian (max imag={np.max(np.abs(block - block.conj().T)):.2e})")

            evals = np.linalg.eigh(block)[0]
            all_evals.append(evals)
            all_sz.append(np.full(len(evals), m_z))

            if verbose:
                print(f"  Sz={m_z:+.1f}  (k={k})  dim={len(indices):5d}  "
                      f"E_min={evals[0]:.6f}  E_max={evals[-1]:.6f}")

        evals     = np.concatenate(all_evals)
        sz_labels = np.concatenate(all_sz)
        order     = np.argsort(evals)
        return {
            "eigenvalues": evals[order],
            "sz_labels":   sz_labels[order],
            "conserved":   True,
            "n_sectors":   len(sectors),
        }

    else:
        dim = 2 ** n
        if dim > 4096 and verbose:
            print(f"[WARN] n={n} qubits, dim={dim}. Full diagonalization needs "
                  f"~{dim**2 * 16 / 1e9:.1f} GB RAM and may be very slow.")

        H_dense = H.toarray()
        if not np.allclose(H_dense, H_dense.conj().T, atol=1e-8):
            raise ValueError("H is not Hermitian")

        evals = np.linalg.eigh(H_dense)[0]
        return {
            "eigenvalues": evals,
            "sz_labels":   None,
            "conserved":   False,
            "n_sectors":   None,
        }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("Sz conservation test — Mironowicz_rand_EE")
    print("=" * 60)

    base = {
        "nqubits_S": 1,
        "nqubits_E": 5,
        "H_SE_Special": "Mironowicz_rand_EE",
        "H_SE_J": 0.1,
        "Mironowicz_H_E": "Z",
        "Mironowicz_alpha2": 0.05,
        "H_EE_bonds": "1DNN",
        "H_EE": [[1.0, "Z", "Z"]],
        "H_EE_J": 0.1,
        "H_EE_p": "Const",
        "seed": 42,
    }

    for theta_label, theta in [("0 (generic)", 0.0),
                                ("pi/4",        np.pi / 4),
                                ("pi/2 (MBL)",  np.pi / 2)]:
        params = {**base, "Mironowicz_theta": theta}
        conserved, reason = check_sz_conservation(params)
        print(f"  theta={theta_label:20s}  conserved={conserved}  ({reason})")

    print()
    print("=" * 60)
    print("Block eigensolver — theta=pi/2 (Sz conserved), n=6 qubits")
    print("=" * 60)

    params_mbl = {**base, "Mironowicz_theta": np.pi / 2}
    result = block_eigensolver(params_mbl, seed=42)

    n = base["nqubits_S"] + base["nqubits_E"]
    print(f"\nTotal eigenvalues: {len(result['eigenvalues'])}")
    print(f"Expected sector sizes: {[math.comb(n, k) for k in range(n+1)]}")
    print(f"Ground state energy: {result['eigenvalues'][0]:.8f}  "
          f"Sz={result['sz_labels'][0]:+.1f}")

    print()
    print("=" * 60)
    print("Fallback test — theta=pi/4 (Sz NOT conserved), n=6 qubits")
    print("=" * 60)

    params_generic = {**base, "Mironowicz_theta": np.pi / 4}
    result2 = block_eigensolver(params_generic, seed=42)
    print(f"Ground state energy: {result2['eigenvalues'][0]:.8f}")
    print(f"Sz labels: {result2['sz_labels']}")

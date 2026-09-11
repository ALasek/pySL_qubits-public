#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:25:48 2025

@author: ALasek
"""

import numpy as np
import random
import hashlib
import copy
import cupy as cp
from scipy.linalg import expm
import cupyx.scipy.sparse as cpsp
from src.Qutils import build_state_ud, build_state_mm, build_state_bias, build_state_rand, partial_trace_cp, vn_entropy_cp, von_neumann_entropy_direct, mutual_information_direct, mutual_information, apply_two_qubit_gate, apply_two_qubit_gate_cp, apply_one_qubit_gate_cp, kron_n, embed_two_qubit_op, swap_gate

_PAPER_B_TWO_MAGNITUDE_LEVELS = (0.5, np.sqrt(1.75))


def _derive_seed(base_seed, label):
    payload = f"{int(base_seed)}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _balanced_signs(count, rng):
    signs = [1] * (count // 2) + [-1] * (count // 2)
    if count % 2:
        signs.append(1 if rng.random() < 0.5 else -1)
    rng.shuffle(signs)
    return signs


def decompose_to_pauli_basis(H,dtype=cp.complex64):
    # Define Pauli matrices
    I = cp.array([[1, 0], [0, 1]],dtype=dtype)
    X = cp.array([[0, 1], [1, 0]],dtype=dtype)
    Y = cp.array([[0, -1j], [1j, 0]],dtype=dtype)
    Z = cp.array([[1, 0], [0, -1]],dtype=dtype)

    paulis = [I, X, Y, Z]
    labels = ['I', 'X', 'Y', 'Z']

    # Check input shape
    if H.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4.")

    # Decompose
    decomposition = []
    for i, A in enumerate(paulis):
        for j, B in enumerate(paulis):
            P = cp.kron(A, B)
            coeff = 0.25 * np.trace(P.conj().T @ H)
            if np.abs(coeff) > 1e-10:  # Filter out near-zero terms
                decomposition.append((labels[i], labels[j], coeff.real))

    return decomposition


class hamiltonian:

    # ------------------------------------------------------------------ #
    # Registry: maps H_SE_Special value -> builder method name            #
    # ------------------------------------------------------------------ #
    _BUILDERS = {
        0:                              "_build_generic",
        "Mironowicz":                   "_build_mironowicz",
        "Mironowicz_rand":              "_build_mironowicz_rand",
        "Mironowicz_rand_EE":           "_build_mironowicz_rand_EE",
        "Mironowicz_rand_EE_transverse": "_build_mironowicz_rand_EE_transverse",
        "Mironowicz_rand_EE_ZX":        "_build_mironowicz_rand_EE_ZX",
        "Mironowicz_rand_EE_ZZ_to_ZX":  "_build_mironowicz_rand_EE_ZZ_to_ZX",
        "PaperB_staged":                "_build_paper_b_staged",
        "Mironowicz_rand_epsilon":      "_build_mironowicz_rand_epsilon",
        "Mironowicz_rand_mod_epsilon":  "_build_mironowicz_rand_mod_epsilon",
        "ZZ_rand":                      "_build_zz_rand",
    }

    # ------------------------------------------------------------------ #
    # Constructor                                                          #
    # ------------------------------------------------------------------ #
    def __init__(self, params, seed):
        self.dtypepbits = params["dtypepbits"]
        if self.dtypepbits == 32:
            self.dtypef = cp.float32
            self.dtypec = cp.complex64
        elif self.dtypepbits == 64:
            self.dtypef = cp.float64
            self.dtypec = cp.complex128

        self.T = 0

        self.nqubitsS = params["nqubits_S"]
        self.nqubitsE = params["nqubits_E"]
        self.nqubits  = self.nqubitsS + self.nqubitsE

        dim_SE = 2 ** (self.nqubitsS + 1)
        self.dim = 2 ** self.nqubits

        sigmadict = {
            'I': cp.array([[1, 0], [0,  1]], dtype=self.dtypec),
            'X': cp.array([[0, 1], [1,  0]], dtype=self.dtypec),
            'Y': cp.array([[0,-1j],[1j, 0]], dtype=self.dtypec),
            'Z': cp.array([[1, 0], [0, -1]], dtype=self.dtypec),
        }

        try:
            self.H_noise    = params["H_noise"]
            self.p_noise_Gate = params["p_noise_Gate"]
        except KeyError:
            self.H_noise    = []
            self.p_noise_Gate = []

        try:
            self.noiseT   = params["noiseT"]
            self.noiseStd = params["noiseStd"]
        except KeyError:
            self.noiseT   = 0
            self.noiseStd = 0

        self.noiseOn = 0
        self.H_noise_Matrix = cp.zeros([2, 2], dtype=self.dtypec)

        for e in self.H_noise:
            self.noiseOn += e[0]
            self.H_noise_Matrix += e[0] * sigmadict[e[1]]

        self.seed = int(seed)
        self._rng_cache = {}

        self.bondsSE = []
        self.bondsEE = []
        self.J_SE    = []
        self.J_EE    = []
        self.J_E     = []

        self.H_SE_matrix = cp.zeros([dim_SE, dim_SE], dtype=self.dtypec)
        self.H_EE_matrix = cp.zeros([4, 4],           dtype=self.dtypec)

        H_SE_Special = params["H_SE_Special"]
        builder_name = self._BUILDERS.get(H_SE_Special)
        if builder_name is None:
            raise ValueError("Unknown special model: " + str(H_SE_Special))
        getattr(self, builder_name)(params, sigmadict)

    # ------------------------------------------------------------------ #
    # Builder helpers                                                      #
    # ------------------------------------------------------------------ #

    def _rng(self, label):
        if label not in self._rng_cache:
            self._rng_cache[label] = random.Random(_derive_seed(self.seed, label))
        return self._rng_cache[label]

    def _mironowicz_SE_matrix(self, theta, params, sigmadict):
        """Fill H_SE_matrix with the standard Mironowicz block."""
        self.H_SE_matrix[2, 2] =  (np.pi / 2) * (1 - np.sin(theta))
        self.H_SE_matrix[3, 3] =  (np.pi / 2) * (1 + np.sin(theta))
        self.H_SE_matrix[2, 3] = -(np.pi / 2) * np.cos(theta)
        self.H_SE_matrix[3, 2] = -(np.pi / 2) * np.cos(theta)
        self.H_E_matrix = sigmadict[params["Mironowicz_H_E"]]

    @staticmethod
    def _couplings(count, distribution, scale, rng):
        if distribution == "Norm":
            return [rng.gauss(0, scale) for _ in range(count)]
        if distribution == "Const":
            return [scale] * count
        raise ValueError(f"unknown coupling distribution: {distribution!r}")

    def _set_EE_bonds_and_couplings(self, params):
        bonds = params["H_EE_bonds"]
        if bonds == "1DNN":
            self.bondsEE = [
                [ne + self.nqubitsS, ne + self.nqubitsS + 1]
                for ne in range(self.nqubitsE - 1)
            ]
        else:
            self.bondsEE = [list(bond) for bond in bonds]
        self.J_EE = self._couplings(
            len(self.bondsEE),
            params["H_EE_p"],
            params["H_EE_J"],
            self._rng("EE.J_EE"),
        )

    def _add_EE_interactions(self, params, sigmadict):
        """Append EE bonds/couplings (shared logic for generic & rand_EE)."""
        H_EE       = params['H_EE']
        self._set_EE_bonds_and_couplings(params)

        for sigmas in H_EE:
            self.H_EE_matrix += sigmas[0] * cp.kron(sigmadict[sigmas[1]], sigmadict[sigmas[2]])

    def _add_EE_transverse_interactions(self, params, sigmadict):
        self._set_EE_bonds_and_couplings(params)

        z = sigmadict['Z']
        self.H_EE_matrix = 0.5 * (
            cp.kron(z, sigmadict['I'])
            + cp.kron(z, sigmadict['X'])
            + cp.kron(z, sigmadict['Y'])
        )

    def _add_EE_ZX_interactions(self, params, sigmadict):
        self._set_EE_bonds_and_couplings(params)

        self.H_EE_matrix = cp.kron(sigmadict['Z'], sigmadict['X'])

    def _add_EE_ZZ_to_ZX_interactions(self, params, sigmadict):
        epsilon    = params['Mironowicz_ZZ_to_ZX_epsilon']
        self._set_EE_bonds_and_couplings(params)

        rotated_second_qubit = (
            np.cos(epsilon) * sigmadict['Z']
            + np.sin(epsilon) * sigmadict['X']
        )
        self.H_EE_matrix = cp.kron(sigmadict['Z'], rotated_second_qubit)

    def component_view(self, *, include_SE=True, include_E=True, include_EE=True):
        view = copy.copy(self)
        view.J_SE = list(self.J_SE) if include_SE else [0.0] * len(self.J_SE)
        view.J_E = list(self.J_E) if include_E else [0.0] * len(self.J_E)
        view.J_EE = list(self.J_EE) if include_EE else [0.0] * len(self.J_EE)
        return view

    # ------------------------------------------------------------------ #
    # Model builders                                                       #
    # ------------------------------------------------------------------ #

    def _build_generic(self, params, sigmadict):
        H_SE_bonds = params['H_SE_bonds']
        H_SE       = params['H_SE']
        H_SE_J     = params['H_SE_J']
        H_SE_p     = params['H_SE_p']
        H_E        = params['H_E']
        H_E_J      = params['H_E_J']
        H_E_p      = params.get('H_E_p', 'Norm')
        rng_se = self._rng("generic.J_SE")
        rng_e = self._rng("generic.J_E")

        if H_SE_bonds == 'S_to_all':
            for ns in range(self.nqubitsS):
                for ne in range(self.nqubitsE):
                    self.bondsSE.append([ns, ne + 1])
                    if H_SE_p == "Norm":
                        self.J_SE.append(rng_se.gauss(0, H_SE_J))
                    elif H_SE_p == "Const":
                        self.J_SE.append(H_SE_J)
        else:
            self.bondsSE = [list(bond) for bond in H_SE_bonds]
            self.J_SE = self._couplings(len(self.bondsSE), H_SE_p, H_SE_J, rng_se)

        for sigmas in H_SE:
            self.H_SE_matrix += sigmas[0] * cp.kron(sigmadict[sigmas[1]], sigmadict[sigmas[2]])

        if len(H_E) > 0:
            self.H_E_matrix = cp.zeros([2, 2], dtype=self.dtypec)
            for _ in range(self.nqubitsE):
                if H_E_p == "Norm":
                    self.J_E.append(rng_e.gauss(0, H_E_J))
                elif H_E_p == "Const":
                    self.J_E.append(H_E_J)
            for sigmas in H_E:
                self.H_E_matrix += sigmas[0] * sigmadict[sigmas[1]]

        self._add_EE_interactions(params, sigmadict)

    def _build_mironowicz(self, params, sigmadict):
        theta   = params["Mironowicz_theta"]
        H_SE_J  = params['H_SE_J']
        alpha2  = params["Mironowicz_alpha2"]

        for ns in range(self.nqubitsS):
            for ne in range(self.nqubitsE):
                self.bondsSE.append([ns, ne + 1])
                self.J_SE.append(H_SE_J)

        self._mironowicz_SE_matrix(theta, params, sigmadict)

        for _ in range(self.nqubitsE):
            self.J_E.append(alpha2)

    def _build_mironowicz_rand(self, params, sigmadict):
        theta   = params["Mironowicz_theta"]
        H_SE_J  = params['H_SE_J']
        alpha2  = params["Mironowicz_alpha2"]
        h0      = params.get("Mironowicz_h0", 0.0)
        rng_se = self._rng("mironowicz.J_SE")
        rng_e = self._rng("mironowicz.J_E")

        for ns in range(self.nqubitsS):
            for ne in range(self.nqubitsE):
                self.bondsSE.append([ns, ne + 1])
                self.J_SE.append(rng_se.gauss(0, H_SE_J))

        self._mironowicz_SE_matrix(theta, params, sigmadict)

        for _ in range(self.nqubitsE):
            self.J_E.append(h0 + rng_e.gauss(0, alpha2))

    def _build_mironowicz_rand_EE(self, params, sigmadict):
        self._build_mironowicz_rand(params, sigmadict)
        self._add_EE_interactions(params, sigmadict)

    def _build_mironowicz_rand_EE_transverse(self, params, sigmadict):
        self._build_mironowicz_rand(params, sigmadict)
        self._add_EE_transverse_interactions(params, sigmadict)

    def _build_mironowicz_rand_EE_ZX(self, params, sigmadict):
        self._build_mironowicz_rand(params, sigmadict)
        self._add_EE_ZX_interactions(params, sigmadict)

    def _build_mironowicz_rand_EE_ZZ_to_ZX(self, params, sigmadict):
        self._build_mironowicz_rand(params, sigmadict)
        self._add_EE_ZZ_to_ZX_interactions(params, sigmadict)

    def _build_paper_b_staged(self, params, sigmadict):
        theta = params["Mironowicz_theta"]
        case = params["PaperB_case"]
        field_strength = float(params["PaperB_field_strength"])
        field_normalization = params.get("PaperB_field_normalization", "nominal")

        for ns in range(self.nqubitsS):
            for ne in range(self.nqubitsE):
                self.bondsSE.append([ns, ne + self.nqubitsS])
        self.J_SE = self._couplings(
            len(self.bondsSE),
            params.get("H_SE_p", "Const"),
            params["H_SE_J"],
            self._rng("paper_b.J_SE"),
        )
        self._mironowicz_SE_matrix(theta, params, sigmadict)

        if "disjoint" in case:
            local_bonds = [(index, index + 1) for index in range(0, self.nqubitsE - 1, 2)]
        else:
            local_bonds = [(index, index + 1) for index in range(self.nqubitsE - 1)]
        self.bondsEE = [
            [left + self.nqubitsS, right + self.nqubitsS]
            for left, right in local_bonds
        ]
        self.J_EE = [params["H_EE_J"]] * len(self.bondsEE)

        z = sigmadict["Z"]
        if case.startswith("zz_"):
            self.H_EE_matrix = cp.kron(z, z)
        elif case.startswith("transverse_"):
            self.H_EE_matrix = 0.5 * (
                cp.kron(z, sigmadict["I"])
                + cp.kron(z, sigmadict["X"])
                + cp.kron(z, sigmadict["Y"])
            )
        else:
            self.H_EE_matrix = cp.kron(z, sigmadict["X"])

        if case.endswith("_sources"):
            active_sites = {left for left, _ in local_bonds}
        elif case.endswith("_targets"):
            active_sites = {right for _, right in local_bonds}
        else:
            active_sites = set(range(self.nqubitsE))

        quasiperiodic_fields = None
        if "_quasiperiodic_" in case:
            phase = 2 * np.pi * self._rng("paper_b.field_phase").random()
            beta = (np.sqrt(5) - 1) / 2
            quasiperiodic_fields = [
                field_strength * np.cos(2 * np.pi * beta * index + phase)
                for index in range(self.nqubitsE)
            ]

        if "_uniform_" in case:
            fields = [field_strength] * self.nqubitsE
        elif case == "zx_quasiperiodic_shuffled_chain":
            fields = list(quasiperiodic_fields)
            self._rng("paper_b.field_shuffle").shuffle(fields)
        elif case == "zx_quasiperiodic_signscrambled_chain":
            signs = _balanced_signs(
                self.nqubitsE,
                self._rng("paper_b.field_signs"),
            )
            fields = [
                abs(field) * sign
                for field, sign in zip(quasiperiodic_fields, signs)
            ]
        elif "_quasiperiodic_" in case:
            fields = quasiperiodic_fields
        elif "_binary_" in case:
            signs = _balanced_signs(
                self.nqubitsE,
                self._rng("paper_b.field"),
            )
            fields = [field_strength * sign for sign in signs]
        elif "_two_magnitude_" in case:
            magnitudes = (
                [_PAPER_B_TWO_MAGNITUDE_LEVELS[0] * field_strength] * (self.nqubitsE // 2)
                + [_PAPER_B_TWO_MAGNITUDE_LEVELS[1] * field_strength]
                * (self.nqubitsE - self.nqubitsE // 2)
            )
            self._rng("paper_b.field_magnitudes").shuffle(magnitudes)
            signs = _balanced_signs(
                self.nqubitsE,
                self._rng("paper_b.field_signs"),
            )
            fields = [
                magnitude * sign
                for magnitude, sign in zip(magnitudes, signs)
            ]
        else:
            rng = self._rng("paper_b.field")
            fields = [rng.gauss(0, field_strength) for _ in range(self.nqubitsE)]

        if field_normalization == "exact_rms" and field_strength > 0:
            if not active_sites:
                raise ValueError("cannot exact-RMS normalize a Paper-B field without active sites")
            active_field_rms = np.sqrt(
                np.mean([fields[index] ** 2 for index in sorted(active_sites)])
            )
            if active_field_rms == 0:
                raise ValueError("cannot exact-RMS normalize a zero Paper-B field realization")
            scale = field_strength / active_field_rms
            fields = [
                field * scale if index in active_sites else field
                for index, field in enumerate(fields)
            ]

        self.J_E = [field if index in active_sites else 0.0 for index, field in enumerate(fields)]
        self.H_E_matrix = z

    def _build_mironowicz_rand_epsilon(self, params, sigmadict):
        theta   = params["Mironowicz_theta"]
        H_SE_J  = params['H_SE_J']
        alpha2  = params["Mironowicz_alpha2"]
        h0      = params.get("Mironowicz_h0", 0.0)
        epsilon = params["Mironowicz_epsilon"]
        rng_se = self._rng("mironowicz.J_SE")
        rng_e = self._rng("mironowicz.J_E")
        x = sigmadict['X']
        z = sigmadict['Z']

        for ns in range(self.nqubitsS):
            for ne in range(self.nqubitsE):
                self.bondsSE.append([ns, ne + 1])
                self.J_SE.append(rng_se.gauss(0, H_SE_J))

        self._mironowicz_SE_matrix(theta, params, sigmadict)

        # Orthogonal epsilon correction
        self.H_SE_matrix += epsilon * (
            cp.kron(x,  np.cos(theta) * z) -
            cp.kron(x,  np.sin(theta) * x)
        )

        for _ in range(self.nqubitsE):
            self.J_E.append(h0 + rng_e.gauss(0, alpha2))

    def _build_mironowicz_rand_mod_epsilon(self, params, sigmadict):
        theta    = params["Mironowicz_theta"]
        H_SE_J   = params['H_SE_J']
        alpha2   = params["Mironowicz_alpha2"]
        h0       = params.get("Mironowicz_h0", 0.0)
        epsilon  = params["Mironowicz_epsilon"]
        epsilon2 = params["Mironowicz_epsilon2"]
        rng_se = self._rng("mironowicz.J_SE")
        rng_e = self._rng("mironowicz.J_E")
        x = sigmadict['X']
        y = sigmadict['Y']
        z = sigmadict['Z']

        for ns in range(self.nqubitsS):
            for ne in range(self.nqubitsE):
                self.bondsSE.append([ns, ne + 1])
                self.J_SE.append(rng_se.gauss(0, H_SE_J))

        self._mironowicz_SE_matrix(theta, params, sigmadict)

        self.H_SE_matrix += epsilon * (
            -cp.kron(x, np.cos(theta) * z) +
             cp.kron(x, np.sin(theta) * x)
        )
        self.H_SE_matrix += epsilon2 * cp.kron(y, y)

        for _ in range(self.nqubitsE):
            self.J_E.append(h0 + rng_e.gauss(0, alpha2))

    def _build_zz_rand(self, params, sigmadict):
        """Pure ZZ S-E coupling with Gaussian random strengths.
        H = sum_j J_j sigma_z^S sigma_z^(E_j),  J_j ~ N(0, H_SE_J^2)
        No E self-interaction, no EE interaction.
        """
        H_SE_J = params['H_SE_J']
        rng_se = self._rng("zz_rand.J_SE")
        z = sigmadict['Z']

        for ns in range(self.nqubitsS):
            for ne in range(self.nqubitsE):
                self.bondsSE.append([ns, ne + 1])
                self.J_SE.append(rng_se.gauss(0, H_SE_J))

        self.H_SE_matrix = cp.kron(z, z)

    # ------------------------------------------------------------------ #
    # Noise / gate methods                                                 #
    # ------------------------------------------------------------------ #

    def run_noiseU_Gate(self, gate_angle=0):
        self.noiseU_Gate_N = []
        if gate_angle == 0:
            gate_angle = self._rng("noise.gate_angle").gauss(0, np.pi)

        A = expm(-(1j / 2) * gate_angle * self.H_noise_Matrix.get())
        A = cp.array(A)
        for i in range(self.nqubitsE):
            self.noiseU_Gate_N.append(
                apply_one_qubit_gate_cp(A, i + self.nqubitsS, self.nqubits, self.dtypec)
            )

    def run_noiseU_Rand(self):
        "prepare 1xN raw noise Hamiltonians"
        self.noiseU_Rand_N = []
        for i in range(self.nqubitsE):
            self.noiseU_Rand_N.append(
                apply_one_qubit_gate_cp(self.H_noise_Matrix, i + self.nqubitsS, self.nqubits, self.dtypec)
            )

    def run_noise_times(self):
        "start at noiseT"
        self.noise_times        = np.zeros(self.nqubitsE)
        self.noise_sign         = np.zeros(self.nqubitsE)
        self.noise_endedQubits  = np.zeros(self.nqubitsE)
        rng = self._rng("noise.times")

        for i in range(self.nqubitsE):
            self.noise_times[i] = rng.gauss(0, self.noiseStd)
            self.noise_sign[i]  = 1 if self.noise_times[i] > 0 else -1

    def start_noise_H(self):
        "build initial noise H"
        self.H_noise = cpsp.csr_matrix((self.dim, self.dim), dtype=self.dtypec)

        for i in range(self.nqubitsE):
            self.H_noise += self.noise_sign[i] * self.noiseU_Rand_N[i]

    def update_noise_H(self, T):
        "update noise H to remove finished 1xN parts"
        for i in range(self.nqubitsE):
            if (not self.noise_endedQubits[i]) and T > self.noise_times[i]:
                self.H_noise -= self.noiseU_Rand_N[i]
                self.noise_endedQubits[i] = 1

    def decompose_H_SE(self):
        D = (decompose_to_pauli_basis(self.H_SE_matrix, dtype=self.dtypec))

        print("Interaction Hamiltonian decomposition as sigma x sigma:")

        for line in D:
            print(line)

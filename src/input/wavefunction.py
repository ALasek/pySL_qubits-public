################################################################################################################################
# Create the main wavefunction class
################################################################################################################################

import math
import hashlib
import numpy as np
import os
import shutil
import tempfile
import time

import cupy as cp
import random
import cupyx.scipy.sparse as cpsp
import scipy.sparse as spsp

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


from scipy.linalg import expm

from src.Qutils import build_state_ud,build_state_UDrand, build_state_mm, build_state_bias, build_state_rand, partial_trace_cp, vn_entropy_cp, vn_entropy_np, von_neumann_entropy_direct, mutual_information_direct, mutual_information, apply_two_qubit_gate_cp, apply_one_qubit_gate_cp, embedded_two_qubit_entries_cp, embedded_one_qubit_entries_cp, embedded_two_qubit_entries_np, embedded_one_qubit_entries_np, kron_n, embed_two_qubit_op, swap_gate,trace_distance,quantum_relative_entropy,fidelity,trace_distance_fractions,build_state_costumprod
from src.analysis.correlator_metrics import CORRELATOR_METRIC_NAMES, connected_correlator_metrics
from src.gpu_memory import cleanup_cupy_memory, gpu_memory_tracker
from src.input.validation import derive_time_grid
from src.sparse_operator import HybridSparseOperator, SumSparseOperator


def draw_bloch_sphere(ax):
    # Transparent sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.07, linewidth=0)

    # Equator and two meridians
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color='k', linewidth=0.8, alpha=0.4)
    for angle in [0, np.pi/2]:
        ax.plot(np.cos(u)*np.cos(angle), np.cos(u)*np.sin(angle), np.sin(u),
                color='k', linewidth=0.5, alpha=0.3)

    # Label the six basis states
    # Basis labels  now 100 % safe characters
    ax.text(0,   0,  1.2, r"$|0\rangle$",     color="blue",    fontsize=16, ha='center')
    ax.text(0,   0, -1.3, r"$|1\rangle$",     color="red",     fontsize=16, ha='center')
    ax.text(1.3, 0,   0, r"$|+\rangle$",     color="green",   fontsize=14)
    ax.text(-1.4,0,   0, r"$|-\rangle$",     color="green",   fontsize=14)   #  fixed
    ax.text(0,  1.3,  0, r"$|i\rangle$",     color="purple",  fontsize=14)
    ax.text(0, -1.4,  0, r"$|-i\rangle$",    color="purple",  fontsize=14)   #  fixed

# ------------------------------------------------------------------
# Function that returns the Bloch vector (x,y,z) from a state vector
# ------------------------------------------------------------------
def state_to_bloch(psi):
    # psi is a 2-element complex array (normalized)
    a, b = psi[0], psi[1]
    x = 2 * np.real(a.conj() * b)
    y = 2 * np.imag(b.conj() * a)   # correct sign convention
    z = np.abs(a)**2 - np.abs(b)**2
    return np.array([x, y, z])

def multiple_unique_sorted_lists(a, b, N, M, rng=None):
    if N > (b - a + 1):
        raise ValueError("N is larger than the number of available unique integers")
    
    unique_sets = set()
    results = []
    
    sampler = rng or random
    while len(results) < M:
        sample = sampler.sample(range(a, b + 1), N)
        sorted_tuple = tuple(sorted(sample))  # immutable & hashable
        if sorted_tuple not in unique_sets:
            unique_sets.add(sorted_tuple)
            results.append(list(sorted_tuple))
    
    return results



def clamp_down(X,a):
    return 0 if X < a else X

def _include_coupling(J):
    return J != 0


def _derive_fragment_seed(seed, fragment_size):
    payload = f"{int(seed)}:fragments:{int(fragment_size)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _half_fragment_sizes(nqubits_environment):
    return range(1, (int(nqubits_environment) + 3) // 2)

class wavefunction:
    
    
    
    #this is called on creation of instance
    def __init__(self,  params,runN):
        
        self.params=params
        
        self.AverageOverRunsN=int(round(float(params["AverageOverRunsN"])))
        self.runN=runN
        
        self.QREmaxFragSize=params["QREmaxFragSize"]
        self.fragment_sample_count = params.get("fragment_sample_count")
        self.fragment_reuse_samples = params.get("fragment_reuse_samples", False)
        self.store_fragment_samples = params.get("store_fragment_samples", False)
        self.fragment_quantile_levels = np.asarray(params.get("fragment_quantiles", []), dtype=float)
        
        self.clampISE=params["clampISE"]
        self.dtypepbits=params["dtypepbits"]
        
        if self.dtypepbits==32:
            self.dtypef=cp.float32
            self.dtypec=cp.complex64
        elif self.dtypepbits==64:
            self.dtypef=cp.float64
            self.dtypec=cp.complex128
            
        self.nqubitsS=params["nqubits_S"]
        self.nqubitsE=params["nqubits_E"]
        self.nqubits=self.nqubitsS+self.nqubitsE
        self.psi_S_spec=params["psi_S_spec"]
        self.psi_E_spec=params["psi_E_spec"]
        self.psi_bias=params["psi_bias"]
        self.T=params["T"]
        self.dT=params["dT"]
        self.printT=params["printT"]
        self.time_grid = derive_time_grid(params)
        self.Tnow=0
        self.printTnow=0

        initial_seed = params.get("run_seeds", [params["seed"]])[0]
        self.setSeed(initial_seed)
        
        self.Tsteps= self.time_grid["total_steps"]
        self.TstepsPrint= self.time_grid["print_sample_count"] - 1
        self.dim=2**self.nqubits
        self.dimS=2**self.nqubitsS
        self.dimE=2**self.nqubitsE
        
        self._reset_run_state()

        Tlen = self.time_grid["print_sample_count"]
        self.store_psi = params.get("store_psi", False)
        self.store_psi_on_disk = self.store_psi and params.get("store_psi_on_disk", False)
        self.psi_snapshot_dtype = np.float32 if self.dtypepbits == 32 else np.float64
        self.psi_snapshot_temp_dir = tempfile.mkdtemp(prefix="pysl_psi_snapshots_") if self.store_psi else None
        self.psi_snapshot_records = []
        self.psi_snapshot_written = None
        self.psi_store_path = None
        self.psi_stored = None
        if self.store_psi:
            self._prepare_psi_snapshot_storage(runN)
        self.store_correlators = params.get("store_correlators", True)
        self.correlator_axis = str(params.get("correlator_axis", "Z")).upper()
        raw_sources = params.get("correlator_sources", "all")
        self.correlator_sources = (
            list(range(self.nqubitsE))
            if raw_sources == "all"
            else [int(value) for value in raw_sources]
        )
        self.correlator_reference_time = float(params.get("correlator_reference_time", 0))
        self.correlators_active = False
        self.compute_discord = params.get("compute_discord", True)
        self.profile_evolution = params.get("profile_evolution", False)
        self.profile_evolution_max_steps = int(params.get("profile_evolution_max_steps", 200))
        self._profile_current_evolve_step = False
        if self.store_correlators:
            n_sources = len(self.correlator_sources)
            self.C_correlators = np.full([Tlen, self.nqubitsE, n_sources], np.nan, dtype=np.complex64)
            self.C_connected_correlators = np.full(
                [Tlen, self.nqubitsE, n_sources], np.nan, dtype=np.complex64
            )
            # C_SE_correlators[t, s, j] = <psi(t)| sigma_z^s |phi_j(t)>  (s = system qubit index)
            self.C_SE_correlators = np.full([Tlen, self.nqubitsS, n_sources], np.nan, dtype=np.complex64)
            self.E_z_initial = None
        else:
            self.C_correlators    = None
            self.C_connected_correlators = None
            self.C_SE_correlators = None
            self.E_z_initial = None

        self._perf = {}

        if runN==0:
            self.logInit()
        
    def setSeed(self,pseed):
        if pseed=="TIME":
            self.seed=time.time_ns()
        else:
            self.seed=int(pseed)
            
    def incRunN(self):
        self.runN=self.runN+1

    def _reset_run_state(self):
        t_len = self.time_grid["print_sample_count"]
        self.Tnow = 0
        self.printTnow = 0
        self.norms = []
        self.modified_norms = []
        self.rhoS_T = []
        self.rhoS_purity = []
        self.rhoS_BlochVecs = []
        self.rhoS_BlochEvals = []
        self.SBS_qre = [None] + [np.full([self.nqubitsE - f + 1, t_len], np.nan) for f in range(1, self.QREmaxFragSize + 1)]
        self.SBS_trace_dist = [None] + [np.full([self.nqubitsE - f + 1, t_len], np.nan) for f in range(1, self.QREmaxFragSize + 1)]
        self.SBS_fid = [None] + [np.full([self.nqubitsE - f + 1, t_len], np.nan) for f in range(1, self.QREmaxFragSize + 1)]
        self._fragment_samples = {}

    def _sample_times(self):
        return np.arange(self.time_grid["print_sample_count"]) * self.time_grid["print_interval_steps"] * self.dT

    def _psi_snapshot_shape(self):
        return (self.time_grid["print_sample_count"], 2, self.dim)

    def _prepare_psi_snapshot_storage(self, runN):
        if not self.store_psi:
            self.psi_stored = None
            return
        shape = self._psi_snapshot_shape()
        self.psi_snapshot_written = np.zeros(shape[0], dtype=bool)
        if self.store_psi_on_disk:
            if self.psi_stored is not None:
                self.psi_stored.flush()
                del self.psi_stored
                self.psi_stored = None
            self.psi_store_path = os.path.join(
                self.psi_snapshot_temp_dir,
                f"run_{int(runN):05d}_psi_snapshots.npy",
            )
            self.psi_stored = np.lib.format.open_memmap(
                self.psi_store_path,
                mode="w+",
                dtype=self.psi_snapshot_dtype,
                shape=shape,
            )
        else:
            self.psi_store_path = None
            if self.psi_stored is None or self.psi_stored.shape != shape:
                self.psi_stored = np.zeros(shape, dtype=self.psi_snapshot_dtype)
            else:
                self.psi_stored.fill(0)

    def store_psi_snapshot(self, t_idx):
        if self.psi_stored is None:
            return
        if t_idx < 0 or t_idx >= self.psi_stored.shape[0]:
            raise IndexError(f"psi snapshot index {t_idx} is out of bounds")
        psi_phys = self.psi if getattr(self, "psi", None) is not None else self.psi_real + 1j * self.psi_imag
        self.psi_stored[t_idx, 0] = cp.real(psi_phys).get()
        self.psi_stored[t_idx, 1] = cp.imag(psi_phys).get()
        self.psi_snapshot_written[t_idx] = True
        if self.store_psi_on_disk:
            self.psi_stored.flush()

    def persist_psi_snapshots_current_run(self, run_seed=None):
        if self.psi_stored is None:
            return None
        written = np.flatnonzero(self.psi_snapshot_written)
        if written.size == 0:
            return None

        run_index = int(self.runN)
        if self.store_psi_on_disk:
            self.psi_stored.flush()
            path = self.psi_store_path
            del self.psi_stored
            self.psi_stored = None
            self.psi_store_path = None
        else:
            path = os.path.join(self.psi_snapshot_temp_dir, f"run_{run_index:05d}_psi_snapshots.npy")
            np.save(path, self.psi_stored)

        record = {
            "run_index": run_index,
            "seed": None if run_seed is None else int(run_seed),
            "path": path,
            "filename": os.path.basename(path),
            "shape": list(self._psi_snapshot_shape()),
            "dtype": str(np.dtype(self.psi_snapshot_dtype)),
            "format": "npy",
            "representation": "physical_psi_real_imag",
            "written_indices": written.astype(int).tolist(),
            "sample_times": self._sample_times().astype(float).tolist(),
        }
        self.psi_snapshot_records = [
            existing for existing in self.psi_snapshot_records
            if existing.get("run_index") != run_index
        ]
        self.psi_snapshot_records.append(record)
        return record

    def start_run(self, runN, seed):
        self.runN = runN
        self.setSeed(seed)
        self._reset_run_state()
        if self.store_correlators:
            self.C_correlators.fill(np.nan)
            self.C_connected_correlators.fill(np.nan)
            self.C_SE_correlators.fill(np.nan)
            self.E_z_initial = None
            self.correlators_active = False
        if self.store_psi:
            self._prepare_psi_snapshot_storage(runN)
        
    def buildPsi(self):
        
        random.seed(int(self.seed))
        
        up = cp.array([1, 0], dtype=self.dtypec)
        down = cp.array([0, 1], dtype=self.dtypec)
        
        if isinstance(self.psi_S_spec, str):
            match self.psi_S_spec:
               case "x+":
                   psi_S= (up + down) / cp.sqrt(2)
               case "x-":
                   psi_S = (up - down) / cp.sqrt(2)
               case "y+":
                   psi_S= (up + 1j*down) / cp.sqrt(2)
               case "y-":
                   psi_S = (up - 1j*down) / cp.sqrt(2)
               case "z+":
                   psi_S= up
               case "z-":
                   psi_S= down
        else:
            psi_S=self.psi_S_spec[0]*up + self.psi_S_spec[1]*down
            psi_S=psi_S/cp.linalg.norm(psi_S)
               
            
        psi_S=psi_S.astype(self.dtypec)
        
        #psi_S=self.build_state_Sspec()
               
        match self.psi_E_spec:
            case "mm":
                psi_E=build_state_mm(self)
            case "ud":
                psi_E=build_state_ud(self)
            case "bias":
                psi_E=build_state_bias(self)
            case "rand":
                psi_E=build_state_rand(self)
            case "UDrand":
                psi_E=build_state_UDrand(self)
            case _:
                psi_E=build_state_costumprod(self)
            
        self.psi=cp.kron(psi_S,psi_E)
        
        self.psi/=cp.linalg.norm(self.psi)
        
        # The complex work-vector extension keeps the same explicit leapfrog
        # algebra for real and complex Hermitian Hamiltonians.
        self.psi_real = cp.real(self.psi).astype(self.dtypec)
        self.psi_imag = cp.imag(self.psi).astype(self.dtypec)
        del self.psi  # free full complex state; use psiparts() to reconstruct when needed
        
        
        down_row = cpsp.csr_matrix(down.reshape(1, 2))
        up_row = cpsp.csr_matrix(up.reshape(1, 2))
        eye_E = cpsp.identity(self.dimE, format='csr', dtype=self.dtypec)
        self.pointerProj0 = cpsp.kron(down_row, eye_E, format='csr').tocsr()
        self.pointerProj1 = cpsp.kron(up_row, eye_E, format='csr').tocsr()
        
        self.psi_E_init=psi_E
        
    def logInit(self):
        t_len = self.time_grid["print_sample_count"]
        if self.store_fragment_samples:
            self.fragment_sample_members = np.full(
                [
                    self.nqubits,
                    self.AverageOverRunsN,
                    int(self.fragment_sample_count),
                    self.nqubitsE,
                ],
                -1,
                dtype=np.int16,
            )
        self.SBS_fid_runs = [None] + [
            np.full([self.nqubitsE - f + 1, t_len, self.AverageOverRunsN], np.nan)
            for f in range(1, self.QREmaxFragSize + 1)
        ]
        self._init_fraction_metric("I_S_Ef", t_len)
        if self.compute_discord:
            self._init_fraction_metric("Holevo_Z_S_Ef", t_len)
            self._init_fraction_metric("Discord_Z_S_Ef", t_len)
        self.S_vn=np.zeros([t_len,self.AverageOverRunsN])
        self.S_vn_runAv=np.zeros([t_len])
        self.S_vn_runSTD=np.zeros([t_len])
        self.S_vn_runSEM=np.zeros([t_len])
        self.branch_probabilities = np.full([2, self.AverageOverRunsN], np.nan)
        self.branch_energy_density = np.full([2, self.AverageOverRunsN], np.nan)
        self.branch_energy_density_std = np.full([2, self.AverageOverRunsN], np.nan)
        self.branch_energy_density_gap = np.full([self.AverageOverRunsN], np.nan)
        
        self.TraceDist_fractionsT= np.zeros([self.nqubits,t_len,self.AverageOverRunsN])
        self.TraceDist_fractionsT_STD= np.zeros([self.nqubits,t_len,self.AverageOverRunsN])
        self.TraceDist_fractionsT_runAv= np.zeros([self.nqubits,t_len])
        self.TraceDist_fractionsT_STD_runAv= np.zeros([self.nqubits,t_len])
        self.TraceDist_fractionsT_runSTD= np.zeros([self.nqubits,t_len])
        self.TraceDist_fractionsT_runSEM= np.zeros([self.nqubits,t_len])
        self._run_metric_sums = {}
        self._run_metric_counts = {}
        self._metric_runs_finalized = 0
        if self.store_correlators:
            for name in CORRELATOR_METRIC_NAMES:
                setattr(self, f"{name}_T", np.full([t_len, self.AverageOverRunsN], np.nan))
                setattr(self, f"{name}_T_runAv", np.full([t_len], np.nan))
                setattr(self, f"{name}_T_runSTD", np.full([t_len], np.nan))
                setattr(self, f"{name}_T_runSEM", np.full([t_len], np.nan))

    def _init_fraction_metric(self, prefix, t_len):
        setattr(self, f"{prefix}_fractionsT", np.zeros([self.nqubits, t_len, self.AverageOverRunsN]))
        setattr(self, f"{prefix}_fractionsT_STD", np.zeros([self.nqubits, t_len, self.AverageOverRunsN]))
        setattr(
            self,
            f"{prefix}_fractionsT_Nsamples",
            np.ones([self.nqubits, t_len, self.AverageOverRunsN], dtype=np.int32),
        )
        setattr(self, f"{prefix}_fractionsT_runAv", np.zeros([self.nqubits, t_len]))
        setattr(self, f"{prefix}_fractionsT_STD_runAv", np.zeros([self.nqubits, t_len]))
        setattr(self, f"{prefix}_fractionsT_runSTD", np.zeros([self.nqubits, t_len]))
        setattr(self, f"{prefix}_fractionsT_runSEM", np.zeros([self.nqubits, t_len]))
        setattr(self, f"{prefix}_fractionsT_fragmentSTD_runAv", np.zeros([self.nqubits, t_len]))
        setattr(self, f"{prefix}_fractionsT_fragmentSTD_prop", np.zeros([self.nqubits, t_len]))
        setattr(self, f"{prefix}_fractionsT_fragmentSEM_runAv", np.zeros([self.nqubits, t_len]))
        if self.fragment_quantile_levels.size:
            shape = (self.fragment_quantile_levels.size, self.nqubits, t_len, self.AverageOverRunsN)
            setattr(self, f"{prefix}_fractionsT_quantiles", np.full(shape, np.nan))
        if self.store_fragment_samples:
            shape = (self.nqubits, t_len, self.AverageOverRunsN, int(self.fragment_sample_count))
            setattr(self, f"{prefix}_fractionsT_samples", np.full(shape, np.nan))

    def _fragment_combinations(self, fragment_size, maxcombs):
        count = min(maxcombs, math.comb(self.nqubitsE, fragment_size))
        if self.fragment_sample_count is not None:
            count = min(count, int(self.fragment_sample_count))
        if not self.fragment_reuse_samples:
            combinations = multiple_unique_sorted_lists(1, self.nqubitsE, fragment_size, count)
        else:
            if fragment_size not in self._fragment_samples:
                rng = random.Random(_derive_fragment_seed(self.seed, fragment_size))
                self._fragment_samples[fragment_size] = multiple_unique_sorted_lists(
                    1,
                    self.nqubitsE,
                    fragment_size,
                    count,
                    rng=rng,
                )
            combinations = self._fragment_samples[fragment_size]
        if getattr(self, "store_fragment_samples", False):
            for sample_index, combination in enumerate(combinations):
                self.fragment_sample_members[
                    fragment_size,
                    self.runN,
                    sample_index,
                    :fragment_size,
                ] = combination
        return combinations
        
    def doRunAvs(self):
        run_count = self.AverageOverRunsN
        prefixes = ["I_S_Ef"]
        if getattr(self, "compute_discord", True):
            prefixes.extend(["Holevo_Z_S_Ef", "Discord_Z_S_Ef"])
        for prefix in prefixes:
            if hasattr(self, f"{prefix}_fractionsT"):
                self._finalize_fraction_metric(prefix, run_count)
        self.S_vn_runAv=np.mean(self.S_vn, axis=1)
        if run_count > 1:
            self.S_vn_runSTD=np.std(self.S_vn, axis=1, ddof=1)
        else:
            self.S_vn_runSTD=np.zeros_like(self.S_vn_runAv)
        self.S_vn_runSEM=self.S_vn_runSTD / np.sqrt(run_count)
        if hasattr(self, "branch_energy_density_gap") and np.any(np.isfinite(self.branch_energy_density_gap)):
            self.branch_probabilities_runAv = np.nanmean(self.branch_probabilities, axis=1)
            self.branch_energy_density_runAv = np.nanmean(self.branch_energy_density, axis=1)
            self.branch_energy_density_std_runAv = np.nanmean(self.branch_energy_density_std, axis=1)
            self.branch_energy_density_gap_runAv = float(np.nanmean(self.branch_energy_density_gap))
            if run_count > 1:
                self.branch_energy_density_gap_runSTD = float(
                    np.nanstd(self.branch_energy_density_gap, ddof=1)
                )
            else:
                self.branch_energy_density_gap_runSTD = 0.0
            self.branch_energy_density_gap_runSEM = (
                self.branch_energy_density_gap_runSTD / np.sqrt(run_count)
            )
        
        self.TraceDist_fractionsT_runAv = np.mean(self.TraceDist_fractionsT, axis=2)
        if run_count > 1:
            self.TraceDist_fractionsT_runSTD = np.std(self.TraceDist_fractionsT, axis=2, ddof=1)
        else:
            self.TraceDist_fractionsT_runSTD = np.zeros_like(self.TraceDist_fractionsT_runAv)
        self.TraceDist_fractionsT_runSEM = self.TraceDist_fractionsT_runSTD / np.sqrt(run_count)
        self.TraceDist_fractionsT_STD_runAv = self.TraceDist_fractionsT_runSEM
        if self.store_correlators:
            for name in CORRELATOR_METRIC_NAMES:
                data = getattr(self, f"{name}_T")
                avg, std, sem = self._nan_run_average(data)
                setattr(self, f"{name}_T_runAv", avg)
                setattr(self, f"{name}_T_runSTD", std)
                setattr(self, f"{name}_T_runSEM", sem)
        self._finalize_run_metrics()

    def _finalize_fraction_metric(self, prefix, run_count):
        values = getattr(self, f"{prefix}_fractionsT")
        std_values = getattr(self, f"{prefix}_fractionsT_STD")
        nsamples = getattr(self, f"{prefix}_fractionsT_Nsamples")

        run_av = np.mean(values, axis=2)
        if run_count > 1:
            run_std = np.std(values, axis=2, ddof=1)
        else:
            run_std = np.zeros_like(run_av)
        run_sem = run_std / np.sqrt(run_count)
        fragment_std_run_av = np.mean(std_values, axis=2)
        fragment_std_prop = np.sqrt(np.sum(std_values ** 2, axis=2)) / run_count
        samples = np.maximum(nsamples, 1)
        fragment_sem_sq = (std_values ** 2) / samples
        fragment_sem_run_av = np.sqrt(np.sum(fragment_sem_sq, axis=2)) / run_count
        std_run_av = np.sqrt(run_sem ** 2 + fragment_std_prop ** 2)

        setattr(self, f"{prefix}_fractionsT_runAv", run_av)
        setattr(self, f"{prefix}_fractionsT_runSTD", run_std)
        setattr(self, f"{prefix}_fractionsT_runSEM", run_sem)
        setattr(self, f"{prefix}_fractionsT_fragmentSTD_runAv", fragment_std_run_av)
        setattr(self, f"{prefix}_fractionsT_fragmentSTD_prop", fragment_std_prop)
        setattr(self, f"{prefix}_fractionsT_fragmentSEM_runAv", fragment_sem_run_av)
        setattr(self, f"{prefix}_fractionsT_STD_runAv", std_run_av)
        quantiles = getattr(self, f"{prefix}_fractionsT_quantiles", None)
        if quantiles is not None:
            valid = np.isfinite(quantiles)
            count = np.sum(valid, axis=-1)
            total = np.sum(np.where(valid, quantiles, 0.0), axis=-1)
            quantile_run_av = np.full(count.shape, np.nan)
            np.divide(total, count, out=quantile_run_av, where=count > 0)
            setattr(self, f"{prefix}_fractionsT_quantiles_runAv", quantile_run_av)

    def _add_run_metric(self, name, values):
        arr = np.asarray(values)
        valid = np.isfinite(arr.real) & np.isfinite(arr.imag) if np.iscomplexobj(arr) else np.isfinite(arr)
        dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
        if name not in self._run_metric_sums:
            self._run_metric_sums[name] = np.zeros(arr.shape, dtype=dtype)
            self._run_metric_counts[name] = np.zeros(arr.shape, dtype=np.int32)
        self._run_metric_sums[name][valid] += arr[valid]
        self._run_metric_counts[name][valid] += 1

    def _metric_average(self, name):
        total = self._run_metric_sums[name]
        count = self._run_metric_counts[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = total / count
        avg[count == 0] = np.nan
        return avg

    def _nan_run_average(self, data):
        arr = np.asarray(data, dtype=np.float64)
        valid = np.isfinite(arr)
        count = np.sum(valid, axis=1)
        total = np.sum(np.where(valid, arr, 0.0), axis=1)
        avg = np.full(arr.shape[0], np.nan)
        avg[count > 0] = total[count > 0] / count[count > 0]

        std = np.full(arr.shape[0], np.nan)
        for i, n in enumerate(count):
            if n > 1:
                std[i] = np.std(arr[i, valid[i]], ddof=1)
            elif n == 1:
                std[i] = 0.0

        sem = np.full(arr.shape[0], np.nan)
        sem[count > 0] = std[count > 0] / np.sqrt(count[count > 0])
        return avg, std, sem

    def finish_run_metrics(self):
        self._metric_runs_finalized += 1
        self._add_run_metric("rhoS_T", np.asarray(self.rhoS_T))
        self._add_run_metric("rhoS_purity", np.asarray(self.rhoS_purity))

        for attr in ("SBS_qre", "SBS_trace_dist", "SBS_fid"):
            values = getattr(self, attr)
            for frag_size in range(1, self.QREmaxFragSize + 1):
                if attr == "SBS_fid":
                    self.SBS_fid_runs[frag_size][:, :, self.runN] = values[frag_size]
                self._add_run_metric(f"{attr}_{frag_size}", values[frag_size])

        if self.store_correlators:
            for name, series in connected_correlator_metrics(
                self.C_connected_correlators,
                source_indices=getattr(
                    self,
                    "correlator_sources",
                    list(range(self.C_connected_correlators.shape[1])),
                ),
            ).items():
                getattr(self, f"{name}_T")[:, self.runN] = series
            self._add_run_metric("C_correlators", self.C_correlators)
            self._add_run_metric("C_connected_correlators", self.C_connected_correlators)
            self._add_run_metric("C_SE_correlators", self.C_SE_correlators)
            self._add_run_metric("C_correlators_abs", np.abs(self.C_correlators))
            self._add_run_metric("C_connected_correlators_abs", np.abs(self.C_connected_correlators))
            self._add_run_metric("C_SE_correlators_abs", np.abs(self.C_SE_correlators))

    def _finalize_run_metrics(self):
        if self._metric_runs_finalized == 0:
            return

        self.rhoS_T = self._metric_average("rhoS_T")
        self.rhoS_purity = self._metric_average("rhoS_purity")
        self.rhoS_BlochVecs = []
        self.rhoS_BlochEvals = []
        for rho_S in self.rhoS_T:
            rho_S_eval, rho_S_evec = np.linalg.eig(rho_S)
            self.rhoS_BlochVecs.append([rho_S_evec[:, 0], rho_S_evec[:, 1]])
            self.rhoS_BlochEvals.append(rho_S_eval)

        for attr in ("SBS_qre", "SBS_trace_dist", "SBS_fid"):
            averaged = [None]
            for frag_size in range(1, self.QREmaxFragSize + 1):
                averaged.append(self._metric_average(f"{attr}_{frag_size}"))
            setattr(self, attr, averaged)

        if self.store_correlators:
            self.C_correlators = self._metric_average("C_correlators").astype(np.complex64)
            self.C_connected_correlators = self._metric_average("C_connected_correlators").astype(np.complex64)
            self.C_SE_correlators = self._metric_average("C_SE_correlators").astype(np.complex64)
            self.C_correlators_abs_runAv = self._metric_average("C_correlators_abs").astype(np.float32)
            self.C_connected_correlators_abs_runAv = self._metric_average("C_connected_correlators_abs").astype(np.float32)
            self.C_SE_correlators_abs_runAv = self._metric_average("C_SE_correlators_abs").astype(np.float32)
        
    def _plot_fraction_surface(self, Tplot, Fplot, data, title, zlabel, thetastr):
        X, Y = np.meshgrid(Tplot, Fplot)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)
        ax.set_xlabel("T")
        ax.set_ylabel("Frac")
        ax.set_zlabel(zlabel)
        os.makedirs("figs", exist_ok=True)
        plt.savefig('.//figs//'+title+'_p='+str(self.psi_bias)+'_theta='+f"{thetastr:.2f}"+'.jpg',dpi=300)
        plt.show()

    def plot_I_S(self):
        Tplot=np.linspace(0,self.T, self.TstepsPrint+1)
        Fplot=np.arange(0,self.nqubits)/self.nqubitsE
        
            
        title='Ent_VnS_N='+str(self.nqubits)
        # plt.plot(np.arange(1,nqubitsE+1)/nqubitsE,I_S_Ef_fractionsT[:,-1])
        plt.plot(Tplot,self.S_vn_runAv)
        plt.title(title)
        plt.ylabel('Ent_VnS')
        plt.xlabel('T')
        thetastr=self.params["Mironowicz_theta"]
        os.makedirs("figs", exist_ok=True)
        plt.savefig('.//figs//'+title+'_p='+str(self.psi_bias)+'_theta='+f"{thetastr:.2f}"+'.jpg',dpi=300) 
        plt.show()

        self._plot_fraction_surface(
            Tplot,
            Fplot,
            self.I_S_Ef_fractionsT_runAv,
            'ISE_3Dplot_N=' + str(self.nqubits),
            'I(S:E)',
            thetastr,
        )
        if getattr(self, "compute_discord", True) and hasattr(self, "Holevo_Z_S_Ef_fractionsT_runAv"):
            self._plot_fraction_surface(
                Tplot,
                Fplot,
                self.Holevo_Z_S_Ef_fractionsT_runAv,
                'HolevoZ_3Dplot_N=' + str(self.nqubits),
                'chi_Z(S:E)',
                thetastr,
            )
        if getattr(self, "compute_discord", True) and hasattr(self, "Discord_Z_S_Ef_fractionsT_runAv"):
            self._plot_fraction_surface(
                Tplot,
                Fplot,
                self.Discord_Z_S_Ef_fractionsT_runAv,
                'DiscordZ_3Dplot_N=' + str(self.nqubits),
                'D_Z(S:E)',
                thetastr,
            )
   
    def plot_TraceDist_fractions(self):
        Tplot=np.linspace(0,self.T, self.TstepsPrint+1)
        Fplot=np.arange(0,self.nqubits)/self.nqubitsE
        
            
        title='Trace Dist N='+str(self.nqubits)
        # plt.plot(np.arange(1,nqubitsE+1)/nqubitsE,I_S_Ef_fractionsT[:,-1])
        plt.plot(Tplot,self.S_vn_runAv)
        plt.title(title)
        plt.ylabel('Trace Distance')
        plt.xlabel('T')
        # plt.savefig('figs//'+title+'.jpg',dpi=300) 
        plt.show()

        X, Y = np.meshgrid(Tplot, Fplot)
    
        # Create 3D plot
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, self.TraceDist_fractionsT_runAv, cmap='viridis', edgecolor='none')
        title='TD_3Dplot_N='+str(self.nqubits)
        # Add color bar and labels
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)
        ax.set_xlabel("T")
        ax.set_ylabel("Frac")
        ax.set_zlabel("Trace Dist")
        # plt.savefig('figs//'+title+'.jpg',dpi=300) 
        plt.show()     
   
    def precomputeH_cpspCombine(self,H, bundle=1):
        self.bundle=bundle
        mode = self.params.get("H_precompute_mode", "combined_cpu")
        if mode == "combined_cpu":
            try:
                self._precomputeH_combined_cpu(H)
                return
            except MemoryError:
                print("[H] combined_cpu precompute ran out of CPU memory; falling back to grouped_cpu")
            except cp.cuda.memory.OutOfMemoryError:
                print("[H] combined_cpu transfer ran out of GPU memory; falling back to grouped_cpu")
            try:
                self._precomputeH_grouped_cpu(H)
                return
            except MemoryError:
                print("[H] grouped_cpu precompute ran out of CPU memory; falling back to sum_terms")
            except cp.cuda.memory.OutOfMemoryError:
                print("[H] grouped_cpu transfer ran out of GPU memory; falling back to sum_terms")
        elif mode == "grouped_cpu":
            try:
                self._precomputeH_grouped_cpu(H)
                return
            except MemoryError:
                print("[H] grouped_cpu precompute ran out of CPU memory; falling back to sum_terms")
            except cp.cuda.memory.OutOfMemoryError:
                print("[H] grouped_cpu transfer ran out of GPU memory; falling back to sum_terms")
        elif mode != "sum_terms":
            raise ValueError("H_precompute_mode must be 'combined_cpu', 'grouped_cpu', or 'sum_terms'")

        self._precomputeH_sum_terms(H)

    def _precomputeH_combined_cpu(self, H):
        dtype = np.dtype(cp.dtype(self.dtypec).name)
        H_cpu = spsp.csr_matrix((self.dim, self.dim), dtype=dtype)
        term_count = 0

        def add_entries(rows, cols, data):
            nonlocal H_cpu, term_count
            if data.size == 0:
                return
            term = spsp.coo_matrix(
                (data, (rows, cols)),
                shape=(self.dim, self.dim),
                dtype=dtype,
            ).tocsr()
            term.sum_duplicates()
            term.eliminate_zeros()
            if term.nnz == 0:
                return
            H_cpu = H_cpu + term
            H_cpu.sum_duplicates()
            term_count += 1

        for i, bond in enumerate(H.bondsSE):
            if _include_coupling(H.J_SE[i]):
                add_entries(
                    *embedded_two_qubit_entries_np(
                        H.J_SE[i] * H.H_SE_matrix,
                        bond[0],
                        bond[1],
                        self.nqubits,
                        dtypecp=self.dtypec,
                    )
                )

        if len(H.J_E) == self.nqubitsE:
            for i in range(self.nqubitsE):
                if _include_coupling(H.J_E[i]):
                    add_entries(
                        *embedded_one_qubit_entries_np(
                            H.J_E[i] * H.H_E_matrix,
                            i + self.nqubitsS,
                            self.nqubits,
                            dtypecp=self.dtypec,
                        )
                    )

        for i, bond in enumerate(H.bondsEE):
            if _include_coupling(H.J_EE[i]):
                add_entries(
                    *embedded_two_qubit_entries_np(
                        H.J_EE[i] * H.H_EE_matrix,
                        bond[0],
                        bond[1],
                        self.nqubits,
                        dtypecp=self.dtypec,
                    )
                )

        H_cpu.eliminate_zeros()
        self.H_all = cpsp.csr_matrix(
            (
                cp.asarray(H_cpu.data, dtype=self.dtypec),
                cp.asarray(H_cpu.indices),
                cp.asarray(H_cpu.indptr),
            ),
            shape=H_cpu.shape,
            dtype=self.dtypec,
        )
        print(f"[H] precompute mode=combined_cpu terms={term_count} nnz={self.H_all.nnz}")
        gpu_memory_tracker.sample("H precompute combined CPU transferred", include_device=True)

    def _precomputeH_grouped_cpu(self, H):
        dtype = np.dtype(cp.dtype(self.dtypec).name)
        group_size = int(round(float(self.params.get("H_precompute_group_size", 8))))
        group_size = max(1, group_size)
        diag_cpu = np.zeros(self.dim, dtype=dtype)
        groups = []
        group_cpu = spsp.csr_matrix((self.dim, self.dim), dtype=dtype)
        group_terms = 0
        diagonal_terms = 0
        sparse_terms = 0

        def flush_group():
            nonlocal group_cpu, group_terms
            group_cpu.eliminate_zeros()
            if group_cpu.nnz:
                groups.append(self._csr_cpu_to_gpu(group_cpu))
                gpu_memory_tracker.sample(f"H precompute grouped_cpu group {len(groups)} transferred", include_device=True)
            group_cpu = spsp.csr_matrix((self.dim, self.dim), dtype=dtype)
            group_terms = 0

        def add_sparse_entries(rows, cols, data):
            nonlocal group_cpu, group_terms, sparse_terms
            if data.size == 0:
                return
            term = spsp.coo_matrix(
                (data, (rows, cols)),
                shape=(self.dim, self.dim),
                dtype=dtype,
            ).tocsr()
            term.sum_duplicates()
            term.eliminate_zeros()
            if term.nnz == 0:
                return
            group_cpu = group_cpu + term
            group_cpu.sum_duplicates()
            group_terms += 1
            sparse_terms += 1
            if group_terms >= group_size:
                flush_group()

        def add_one_qubit(local_matrix, qubit):
            nonlocal diagonal_terms
            local_np = self._local_matrix_np(local_matrix, dtype)
            local_diag = np.diag(local_np)
            if np.any(local_diag):
                self._add_one_qubit_diagonal_np(diag_cpu, local_diag, qubit)
                diagonal_terms += 1
            offdiag_np = local_np.copy()
            np.fill_diagonal(offdiag_np, 0)
            if np.count_nonzero(offdiag_np):
                add_sparse_entries(
                    *embedded_one_qubit_entries_np(
                        offdiag_np,
                        qubit,
                        self.nqubits,
                        dtypecp=self.dtypec,
                    )
                )

        def add_two_qubit(local_matrix, left, right):
            nonlocal diagonal_terms
            local_np = self._local_matrix_np(local_matrix, dtype)
            local_diag = np.diag(local_np)
            if np.any(local_diag):
                self._add_two_qubit_diagonal_np(diag_cpu, local_diag, left, right)
                diagonal_terms += 1
            offdiag_np = local_np.copy()
            np.fill_diagonal(offdiag_np, 0)
            if np.count_nonzero(offdiag_np):
                add_sparse_entries(
                    *embedded_two_qubit_entries_np(
                        offdiag_np,
                        left,
                        right,
                        self.nqubits,
                        dtypecp=self.dtypec,
                    )
                )

        for i, bond in enumerate(H.bondsSE):
            if _include_coupling(H.J_SE[i]):
                add_two_qubit(H.J_SE[i] * H.H_SE_matrix, bond[0], bond[1])

        if len(H.J_E) == self.nqubitsE:
            for i in range(self.nqubitsE):
                if _include_coupling(H.J_E[i]):
                    add_one_qubit(H.J_E[i] * H.H_E_matrix, i + self.nqubitsS)

        for i, bond in enumerate(H.bondsEE):
            if _include_coupling(H.J_EE[i]):
                add_two_qubit(H.J_EE[i] * H.H_EE_matrix, bond[0], bond[1])

        flush_group()
        diagonal = cp.asarray(diag_cpu, dtype=self.dtypec) if np.any(diag_cpu) else None
        self.H_all = HybridSparseOperator(diagonal, groups, shape=(self.dim, self.dim), dtype=self.dtypec)
        print(
            "[H] precompute mode=grouped_cpu "
            f"group_size={group_size} diagonal_terms={diagonal_terms} "
            f"sparse_terms={sparse_terms} groups={len(groups)} nnz={self.H_all.nnz}"
        )
        gpu_memory_tracker.sample("H precompute grouped_cpu finalized", include_device=True)

    def _csr_cpu_to_gpu(self, matrix):
        return cpsp.csr_matrix(
            (
                cp.asarray(matrix.data, dtype=self.dtypec),
                cp.asarray(matrix.indices),
                cp.asarray(matrix.indptr),
            ),
            shape=matrix.shape,
            dtype=self.dtypec,
        )

    def _local_matrix_np(self, matrix, dtype):
        if isinstance(matrix, cp.ndarray):
            matrix = cp.asnumpy(matrix)
        return np.asarray(matrix, dtype=dtype)

    def _add_one_qubit_diagonal_np(self, diagonal, local_diag, qubit):
        bit = self.nqubits - 1 - qubit
        for start in range(0, self.dim, 1 << 22):
            stop = min(start + (1 << 22), self.dim)
            basis = np.arange(start, stop, dtype=np.int64)
            local = (basis >> bit) & 1
            diagonal[start:stop] += local_diag[local]

    def _add_two_qubit_diagonal_np(self, diagonal, local_diag, left, right):
        bit_left = self.nqubits - 1 - left
        bit_right = self.nqubits - 1 - right
        for start in range(0, self.dim, 1 << 22):
            stop = min(start + (1 << 22), self.dim)
            basis = np.arange(start, stop, dtype=np.int64)
            local = (((basis >> bit_left) & 1) << 1) | ((basis >> bit_right) & 1)
            diagonal[start:stop] += local_diag[local]

    def _precomputeH_sum_terms(self, H):

        terms = []

        def add_entries(rows, cols, data, label):
            if data.size == 0:
                return
            term = cpsp.coo_matrix(
                (data, (rows, cols)),
                shape=(self.dim, self.dim),
                dtype=self.dtypec,
            ).tocsr()
            term.sum_duplicates()
            gpu_memory_tracker.sample(f"H precompute {label} term CSR")
            terms.append(term)
            gpu_memory_tracker.sample(f"H precompute {label} stored")
            del rows, cols, data
            cleanup_cupy_memory(f"H precompute {label} pool cleanup", synchronize=False, collect=False)

        #Do SE
        for [i,bond] in enumerate(H.bondsSE):
            if _include_coupling(H.J_SE[i]):
                add_entries(
                    *embedded_two_qubit_entries_cp(H.J_SE[i]*H.H_SE_matrix,bond[0],bond[1],self.nqubits,dtypecp=self.dtypec),
                    label=f"SE {i}",
                )
        #Do E
        if len(H.J_E)==self.nqubitsE:
            for i in range(self.nqubitsE):
                if _include_coupling(H.J_E[i]):
                    add_entries(
                        *embedded_one_qubit_entries_cp(H.J_E[i]*H.H_E_matrix,i+self.nqubitsS,self.nqubits,self.dtypec),
                        label=f"E {i}",
                    )
                 
        #Do EE
        for [i,bond] in enumerate(H.bondsEE):
            if _include_coupling(H.J_EE[i]):
                add_entries(
                    *embedded_two_qubit_entries_cp(H.J_EE[i]*H.H_EE_matrix,bond[0],bond[1],self.nqubits,dtypecp=self.dtypec),
                    label=f"EE {i}",
                )

        self.H_all = SumSparseOperator(terms, shape=(self.dim, self.dim), dtype=self.dtypec)
        print(f"[H] precompute mode=sum_terms terms={len(terms)} nnz={self.H_all.nnz}")
        gpu_memory_tracker.sample("H precompute terms finalized", include_device=True)


    def apply_noise_Gate(self,H,params):
        if H.noiseOn>0:
            applyseq=[]
            for i in range(self.nqubitsE):
                randtrigger=random.uniform(0, 1) <= H.p_noise
                if randtrigger==1:
                    applyseq.append(1)
                else:
                    applyseq.append(0)
                
            if sum(applyseq)>0:
                self.psi = self.psi_real + 1j* self.psi_imag
                
                for i in range(self.nqubitsE):
                    if applyseq[i]:
                        self.psi=H.noiseU_Gate_N[i]@ self.psi
                         
                self.psi_real = cp.real(self.psi).astype(self.dtypec)
                self.psi_imag = cp.imag(self.psi).astype(self.dtypec)
        
        
        # def apply_noise_Rand(self,H,params):
        #     if H.noiseOn>0:
                
        #         self.H_noise_Matrix=cpsp.csr_matrix((self.dim, self.dim), dtype=self.dtypec)
                
        #         for n in H.noise_times:

        #         if sum(applyseq)>0:
        #             self.psi = self.psi_real + 1j* self.psi_imag
                    
        #             for i in range(self.nqubitsE):
        #                 if applyseq[i]:
        #                     self.psi=H.noiseU_Gate_N[i]@ self.psi
                             
        #             self.psi_real=cp.real(self.psi)
        #             self.psi_imag=cp.imag(self.psi)
        
    def evolve_euler(self, H, dt):
        self.psiparts()
        self.psi -= 1j * self.H_all @ self.psi * dt
        self.psi_real = cp.real(self.psi).astype(self.dtypec)
        self.psi_imag = cp.imag(self.psi).astype(self.dtypec)
            
    def _diag_phi_matmul(self, H_R):
        """One-shot diagnostic: verify H_R @ phi (SpMM) is correct.
        Uses a random test matrix so the check is valid even when phi is all-zeros at init."""
        if getattr(self, '_phi_diag_done', False):
            return
        self._phi_diag_done = True
        rng = cp.random.default_rng(42)
        dtype = cp.float32 if self.dtypef == cp.float32 else cp.float64
        test_phi = rng.standard_normal((self.dim, 2)).astype(dtype)
        result = H_R @ test_phi
        col0_ok = cp.allclose(result[:, 0], H_R @ test_phi[:, 0], atol=1e-4)
        col1_ok = cp.allclose(result[:, 1], H_R @ test_phi[:, 1], atol=1e-4)
        spmm_ok = bool(col0_ok) and bool(col1_ok)
        self._phi_use_loop = not spmm_ok
        print(f"[DIAG] SpMM {'OK - using batched multiply' if spmm_ok else 'BROKEN - falling back to column loop'}")

    def _H_phi(self, H, phi):
        """Apply sparse H to phi [dim, nSources]. Uses SpMM if verified correct, else column loop."""
        if getattr(self, '_phi_use_loop', False):
            return cp.stack([H @ phi[:, j] for j in range(phi.shape[1])], axis=1)
        return H @ phi

    def _add_perf_time(self, key, value):
        p = self._perf
        p[key] = p.get(key, 0.0) + value

    def _add_perf_count(self, key, value=1):
        p = self._perf
        p[key] = p.get(key, 0) + value

    def _time_gpu(self, key, fn):
        if not self._profile_current_evolve_step:
            return fn()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        self._add_perf_time(key, cp.cuda.get_elapsed_time(start, end) / 1000.0)
        self._add_perf_count(f"{key}_count")
        return result

    def _time_cpu(self, key, fn):
        if not self._profile_current_evolve_step:
            return fn()

        t0 = time.perf_counter()
        result = fn()
        self._add_perf_time(key, time.perf_counter() - t0)
        self._add_perf_count(f"{key}_count")
        return result

    def _evolve_staggered(self, H_matrix, dt):
        half_dt = 0.5 * dt
        self.psi_imag = self._time_gpu(
            "evolve_psi_imag",
            lambda: self.psi_imag - H_matrix @ self.psi_real * half_dt,
        )
        self.psi_real = self._time_gpu(
            "evolve_psi_real",
            lambda: self.psi_real + H_matrix @ self.psi_imag * dt,
        )
        self.psi_imag = self._time_gpu(
            "evolve_psi_imag",
            lambda: self.psi_imag - H_matrix @ self.psi_real * half_dt,
        )

        if self.store_correlators and getattr(self, "correlators_active", True):
            self._add_perf_count("evolve_phi_steps")
            if getattr(self, '_phi_diag_done', False):
                self._diag_phi_matmul(H_matrix)
            else:
                self._time_gpu("evolve_phi_diag", lambda: self._diag_phi_matmul(H_matrix))
            self.phi_imag = self._time_gpu(
                "evolve_phi_imag",
                lambda: self.phi_imag - self._H_phi(H_matrix, self.phi_real) * half_dt,
            )
            self.phi_real = self._time_gpu(
                "evolve_phi_real",
                lambda: self.phi_real + self._H_phi(H_matrix, self.phi_imag) * dt,
            )
            self.phi_imag = self._time_gpu(
                "evolve_phi_imag",
                lambda: self.phi_imag - self._H_phi(H_matrix, self.phi_real) * half_dt,
            )
        self.psi = None

    def evolve_leapfrog(self,params,H,dt,iterateRe, combine=False):
        if not combine:
            raise ValueError("term-by-term leapfrog evolution is no longer supported")
        if combine and iterateRe:
            self.evolve_step(H, dt)

    def evolve_step(self, H, dt):
        """Advance one staggered leapfrog step."""
        self._add_perf_count("evolve_steps")
        profiled_steps = self._perf.get("evolve_profiled_steps", 0)
        self._profile_current_evolve_step = (
            self.profile_evolution and profiled_steps < self.profile_evolution_max_steps
        )
        try:
            if H.noiseT > 0:
                H_eff = self._time_cpu("evolve_noise_h_add", lambda: self.H_all + H.H_noise)
                self._evolve_staggered(H_eff, dt)
                return

            self._evolve_staggered(self.H_all, dt)
        finally:
            if self._profile_current_evolve_step:
                self._add_perf_count("evolve_profiled_steps")
            self._profile_current_evolve_step = False

        
    def psiparts(self):
        self.psi = self.psi_real + 1j * self.psi_imag

    def record_branch_energy_diagnostics(self, H_store):
        """Measure storage-energy density using the existing SBS pointer-branch labels."""
        psi_phys = self.psi_real + 1j * self.psi_imag
        psi_by_system = psi_phys.reshape((2, self.dimE))
        densities = []
        for branch in range(2):
            system_row = 1 - branch
            branch_state = cp.zeros_like(psi_by_system)
            branch_state[system_row] = psi_by_system[system_row]
            branch_state = branch_state.reshape(self.dim)
            probability = float(cp.real(cp.vdot(branch_state, branch_state)).item())
            self.branch_probabilities[branch, self.runN] = probability
            if probability <= 1e-12:
                densities.append(np.nan)
                continue
            branch_state /= np.sqrt(probability)
            H_branch = H_store @ branch_state
            energy = float(cp.real(cp.vdot(branch_state, H_branch)).item())
            variance = max(0.0, float(cp.real(cp.vdot(H_branch, H_branch)).item()) - energy**2)
            density = energy / self.nqubitsE
            self.branch_energy_density[branch, self.runN] = density
            self.branch_energy_density_std[branch, self.runN] = np.sqrt(variance) / self.nqubitsE
            densities.append(density)
        if np.all(np.isfinite(densities)):
            self.branch_energy_density_gap[self.runN] = abs(densities[0] - densities[1])

    def reconstructed_norm(self):
        norm_sq = cp.linalg.norm(self.psi_real) ** 2 + cp.linalg.norm(self.psi_imag) ** 2
        overlap = cp.vdot(self.psi_real, self.psi_imag)
        norm_sq += 2 * cp.real(1j * overlap)
        return cp.sqrt(cp.maximum(cp.real(norm_sq), 0)).item()

    def modified_staggered_norm(self, H_matrix=None, dt=None):
        if H_matrix is None:
            H_matrix = self.H_all
        if dt is None:
            dt = self.dT
        base = cp.linalg.norm(self.psi_real) ** 2 + cp.linalg.norm(self.psi_imag) ** 2
        h_real = H_matrix @ self.psi_real
        correction = 0.25 * dt * dt * cp.linalg.norm(h_real) ** 2
        invariant = cp.real(base - correction)
        return cp.sqrt(cp.maximum(invariant, 0)).item()

    def print_norm(self, H_matrix=None, dt=None):
        norm = self.reconstructed_norm()
        modified_norm = self.modified_staggered_norm(H_matrix=H_matrix, dt=dt)
        self.norms.append(norm)
        self.modified_norms.append(modified_norm)
        print(
            f"  [norm] ||psi||={norm:.8e}  dev={norm - 1:+.3e}  "
            f"modified={modified_norm:.8e}  mod_dev={modified_norm - 1:+.3e}"
        )
        return norm, modified_norm
        
    def plot_norm(self):
        
        normsarr=np.array(self.norms)
        
        max_dev=max(abs(normsarr-1))
        min_val=min(normsarr)
        max_val=max(normsarr)
        
        mean_val=np.mean(normsarr)
        
        plt.axhline(y=min_val, color='red', linestyle='--', label=f'Min: {min_val}')
        plt.axhline(y=max_val, color='green', linestyle='--', label=f'Max: {max_val}')
        
        
        t= np.arange(int(self.T/self.printT) + 1)
        plt.plot(t,normsarr)
        plt.ylabel("Norm")
        plt.xlabel('T')
        title=f'Norm, max dev= {max_dev:.{5}f} Av= {mean_val:.{5}f}'
        plt.title(title)
        # plt.savefig('figs//'+title+'.jpg',dpi=300) 
        plt.show()
    
    def print_exp(self,op):
        print(cp.vdot(self.psi, op.op_cpsp @ self.psi))
        
    def print_vn(self,ti,maxcombs, halfN=False):
        _t0_vn = time.time()

        dims_cp=[2]*self.nqubits
        nqubits=self.nqubits
        nqubitsE=self.nqubitsE
        nqubitsS=self.nqubitsS
        compute_discord = self.compute_discord

        # --- rho_S ---
        _ts = time.time()
        rho_S=partial_trace_cp(self.psi, list(range(nqubitsS )), dims_cp)
        rho_S_np=(rho_S.get())
        self.rhoS_T.append(rho_S_np)
        self.rhoS_purity.append(abs(np.trace(rho_S_np@rho_S_np)))
        rho_S_eval, rho_S_evec = np.linalg.eig(rho_S_np)
        self.rhoS_BlochVecs.append([rho_S_evec[:,0],rho_S_evec[:,1]])
        self.rhoS_BlochEvals.append(rho_S_eval)
        _t_rhoS = time.time() - _ts

        # --- SBS fragments ---
        _ts = time.time()
        SBS0_psi=self.pointerProj0 @ self.psi
        SBS1_psi=self.pointerProj1 @ self.psi

        norm0 = cp.linalg.norm(SBS0_psi).item()
        norm1 = cp.linalg.norm(SBS1_psi).item()
        prob0 = norm0 * norm0
        prob1 = norm1 * norm1
        prob_total = prob0 + prob1
        branch0_valid = norm0 > 1e-12
        branch1_valid = norm1 > 1e-12
        sbs_states_valid = branch0_valid and branch1_valid

        if branch0_valid:
            SBS0_psi /= norm0
        if branch1_valid:
            SBS1_psi /= norm1

        _sbs_syncs = 0
        for f in range(1, self.QREmaxFragSize + 1):
            for i in range(self.nqubitsE - f + 1):
                if not sbs_states_valid:
                    self.SBS_trace_dist[f][i, self.printTnow] = np.nan
                    self.SBS_qre[f][i, self.printTnow] = np.nan
                    self.SBS_fid[f][i, self.printTnow] = np.nan
                    continue
                frag = list(range(i, i + f))
                rho_E_i_0 = partial_trace_cp(SBS0_psi, frag, [2] * self.nqubitsE)
                rho_E_i_1 = partial_trace_cp(SBS1_psi, frag, [2] * self.nqubitsE)
                rho_E_i_0_np = rho_E_i_0.get()
                rho_E_i_1_np = rho_E_i_1.get()
                _sbs_syncs += 2
                self.SBS_trace_dist[f][i, self.printTnow] = trace_distance(rho_E_i_0_np, rho_E_i_1_np)
                self.SBS_qre[f][i, self.printTnow]        = quantum_relative_entropy(rho_E_i_0_np, rho_E_i_1_np)
                self.SBS_fid[f][i, self.printTnow]        = fidelity(rho_E_i_0_np, rho_E_i_1_np)
        _t_sbs = time.time() - _ts

        # --- S_vn ---
        _ts = time.time()
        self.S_vn[ti,self.runN]=clamp_down(max(0,vn_entropy_np(rho_S_np)),self.clampISE)
        _t_svn = time.time() - _ts

        # --- mutual information ---
        _ts = time.time()
        _mi_entropy_calls = 0
        A_qubits = list(range(nqubitsS))
        S_A = von_neumann_entropy_direct(self.psi, A_qubits, nqubits)  # cached: same for all MI calls
        _mi_entropy_calls += 1
        branch_states = []
        if compute_discord and prob_total > 1e-12:
            if branch0_valid:
                branch_states.append((prob0 / prob_total, SBS0_psi))
            if branch1_valid:
                branch_states.append((prob1 / prob_total, SBS1_psi))

        def store_fraction_metric(prefix, ef, values):
            arr = np.asarray(values, dtype=float)
            valid = arr[np.isfinite(arr)]
            target = getattr(self, f"{prefix}_fractionsT")
            target_std = getattr(self, f"{prefix}_fractionsT_STD")
            target_nsamples = getattr(self, f"{prefix}_fractionsT_Nsamples")
            if valid.size == 0:
                target[ef, ti, self.runN] = np.nan
                target_std[ef, ti, self.runN] = np.nan
                target_nsamples[ef, ti, self.runN] = 0
                return
            target[ef, ti, self.runN] = float(np.mean(valid))
            target_std[ef, ti, self.runN] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
            target_nsamples[ef, ti, self.runN] = int(valid.size)
            target_samples = getattr(self, f"{prefix}_fractionsT_samples", None)
            if target_samples is not None:
                target_samples[ef, ti, self.runN, :arr.size] = arr
            target_quantiles = getattr(self, f"{prefix}_fractionsT_quantiles", None)
            if target_quantiles is not None:
                target_quantiles[:, ef, ti, self.runN] = np.quantile(
                    valid,
                    self.fragment_quantile_levels,
                )

        def z_holevo_for_fragment(S_B, comb):
            if not branch_states:
                return np.nan, 0
            branch_frag = [q - nqubitsS for q in comb]
            conditional_entropy = 0.0
            entropy_calls = 0
            for weight, branch_state in branch_states:
                conditional_entropy += weight * von_neumann_entropy_direct(branch_state, branch_frag, nqubitsE)
                entropy_calls += 1
            holevo = clamp_down(max(0.0, S_B - conditional_entropy), self.clampISE)
            return holevo, entropy_calls

        if halfN:
            for ef in _half_fragment_sizes(nqubitsE):
                Nmax=math.comb(nqubitsE , ef)
                Env_qubits=self._fragment_combinations(ef, min(maxcombs, Nmax))
                IIIs=[]
                holevos=[] if compute_discord else None
                discords=[] if compute_discord else None
                for comb in Env_qubits:
                    S_B  = von_neumann_entropy_direct(self.psi, list(comb), nqubits)
                    S_AB = von_neumann_entropy_direct(self.psi, A_qubits + list(comb), nqubits)
                    mi_value = clamp_down(max(0, S_A + S_B - S_AB), self.clampISE)
                    IIIs.append(mi_value)
                    _mi_entropy_calls += 2
                    if compute_discord:
                        holevo_value, holevo_entropy_calls = z_holevo_for_fragment(S_B, comb)
                        discord_value = (
                            np.nan
                            if not np.isfinite(holevo_value)
                            else clamp_down(max(0.0, mi_value - holevo_value), self.clampISE)
                        )
                        holevos.append(holevo_value)
                        discords.append(discord_value)
                        _mi_entropy_calls += holevo_entropy_calls
                store_fraction_metric("I_S_Ef", ef, IIIs)
                if compute_discord:
                    store_fraction_metric("Holevo_Z_S_Ef", ef, holevos)
                    store_fraction_metric("Discord_Z_S_Ef", ef, discords)
            for [ei,ef] in enumerate(range((nqubitsE+1) // 2  +1, nqubitsE+1)):
                if ef==nqubitsE:
                    self.I_S_Ef_fractionsT[ef,ti,self.runN]=2*(self.I_S_Ef_fractionsT[nqubitsE//2,ti,self.runN])
                    self.I_S_Ef_fractionsT_STD[ef,ti,self.runN]=self.I_S_Ef_fractionsT_STD[nqubitsE//2-ei -1 ,ti,self.runN]
                    self.I_S_Ef_fractionsT_Nsamples[ef,ti,self.runN]=self.I_S_Ef_fractionsT_Nsamples[nqubitsE//2-ei -1 ,ti,self.runN]
                else:
                    self.I_S_Ef_fractionsT[ef,ti,self.runN]=self.I_S_Ef_fractionsT[nqubitsE//2,ti,self.runN] +2*(self.I_S_Ef_fractionsT[nqubitsE//2,ti,self.runN]-self.I_S_Ef_fractionsT[nqubitsE//2-ei -1 ,ti,self.runN])
                    self.I_S_Ef_fractionsT_STD[ef,ti,self.runN]=self.I_S_Ef_fractionsT_STD[nqubitsE//2-ei -1 ,ti,self.runN]
                    self.I_S_Ef_fractionsT_Nsamples[ef,ti,self.runN]=self.I_S_Ef_fractionsT_Nsamples[nqubitsE//2-ei -1 ,ti,self.runN]
                if compute_discord:
                    self.Holevo_Z_S_Ef_fractionsT[ef, ti, self.runN] = np.nan
                    self.Holevo_Z_S_Ef_fractionsT_STD[ef, ti, self.runN] = np.nan
                    self.Holevo_Z_S_Ef_fractionsT_Nsamples[ef, ti, self.runN] = 0
                    self.Discord_Z_S_Ef_fractionsT[ef, ti, self.runN] = np.nan
                    self.Discord_Z_S_Ef_fractionsT_STD[ef, ti, self.runN] = np.nan
                    self.Discord_Z_S_Ef_fractionsT_Nsamples[ef, ti, self.runN] = 0
        else:
            for ef in range(1,nqubitsE+1):
                Nmax=math.comb(nqubitsE , ef)
                Env_qubits=self._fragment_combinations(ef, min(maxcombs, Nmax))
                IIIs=[]
                holevos=[] if compute_discord else None
                discords=[] if compute_discord else None
                for comb in Env_qubits:
                    S_B  = von_neumann_entropy_direct(self.psi, list(comb), nqubits)
                    S_AB = von_neumann_entropy_direct(self.psi, A_qubits + list(comb), nqubits)
                    mi_value = clamp_down(max(0, S_A + S_B - S_AB), self.clampISE)
                    IIIs.append(mi_value)
                    _mi_entropy_calls += 2
                    if compute_discord:
                        holevo_value, holevo_entropy_calls = z_holevo_for_fragment(S_B, comb)
                        discord_value = (
                            np.nan
                            if not np.isfinite(holevo_value)
                            else clamp_down(max(0.0, mi_value - holevo_value), self.clampISE)
                        )
                        holevos.append(holevo_value)
                        discords.append(discord_value)
                        _mi_entropy_calls += holevo_entropy_calls
                store_fraction_metric("I_S_Ef", ef, IIIs)
                if compute_discord:
                    store_fraction_metric("Holevo_Z_S_Ef", ef, holevos)
                    store_fraction_metric("Discord_Z_S_Ef", ef, discords)
        _t_mi = time.time() - _ts

        # --- perf summary ---
        _t_total = time.time() - _t0_vn
        p = self._perf
        p['n_calls'] = p.get('n_calls', 0) + 1
        p['rhoS']    = p.get('rhoS', 0.0) + _t_rhoS
        p['sbs']     = p.get('sbs', 0.0) + _t_sbs
        p['svn']     = p.get('svn', 0.0) + _t_svn
        p['mi']      = p.get('mi', 0.0) + _t_mi
        p['total']   = p.get('total', 0.0) + _t_total
        p['sbs_syncs']  = p.get('sbs_syncs', 0) + _sbs_syncs
        p['mi_entropy_calls'] = p.get('mi_entropy_calls', 0) + _mi_entropy_calls
        gpu_memory_tracker.sample("print_vn before pool cleanup", include_device=True)
        mi_label = "MI/Holevo" if compute_discord else "MI"
        print(f"  [perf] print_vn: rho_S={_t_rhoS:.3f}s  SBS={_t_sbs:.3f}s ({_sbs_syncs} GPU->CPU)  {mi_label}={_t_mi:.3f}s ({_mi_entropy_calls} entropies)  total={_t_total:.3f}s")

        del self.psi
        cleanup_cupy_memory("print_vn after pool cleanup", include_device=True)

    def print_perf_summary(self):
        p = self._perf
        n = p.get('n_calls', 0)
        if n == 0:
            print("[perf] No print_vn calls recorded.")
            return
        total = p.get('total', 0.0)
        print(f"\n[perf] print_vn summary over {n} calls:")
        print(f"  {'section':<10}  {'total':>8}  {'per call':>10}  {'%':>6}")
        print(f"  {'-'*40}")
        mi_label = "MI/Holevo" if getattr(self, "compute_discord", True) else "MI"
        for key, label in [('rhoS','rho_S'), ('sbs','SBS'), ('svn','S_vn'), ('mi',mi_label)]:
            v = p.get(key, 0.0)
            print(f"  {label:<10}  {v:8.3f}s  {v/n:10.3f}s  {100*v/total:6.1f}%")
        print(f"  {'TOTAL':<10}  {total:8.3f}s  {total/n:10.3f}s  {'100.0':>6}%")
        print(f"  SBS GPU->CPU syncs per call : {p.get('sbs_syncs',0)/n:.0f}")
        print(f"  {mi_label} entropy calls per call : {p.get('mi_entropy_calls',0)/n:.0f}")

        steps = p.get('evolve_steps', 0)
        if steps:
            print(f"\n[perf] evolve_step summary over {steps} steps:")
            phi_steps = p.get('evolve_phi_steps', 0)
            print(
                f"  profile_evolution={self.profile_evolution}  "
                f"store_correlators={self.store_correlators}  "
                f"phi_steps={phi_steps}  "
                f"profiled_steps={p.get('evolve_profiled_steps', 0)}"
            )
            if not self.profile_evolution:
                print("  detailed GPU timing disabled; set profile_evolution=true in params to enable")
                if self.store_correlators:
                    print(
                        "  note: store_correlators=true propagates "
                        f"{len(getattr(self, 'correlator_sources', range(self.nqubitsE)))} phi source(s) per step"
                    )
                return

            step_keys = [
                ('evolve_psi_real', 'psi real update'),
                ('evolve_psi_imag', 'psi imag update'),
                ('evolve_phi_real', 'phi real update'),
                ('evolve_phi_imag', 'phi imag update'),
                ('evolve_noise_h_add', 'noise H add'),
            ]
            one_shot_keys = [
                ('evolve_phi_diag', 'phi SpMM diag'),
            ]
            profiled_steps = max(1, p.get('evolve_profiled_steps', 0))
            profiled_total = sum(p.get(key, 0.0) for key, _label in step_keys)
            extrapolated = (profiled_total / profiled_steps) * steps
            print(
                f"  sampled step GPU time={profiled_total:.3f}s  "
                f"per profiled step={profiled_total/profiled_steps:.6f}s  "
                f"extrapolated={extrapolated:.1f}s"
            )
            print(f"  {'section':<18}  {'sample':>8}  {'per profiled step':>17}  {'%':>6}")
            print(f"  {'-'*50}")
            for key, label in step_keys:
                value = p.get(key, 0.0)
                if value == 0:
                    continue
                pct = 100 * value / profiled_total if profiled_total > 0 else 0
                print(f"  {label:<18}  {value:8.3f}s  {value/profiled_steps:17.6f}s  {pct:6.1f}%")
            one_shot_values = [(key, label, p.get(key, 0.0)) for key, label in one_shot_keys if p.get(key, 0.0)]
            if one_shot_values:
                print("  one-shot diagnostics:")
                for key, label, value in one_shot_values:
                    print(f"  {label:<18}  {value:8.3f}s  events={p.get(f'{key}_count', 0)}")
            if self.store_correlators:
                print("  note: phi updates are the per-step cost of enabling two-time correlators")

    def redundancy_fractionScore_at_T(self,T,cutoff=0.5,minfraction=0.4, minTime=0):
        if T < 0:
            return self.redundancy_fractionScore(cutoff, minfraction, minTime=minTime)

        I_SE_tf = self.I_S_Ef_fractionsT_runAv
        fragments = I_SE_tf.shape[0]
        Xdata = np.linspace(0, 1, fragments)
        it = int(T)
        curve = I_SE_tf[:, it]
        curve_max = max(curve)
        global_max = max(I_SE_tf.ravel())
        if curve_max <= cutoff * global_max or it < minTime:
            return [1, -1, -1, -1]

        degree = min(6, max(1, fragments - 1))
        ys = np.poly1d(np.polyfit(Xdata, curve, degree))(Xdata)
        ratios = ys / max(ys)
        frac_index = int(np.argmax(ratios > minfraction)) if np.any(ratios > minfraction) else -1
        return [Xdata[frac_index], it, curve_max, curve_max / global_max]
         
    def redundancy_fractionScore(self,cutoff=0.5,minfraction=0.4,minTime=0):
        
            
            I_SE_tf=self.I_S_Ef_fractionsT_runAv
            
            times=[]
            
            fracs=[]
            
            fragments=I_SE_tf.shape[0]
            Xdata=np.linspace(0,1,fragments)
            
           
            for it in range(I_SE_tf.shape[1]):
                curve=I_SE_tf[:,it]
            
                if (max(curve) > cutoff*max(I_SE_tf.ravel())) and it>=minTime:
                
                    # Fit a 3rd or 4th degree polynomial
                    degree = 6
                    coeffs = np.polyfit(Xdata, curve, degree)
                    poly = np.poly1d(coeffs)
                    
                    
                    ys=poly(Xdata)
                    
                    frac = np.argmax(ys/max(ys) > minfraction) if np.any(ys > minfraction) else -1
                    frac=Xdata[frac]
                    
                    times.append(it)
                    fracs.append(frac)
                
    
            if len(fracs)>0:   
                return [min(fracs),times[fracs.index(min(fracs))],max(curve),max(curve)/max(I_SE_tf.ravel())]
            else:
                return [1,-1,-1,-1]
    def redundancy_slope_at_T(self,T,cutoff=0.5, minTime=0):
        #fit a  curve, take deriv at middle
        if T==-1:
            return self.max_redundancy_slope(cutoff,minTime=minTime)
        else:
            it=T
            
            I_SE_tf=self.I_S_Ef_fractionsT_runAv
            slopes_t=[]
            
            polyfits=[]
            fragments=I_SE_tf.shape[0]
            Xdata=np.linspace(0,1,fragments)
            
            curve=I_SE_tf[:,it]
            
            if (max(curve) > cutoff*max(I_SE_tf.ravel())) and it>=minTime:
            
                # Fit a 3rd or 4th degree polynomial
                degree = 6
                coeffs = np.polyfit(Xdata, curve, degree)
                poly = np.poly1d(coeffs)
                
                # Evaluate the polynomial
                # x_fit = np.linspace(min(Xdata), max(Xdata), 200)
                # y_fit = poly(x_fit)
                
                poly_derivative = np.polyder(poly)
        
                # Evaluate derivative at the midpoint
                # x_mid = np.median(0.5)
                dy_dx_mid = poly_derivative(0.5)
                
                
                
                slopes_t.append(abs(dy_dx_mid)/max(curve))
                
                polyfits.append(coeffs)
    
    
                return [min(slopes_t),it,max(curve),max(curve)/max(I_SE_tf.ravel()),polyfits[slopes_t.index(min(slopes_t))]]
     
         
    def max_redundancy_slope(self,cutoff=0.5, minTime=0):
        #fit a  curve, take deriv at middle
        I_SE_tf=self.I_S_Ef_fractionsT_runAv
        slopes_t=[]
        times=[]
        polyfits=[]
        fragments=I_SE_tf.shape[0]
        Xdata=np.linspace(0,1,fragments)
        for it in range(I_SE_tf.shape[1]):
            curve=I_SE_tf[:,it]
            
            if (max(curve) > cutoff*max(I_SE_tf.ravel())) and it>=minTime:
            
                # Fit a 3rd or 4th degree polynomial
                degree = 6
                coeffs = np.polyfit(Xdata, curve, degree)
                poly = np.poly1d(coeffs)
                
                # Evaluate the polynomial
                # x_fit = np.linspace(min(Xdata), max(Xdata), 200)
                # y_fit = poly(x_fit)
                
                poly_derivative = np.polyder(poly)
        
                # Evaluate derivative at the midpoint
                # x_mid = np.median(0.5)
                dy_dx_mid = poly_derivative(0.5)
                
                
                
                slopes_t.append(abs(dy_dx_mid)/max(curve))
                times.append(it)
                polyfits.append(coeffs)
    
        if len(slopes_t)>0:
            self.QD_slope= [min(slopes_t),times[slopes_t.index(min(slopes_t))],max(curve),max(curve)/max(I_SE_tf.ravel()),polyfits[slopes_t.index(min(slopes_t))]]
            print(self.QD_slope)
            
        else:
            self.QD_slope=[-1,-1,-1,-1,-1]
            
        return self.QD_slope
            
    def print_ISEerrorbarQDslope(self):
        if self.QD_slope[0]!=-1:
            t=self.QD_slope[1] 
            title='ISEplat_T='+str(t*self.printT)+'_N='+str(self.nqubits)
            
            plt.errorbar(np.arange(0,self.nqubitsE+1)/self.nqubitsE,self.I_S_Ef_fractionsT_runAv[:,t], yerr=self.I_S_Ef_fractionsT_STD_runAv[:,t], fmt='-o', capsize=5)
            
            coeffs=self.QD_slope[4] 
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(0, 1, 200)
            y_fit = poly(x_fit)
            plt.plot(x_fit,y_fit,color='k')
            plt.ylabel('I(S:Ef)')
            plt.xlabel('Fraction of E at T='+str(t*self.printT))
            plt.title(title)
            thetastr=self.params["Mironowicz_theta"]
            plt.savefig('.//figs//'+title+'_p='+str(self.psi_bias)+'_theta='+f"{thetastr:.2f}"+'.jpg',dpi=300) 
      
            # plt.savefig('figs//'+title+'.jpg',dpi=300) 
            plt.show()
            
    def get_stored_psi(self, t_idx):
        if self.psi_stored is None:
            raise RuntimeError("psi history storage is disabled; set store_psi=True to enable it")
        complex_dtype = np.complex64 if self.psi_stored.dtype == np.float32 else np.complex128
        return self.psi_stored[t_idx, 0].astype(complex_dtype) + 1j * self.psi_stored[t_idx, 1].astype(complex_dtype)

    def clearMem(self):
        self.psi = None
        self.psi_real = None
        self.psi_imag = None
        self.phi_real = None
        self.phi_imag = None
        self.H_all = None
        self.C_correlators = None
        self.C_connected_correlators = None
        self.C_SE_correlators = None
        self.C_correlators_abs_runAv = None
        self.C_connected_correlators_abs_runAv = None
        self.C_SE_correlators_abs_runAv = None
        self.E_z_initial = None
        for prefix in ("Holevo_Z_S_Ef", "Discord_Z_S_Ef"):
            for suffix in (
                "fractionsT",
                "fractionsT_STD",
                "fractionsT_Nsamples",
                "fractionsT_runAv",
                "fractionsT_STD_runAv",
                "fractionsT_runSTD",
                "fractionsT_runSEM",
                "fractionsT_fragmentSTD_runAv",
                "fractionsT_fragmentSTD_prop",
                "fractionsT_fragmentSEM_runAv",
            ):
                attr = f"{prefix}_{suffix}"
                if hasattr(self, attr):
                    setattr(self, attr, None)
        self.pointerProj0 = None
        self.pointerProj1 = None
        if self.psi_stored is not None:
            del self.psi_stored  # closes memmap handle if on disk, frees RAM if in memory
            self.psi_stored = None
        if self.psi_snapshot_temp_dir is not None:
            shutil.rmtree(self.psi_snapshot_temp_dir, ignore_errors=True)
            self.psi_snapshot_temp_dir = None
        cleanup_cupy_memory("wavefunction.clearMem", include_device=False)
        
        
    def plot_rhoS_Bloch(self, plotMaxEval=True):
        
        plt.rcParams['figure.dpi'] = 300          # <- this makes everything crisp in Spyder
        plt.rcParams['savefig.dpi'] = 300
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_bloch_sphere(ax)
        
        # Time array
        t = np.linspace(0, self.T, int(self.T/self.printT +1))        # change duration and density as you like
        
        # ------------------------------------------------------------------
        # Choose ONE of the three examples below
        # ------------------------------------------------------------------
        
        # Example 1: Precession around Z-axis (like a magnetic field in z)
     
        psi_t = np.zeros((len(t), 2), dtype=complex)
        
        
        for i in range(len(self.rhoS_BlochVecs)):
            arg=np.argmax(abs(self.rhoS_BlochEvals[i]))
            
            if plotMaxEval:
                psi_t[i][0] = self.rhoS_BlochVecs[i][arg][0]
                psi_t[i][1] = self.rhoS_BlochVecs[i][arg][1]
            else:
                psi_t[i][0] = self.rhoS_BlochVecs[i][np.mod(arg+1,2)][0]
                psi_t[i][1] = self.rhoS_BlochVecs[i][np.mod(arg+1,2)][1]
                
      
        
        # ------------------------------------------------------------------
        # Compute Bloch vectors for all times
        # ------------------------------------------------------------------
        bloch_points = np.array([state_to_bloch(psi) for psi in psi_t])
        
        # Plot the trajectory
        ax.plot(bloch_points[:,0], bloch_points[:,1], bloch_points[:,2],
                color='red', linewidth=2.5, label='Trajectory')
        
        # Plot discrete points (optional, makes it clearer)
        ax.scatter(bloch_points[::15,0], bloch_points[::15,1], bloch_points[::15,2],
                   color='darkred', s=30)
        
        # Plot start and end points in different colors
        ax.scatter(*bloch_points[0], color='lime', s=100, label='Start', zorder=5)
        ax.scatter(*bloch_points[-1], color='purple', s=100, label='End', zorder=5)
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2,1.2])
        ax.set_axis_off()
        ax.set_title("Time evolution of a qubit on the Bloch sphere, EvalMax "+str(plotMaxEval), pad=30, size=16)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        
    def plot_rhoS_Evals(self):
        
        plt.rcParams['figure.dpi'] = 300          # <- this makes everything crisp in Spyder
        plt.rcParams['savefig.dpi'] = 300
        
        t = np.linspace(0, self.T, int(self.T/self.printT +1))        # change duration and density as you like
        
        
        evalMax=[]
        evalMin=[]
        
        for i in range(len(self.rhoS_BlochVecs)):
            arg=np.argmax(abs(self.rhoS_BlochEvals[i]))

            evalMax.append(self.rhoS_BlochEvals[i][arg])
            evalMin.append(self.rhoS_BlochEvals[i][np.mod(arg+1,2)])
      
        fig, ax = plt.subplots()
        ax.plot(t,abs(np.array(evalMax)),label='max e')
        ax.plot(t,abs(np.array(evalMin)),label='min e')
        ax.set_xlabel("Time")
        ax.set_ylabel('|e|')
        ax.set_title("Max and Min |e| of rho_S")
        ax.legend()
        plt.show()
       
    def plot_Psi_E_init(self):
        overlapUD = self._initial_environment_z_overlaps()

        fig, ax = plt.subplots()
        ax.plot(overlapUD)
        ax.set_xlabel("E Qubit")
        ax.set_ylabel("Max initial Z-basis population")
        plt.show()
        
        return overlapUD

    def _initial_environment_z_overlaps(self):
        probs = cp.abs(self.psi_E_init.reshape([2] * self.nqubitsE)) ** 2
        overlapUD = []
        for i in range(self.nqubitsE):
            p_up = cp.sum(cp.take(probs, 0, axis=i)).item()
            p_down = cp.sum(cp.take(probs, 1, axis=i)).item()
            overlapUD.append(max(float(p_up), float(p_down)))
        return overlapUD
        
    def plot_RhoSoffdiagDecay(self):
        title="Decay of off-diagonal element (decoherence)"
        t = self._sample_times()
        fig, ax = plt.subplots()
        ax.plot(t, abs(np.array([x[1][0] for x in self.rhoS_T])))
        ax.set_xlabel("Time")
        ax.set_ylabel("|rho_01|")
        ax.set_title(title)
        plt.show()
    
    def _apply_environment_pauli(self, state, environment_index):
        indices = cp.arange(self.dim, dtype=cp.int32)
        bit_pos = self.nqubitsE - 1 - environment_index
        bits = (indices >> bit_pos) & 1
        if self.correlator_axis == "Z":
            return (1 - 2 * bits).astype(self.dtypef) * state
        flipped = state[indices ^ (1 << bit_pos)]
        if self.correlator_axis == "X":
            return flipped
        phase = cp.where(bits == 0, -1j, 1j).astype(self.dtypec)
        return phase * flipped

    def buildPhi(self):
        """Initialize selected local-operator perturbations at the configured reference time."""
        if not self.store_correlators:
            self.phi_real = None
            self.phi_imag = None
            return
        psi_phys = self.psi_real + 1j * self.psi_imag
        n_sources = len(self.correlator_sources)
        self.E_z_initial = cp.empty(n_sources, dtype=self.dtypef)
        self.phi_real = cp.zeros((self.dim, n_sources), dtype=self.dtypec)
        self.phi_imag = cp.zeros((self.dim, n_sources), dtype=self.dtypec)
        for column, source in enumerate(self.correlator_sources):
            op_psi = self._apply_environment_pauli(psi_phys, source)
            self.E_z_initial[column] = cp.real(cp.vdot(psi_phys, op_psi))
            self.phi_real[:, column] = self._apply_environment_pauli(self.psi_real, source)
            self.phi_imag[:, column] = self._apply_environment_pauli(self.psi_imag, source)
        self.correlators_active = True

    def print_correlators(self, t_idx):
        """
        Compute the selected record-axis two-time correlator for all target sites and configured sources.

        The staggered work arrays may be complex for transverse Hamiltonians,
        so the physical states are reconstructed locally for the inner product.

        Stores result into self.C_correlators[t_idx].
        """
        if not self.correlators_active:
            return
        indices = cp.arange(self.dim, dtype=cp.int32)
        psi_phys = self.psi_real + 1j * self.psi_imag
        phi_phys = self.phi_real + 1j * self.phi_imag

        for i in range(self.nqubitsE):
            op_psi = self._apply_environment_pauli(psi_phys, i)
            weighted_bra = cp.conj(op_psi)
            raw_row = phi_phys.T @ weighted_bra
            local_expectation = cp.real(cp.vdot(psi_phys, op_psi))
            self.C_correlators[t_idx, i, :] = raw_row.get().astype(np.complex64)
            self.C_connected_correlators[t_idx, i, :] = (
                raw_row - local_expectation * self.E_z_initial
            ).get().astype(np.complex64)

        for s in range(self.nqubitsS):
            bit_pos = self.nqubits - 1 - s
            signs_s = (1 - 2 * ((indices >> bit_pos) & 1)).astype(self.dtypef)
            weighted_bra = signs_s * cp.conj(psi_phys)
            self.C_SE_correlators[t_idx, s, :] = (phi_phys.T @ weighted_bra).get().astype(np.complex64)

    def plot_correlators(self):
        """Plot the ordered two-time correlator C_ij(t) = <sigma_i(t) sigma_j(0)>."""
        Tplot = self._sample_times()
        final_t = Tplot[-1]
        C_abs = getattr(self, "C_correlators_abs_runAv", np.abs(self.C_correlators))

        # Time-averaged |C_ij|
        C_avg = np.nanmean(C_abs, axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        fig.suptitle("E-E ordered two-time correlators: time average and sample traces")
        im = axes[0].imshow(C_avg, cmap='viridis', origin='lower')
        axes[0].set_title(f'Time average: mean_t |C_ij(t)|, N={self.nqubits}')
        axes[0].set_xlabel('j (env qubit)')
        axes[0].set_ylabel('i (env qubit)')
        plt.colorbar(im, ax=axes[0])

        # |C_ij| vs time for diagonal and first off-diagonal
        first_source = self.correlator_sources[0]
        axes[1].plot(Tplot, C_abs[:, first_source, 0], label=f'C_{first_source},{first_source}')
        if first_source + 1 < self.nqubitsE:
            axes[1].plot(Tplot, C_abs[:, first_source + 1, 0], label=f'C_{first_source + 1},{first_source}')
        if len(self.correlator_sources) == self.nqubitsE:
            axes[1].plot(
                Tplot,
                np.nanmean([C_abs[:, i, i] for i in range(self.nqubitsE)], axis=0),
                label='diag avg',
                linestyle='--',
            )
        axes[1].set_xlabel('T')
        axes[1].set_ylabel('|C_ij|')
        axes[1].set_title('Representative |C_ij(t)| traces')
        axes[1].legend()
        fig.text(
            0.5,
            0.02,
            "C_ij(t) = <psi(t)| sigma_z on E_i |phi_j(t)>, with |phi_j(0)> = sigma_z on E_j |psi(0)>. "
            "For averaged runs, magnitudes are averaged over runs before plotting. Left panel averages over every printed time sample; "
            "it is not the same quantity as the final-time heatmap.",
            ha="center",
            va="bottom",
            fontsize=9,
            wrap=True,
        )

        fig.tight_layout(rect=[0, 0.1, 1, 0.94])
        thetastr = self.params["Mironowicz_theta"]
        title = f'Correlators_N={self.nqubits}_theta={thetastr:.2f}'
        plt.savefig(f'.//figs//{title}.jpg', dpi=300)
        plt.show()

        # Snapshot heatmap at last print step
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        fig.suptitle("E-E ordered two-time correlator at final sampled time")
        im2 = ax.imshow(C_abs[-1], cmap='viridis', origin='lower')
        ax.set_title(f'Final sample only: |C_ij(T)| at T={final_t:.1f}')
        ax.set_xlabel('j (env qubit)')
        ax.set_ylabel('i (env qubit)')
        plt.colorbar(im2, ax=ax)
        fig.text(
            0.5,
            0.02,
            "Rows are measured environment qubits E_i at the final sampled time; columns are the initial sigma_z perturbation E_j. "
            "This snapshot can differ strongly from the previous time-averaged heatmap when correlations move or oscillate.",
            ha="center",
            va="bottom",
            fontsize=9,
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.13, 1, 0.93])
        plt.show()

        # Connected correlator: subtract <sigma_i(t)><sigma_j(0)>.
        C_connected_abs = getattr(
            self,
            "C_connected_correlators_abs_runAv",
            np.abs(self.C_connected_correlators),
        )
        C_connected_avg = np.nanmean(C_connected_abs, axis=0)

        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        fig.suptitle("Connected E-E ordered two-time correlator averaged over sampled times")
        im3 = ax.imshow(C_connected_avg, cmap='magma', origin='lower')
        ax.set_title('Time average: mean_t |C_ij(t) - <Z_i(t)><Z_j(0)>|')
        ax.set_xlabel('j (initial env qubit)')
        ax.set_ylabel('i (measured env qubit)')
        plt.colorbar(im3, ax=ax)
        fig.text(
            0.5,
            0.02,
            "This averages the magnitude of the connected E-E correlator over every printed time sample. "
            "It highlights persistent or repeatedly recurring generated correlations, while the final-time snapshot only shows the last sample.",
            ha="center",
            va="bottom",
            fontsize=9,
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.14, 1, 0.93])
        plt.show()

        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        fig.suptitle("Connected E-E ordered two-time correlator at final sampled time")
        im4 = ax.imshow(C_connected_abs[-1], cmap='magma', origin='lower')
        ax.set_title(f'Final sample only: |C_ij(T) - <Z_i(T)><Z_j(0)>| at T={final_t:.1f}')
        ax.set_xlabel('j (initial env qubit)')
        ax.set_ylabel('i (measured env qubit)')
        plt.colorbar(im4, ax=ax)
        fig.text(
            0.5,
            0.02,
            "This subtracts the disconnected product of one-point functions from the raw E-E correlator. "
            "It is a better indicator of generated/spread correlations than the raw heatmap.",
            ha="center",
            va="bottom",
            fontsize=9,
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.13, 1, 0.93])
        plt.show()

        # S-E correlators: |C_SE[s, j]| vs time for each system qubit s
        C_SE_abs = getattr(self, "C_SE_correlators_abs_runAv", np.abs(self.C_SE_correlators))
        for s in range(self.nqubitsS):
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            fig.suptitle(f"S-E ordered two-time correlators for system qubit {s}")
            for column, source in enumerate(self.correlator_sources[:6]):
                ax.plot(Tplot, C_SE_abs[:, s, column], label=f'C_S{s},E{source}')
            source_values = C_SE_abs[:, s, :]
            source_count = np.sum(np.isfinite(source_values), axis=1)
            source_mean = np.full(source_values.shape[0], np.nan)
            source_mean[source_count > 0] = (
                np.nansum(source_values[source_count > 0], axis=1) / source_count[source_count > 0]
            )
            ax.plot(Tplot, source_mean, 'k--', label='source avg')
            ax.set_xlabel('T')
            ax.set_ylabel('|C_SE|')
            ax.set_title(f'|<psi(t)| sigma_z on S{s} |phi_j(t)>|, N={self.nqubits}')
            ax.legend(fontsize=7)
            fig.text(
                0.5,
                0.02,
                "Solid lines show the first environment perturbations j; dashed line averages over all environment qubits.",
                ha="center",
                va="bottom",
                fontsize=9,
                wrap=True,
            )
            fig.tight_layout(rect=[0, 0.12, 1, 0.92])
            plt.savefig(f'.//figs//C_SE_s{s}_N={self.nqubits}_theta={thetastr:.2f}.jpg', dpi=300)
            plt.show()

    def plot_SBSq(self, fragsize=1):

        Tplot = self._sample_times()
        Fplot = np.arange(0, self.nqubitsE - fragsize + 1)
        X, Y = np.meshgrid(Tplot, Fplot)

        for data, label, metric in [
            (self.SBS_qre[fragsize],        'QRE',        'Quantum relative entropy'),
            (self.SBS_trace_dist[fragsize],  'Trace dist', 'Trace distance'),
            (self.SBS_fid[fragsize],         'Fidelity',   'Fidelity'),
        ]:
            fig, ax = plt.subplots()
            c = ax.pcolormesh(X, Y, data, cmap='viridis', shading='auto')
            fig.colorbar(c, ax=ax, label=label)
            title = f'SBS {metric} N={self.nqubits} fragmentN={fragsize}'
            plt.title(title)
            ax.set_xlabel("T")
            ax.set_ylabel("Environment qubit fragment")
            plt.show()

            if np.all(np.isnan(data)):
                avg_data = np.full(data.shape[1], np.nan)
            else:
                avg_data = np.nanmean(data, axis=0)

            fig, ax = plt.subplots()
            ax.plot(Tplot, avg_data)
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Average {label} fragmentN={fragsize}")
            plt.show()


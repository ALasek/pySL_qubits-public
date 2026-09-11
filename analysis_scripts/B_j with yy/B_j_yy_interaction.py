"""
B_j(t) with non-separable interaction: epsilon * sigma_y x sigma_y

Full two-body Hamiltonian for S-E_j pair:

  H_j = g_j * |1><1| x A  +  epsilon * sigma_y x sigma_y

where A = (pi/2)(cos(theta) sigma_x + sin(theta) sigma_z).

The sigma_y x sigma_y term couples the S pointer states, so we cannot
use conditional Hamiltonians. Instead we diagonalize the full 4x4 matrix.

B_j(t) = |<chi_0(t)|chi_1(t)>| / (||chi_0|| * ||chi_1||)

where |chi_s(t)> = <s_S|Psi(t)> are the (unnormalized) conditional 
environment states obtained by projecting the evolved state onto S=|0>,|1>.
"""

import pathlib
import numpy as np
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path(__file__).parent.parent.parent / "analysis_plots"
OUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Pauli matrices
# =============================================================================
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


# =============================================================================
# Build 4x4 Hamiltonian
# =============================================================================

def build_H_pair(g_j, epsilon, theta):
    """
    Build the 4x4 Hamiltonian for the S-E_j pair.
    
    H = g_j * |1><1| x A + epsilon * sigma_y x sigma_y
    
    Basis: |00>, |01>, |10>, |11>  (first qubit = S)
    """
    # |1><1| x A
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
    A = np.pi / 2 * (np.cos(theta) * sx + np.sin(theta) * sz)
    H_int = g_j * np.kron(proj1, A)
    
    # sigma_y x sigma_y
    H_yy = epsilon * np.kron(sy, sy)
    
    return H_int + H_yy


# =============================================================================
# Compute B_j(t) numerically
# =============================================================================

def B_j_numerical(t_val, g_j, epsilon, theta, p=0.5):
    """
    Compute B_j at time t by exact 4x4 matrix exponentiation.
    
    Parameters
    ----------
    t_val : float
        Time.
    g_j : float
        S-E coupling.
    epsilon : float
        sigma_y x sigma_y coupling strength.
    theta : float
        Hamiltonian angle.
    p : float
        System initial state: |psi_S> = sqrt(1-p)|0> + sqrt(p)|1>.
    
    Returns
    -------
    B_j : float
        Bhattacharyya coefficient (fidelity between conditional env states).
    """
    H = build_H_pair(g_j, epsilon, theta)
    
    # Initial state: (sqrt(1-p)|0> + sqrt(p)|1>) x |0>
    psi0 = np.array([np.sqrt(1 - p), 0, np.sqrt(p), 0], dtype=complex)
    
    # Evolve
    U = expm(-1j * H * t_val)
    psi_t = U @ psi0
    
    # Project onto S = |0> and S = |1>
    chi_0 = psi_t[:2]   # <0_S|Psi(t)> = (Psi_00, Psi_01)
    chi_1 = psi_t[2:]   # <1_S|Psi(t)> = (Psi_10, Psi_11)
    
    norm_0 = np.linalg.norm(chi_0)
    norm_1 = np.linalg.norm(chi_1)
    
    if norm_0 < 1e-15 or norm_1 < 1e-15:
        return 1.0  # one branch has zero population, B undefined -> set to 1
    
    # B_j = |<chi_0|chi_1>| / (||chi_0|| ||chi_1||)
    overlap = np.vdot(chi_0, chi_1)
    B = np.abs(overlap) / (norm_0 * norm_1)
    
    return B


def B_j_numerical_array(t_arr, g_j, epsilon, theta, p=0.5):
    """Compute B_j over an array of times."""
    return np.array([B_j_numerical(t, g_j, epsilon, theta, p) for t in t_arr])


# =============================================================================
# Also compute S populations to monitor pointer basis mixing
# =============================================================================

def S_populations(t_val, g_j, epsilon, theta, p=0.5):
    """
    Compute P(S=0) and P(S=1) at time t.
    
    If these change from their initial values (1-p, p), the pointer basis
    is being mixed by the sigma_y x sigma_y term.
    """
    H = build_H_pair(g_j, epsilon, theta)
    psi0 = np.array([np.sqrt(1 - p), 0, np.sqrt(p), 0], dtype=complex)
    
    U = expm(-1j * H * t_val)
    psi_t = U @ psi0
    
    P0 = np.abs(psi_t[0])**2 + np.abs(psi_t[1])**2
    P1 = np.abs(psi_t[2])**2 + np.abs(psi_t[3])**2
    
    return P0, P1


# =============================================================================
# Analytical attempt: theta=0 case
# =============================================================================

def analyze_theta0_structure(g_j, epsilon):
    """
    At theta=0, H has a specific structure. Check if it block-diagonalizes.
    
    H|_{theta=0} = [[  0,       0,      0,    -eps  ],
                     [  0,       0,     eps,     0   ],
                     [  0,      eps,     0,   pi*g/2 ],
                     [-eps,      0,   pi*g/2,   0   ]]
    
    Check commutation with sigma_z x sigma_z (parity).
    """
    H = build_H_pair(g_j, epsilon, theta=0.0)
    P = np.kron(sz, sz)
    
    commutator = H @ P - P @ H
    comm_norm = np.linalg.norm(commutator)
    
    print(f"  [H, sigma_z x sigma_z] norm = {comm_norm:.2e}")
    
    # Check sigma_x x I
    Px = np.kron(sx, I2)
    comm_x = H @ Px - Px @ H
    print(f"  [H, sigma_x x I] norm = {np.linalg.norm(comm_x):.2e}")
    
    # The Hamiltonian at theta=0 has all real entries and is symmetric.
    # Its eigenvalues can be found analytically.
    eigenvalues, eigenvectors = eigh(H)
    print(f"  Eigenvalues: {eigenvalues}")
    
    return eigenvalues, eigenvectors


def B_j_analytical_theta0(t, g_j, epsilon, p=0.5):
    """
    Attempt analytical B_j for theta=0.
    
    H = [[  0,       0,      0,    -eps  ],
         [  0,       0,     eps,     0   ],
         [  0,      eps,     0,    G/2   ],
         [-eps,      0,    G/2,     0   ]]
    
    where G = pi * g_j.
    
    This matrix is real symmetric with block structure coupling 
    {|00>,|11>} and {|01>,|10>} via epsilon, with additional 
    coupling |10>-|11> via G/2.
    
    Since the full 4x4 doesn't block-diagonalize simply,
    we diagonalize numerically but can express B_j in terms of 
    the eigenvalues and eigenvectors analytically for specific limits.
    """
    # For now, use numerical diagonalization but express result
    # in terms of spectral decomposition for insight
    H = build_H_pair(g_j, epsilon, theta=0.0)
    eigenvalues, eigvecs = eigh(H)
    
    # Initial state
    psi0 = np.array([np.sqrt(1 - p), 0, np.sqrt(p), 0], dtype=complex)
    
    # Decompose into eigenbasis
    coeffs = eigvecs.T.conj() @ psi0  # c_n = <n|psi0>
    
    # Time evolution: |psi(t)> = sum_n c_n exp(-i E_n t) |n>
    # chi_0(t) = sum_n c_n exp(-i E_n t) (|n>)_{S=0 part}
    # chi_1(t) = sum_n c_n exp(-i E_n t) (|n>)_{S=1 part}
    
    if np.isscalar(t):
        t = np.array([t])
    
    B_vals = np.zeros(len(t))
    
    for idx, t_val in enumerate(t):
        phases = np.exp(-1j * eigenvalues * t_val)
        psi_t = eigvecs @ (coeffs * phases)
        
        chi_0 = psi_t[:2]
        chi_1 = psi_t[2:]
        
        n0 = np.linalg.norm(chi_0)
        n1 = np.linalg.norm(chi_1)
        
        if n0 < 1e-15 or n1 < 1e-15:
            B_vals[idx] = 1.0
        else:
            B_vals[idx] = np.abs(np.vdot(chi_0, chi_1)) / (n0 * n1)
    
    return B_vals


# =============================================================================
# Perturbative analysis: small epsilon
# =============================================================================

def B_j_perturbative_correction(t, g_j, epsilon, theta, p=0.5):
    """
    First-order perturbative correction to B_j in epsilon.
    
    Unperturbed: H0 = g_j |1><1| x A  (separable, known solution)
    Perturbation: V = epsilon * sigma_y x sigma_y
    
    To first order in epsilon, |Psi(t)> = |Psi_0(t)> + epsilon |Psi_1(t)>
    where |Psi_1(t)> = -i int_0^t U0(t-s) V U0(s) |psi0> ds
    
    This gives a first-order correction to B_j that captures the leading 
    effect of the non-separable term.
    
    Returns both the unperturbed B_j and the full (numerical) B_j for comparison.
    """
    # Unperturbed
    H0 = build_H_pair(g_j, 0.0, theta)
    
    # Full
    H_full = build_H_pair(g_j, epsilon, theta)
    
    psi0 = np.array([np.sqrt(1 - p), 0, np.sqrt(p), 0], dtype=complex)
    
    B_unperturbed = np.zeros(len(t))
    B_full = np.zeros(len(t))
    
    for i, t_val in enumerate(t):
        # Unperturbed evolution
        U0 = expm(-1j * H0 * t_val)
        psi_0t = U0 @ psi0
        chi0_0 = psi_0t[:2]
        chi0_1 = psi_0t[2:]
        n0_0 = np.linalg.norm(chi0_0)
        n0_1 = np.linalg.norm(chi0_1)
        if n0_0 > 1e-15 and n0_1 > 1e-15:
            B_unperturbed[i] = np.abs(np.vdot(chi0_0, chi0_1)) / (n0_0 * n0_1)
        else:
            B_unperturbed[i] = 1.0
        
        # Full evolution
        U_full = expm(-1j * H_full * t_val)
        psi_ft = U_full @ psi0
        chi_f0 = psi_ft[:2]
        chi_f1 = psi_ft[2:]
        nf_0 = np.linalg.norm(chi_f0)
        nf_1 = np.linalg.norm(chi_f1)
        if nf_0 > 1e-15 and nf_1 > 1e-15:
            B_full[i] = np.abs(np.vdot(chi_f0, chi_f1)) / (nf_0 * nf_1)
        else:
            B_full[i] = 1.0
    
    return B_unperturbed, B_full


# =============================================================================
# Plotting
# =============================================================================

def plot_B_vs_time_yy(theta=0.0, p=0.5):
    """B_j(t) for several epsilon values."""
    t = np.linspace(0, 5, 500)
    g_j = 1.0
    
    epsilons = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for eps in epsilons:
        B = B_j_numerical_array(t, g_j, eps, theta, p)
        ax1.plot(t, B, label=rf'$\epsilon={eps}$')
    
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$B_j(t)$')
    ax1.set_title(rf'$B_j(t)$ with $\sigma_y\otimes\sigma_y$, '
                  rf'$g={g_j}$, $\theta={theta:.2f}$, $p={p}$')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: S populations (pointer basis mixing)
    for eps in [0.0, 0.3, 1.0, 2.0]:
        P0_arr = []
        for t_val in t:
            P0, P1 = S_populations(t_val, g_j, eps, theta, p)
            P0_arr.append(P0)
        ax2.plot(t, P0_arr, label=rf'$\epsilon={eps}$')
    
    ax2.axhline(y=1-p, color='gray', linestyle='--', alpha=0.5, label=f'initial P(S=0)={1-p}')
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$P(S=0)$')
    ax2.set_title(r'System population (pointer basis mixing)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'yy_B_j_vs_time_theta{theta:.2f}.png', dpi=150)
    plt.close()


def plot_B_heatmap_yy(theta=0.0, p=0.5, t_val=1.0):
    """Heatmap of B_j at fixed time as function of (g_j, epsilon)."""
    g_range = np.linspace(0.01, 3.0, 150)
    eps_range = np.linspace(0.0, 3.0, 150)
    G, E = np.meshgrid(g_range, eps_range)
    
    B = np.zeros_like(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            B[i, j] = B_j_numerical(t_val, G[i, j], E[i, j], theta, p)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(G, E, B, cmap='RdYlBu_r', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax, label=r'$B_j(t)$')
    ax.set_xlabel(r'$g_j$')
    ax.set_ylabel(r'$\epsilon$')
    ax.set_title(rf'$B_j$ at $t={t_val}$, $\theta={theta:.2f}$, $p={p}$'
                 '\n' r'$H = g_j|1\rangle\langle 1|\otimes A + \epsilon\,\sigma_y\otimes\sigma_y$')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'yy_B_j_heatmap_theta{theta:.2f}.png', dpi=150)
    plt.close()


def plot_perturbative_comparison(theta=0.0, p=0.5):
    """Compare unperturbed vs full B_j for several epsilon."""
    t = np.linspace(0, 5, 300)
    g_j = 1.0
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, eps in zip(axes.flat, [0.05, 0.2, 0.5, 1.0]):
        B_unpert, B_full = B_j_perturbative_correction(t, g_j, eps, theta, p)
        
        ax.plot(t, B_unpert, 'b-', label=r'$\epsilon=0$ (unperturbed)')
        ax.plot(t, B_full, 'r-', label=rf'$\epsilon={eps}$ (full)')
        ax.fill_between(t, B_unpert, B_full, alpha=0.2, color='red')
        ax.set_xlabel('t')
        ax.set_ylabel(r'$B_j(t)$')
        ax.set_title(rf'$\epsilon = {eps}$')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(rf'Effect of $\sigma_y\otimes\sigma_y$ perturbation, '
                 rf'$g={g_j}$, $\theta={theta:.2f}$, $p={p}$', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'yy_B_j_perturbative_theta{theta:.2f}.png', dpi=150)
    plt.close()


def plot_pointer_basis_fidelity(theta=0.0, p=0.5):
    """
    Measure how much the sigma_y x sigma_y term disrupts the pointer basis.
    
    Plot: 1 - |<psi_S(0)|rho_S(t)|psi_S(0)>| as a function of epsilon.
    If this grows, the system state is being rotated away from its initial
    superposition, meaning the pointer basis assumption is breaking down.
    """
    t = np.linspace(0, 5, 300)
    g_j = 1.0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for eps in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        decoherence = []
        for t_val in t:
            H = build_H_pair(g_j, eps, theta)
            psi0 = np.array([np.sqrt(1 - p), 0, np.sqrt(p), 0], dtype=complex)
            U = expm(-1j * H * t_val)
            psi_t = U @ psi0
            
            # Reduced density matrix of S: trace over E
            rho_S = np.zeros((2, 2), dtype=complex)
            rho_S[0, 0] = np.abs(psi_t[0])**2 + np.abs(psi_t[1])**2
            rho_S[0, 1] = psi_t[0]*np.conj(psi_t[2]) + psi_t[1]*np.conj(psi_t[3])
            rho_S[1, 0] = np.conj(rho_S[0, 1])
            rho_S[1, 1] = np.abs(psi_t[2])**2 + np.abs(psi_t[3])**2
            
            # Off-diagonal magnitude = coherence in pointer basis
            decoherence.append(np.abs(rho_S[0, 1]))
        
        ax.plot(t, decoherence, label=rf'$\epsilon={eps}$')
    
    ax.axhline(y=np.sqrt(p*(1-p)), color='gray', linestyle='--', alpha=0.5,
               label=f'initial coherence = {np.sqrt(p*(1-p)):.3f}')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$|\rho_S^{01}(t)|$')
    ax.set_title(rf'System coherence in pointer basis, $\theta={theta:.2f}$, $p={p}$'
                 '\n' r'(should decay for QD; $\sigma_y\otimes\sigma_y$ can regenerate it)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'yy_B_j_coherence_theta{theta:.2f}.png', dpi=150)
    plt.close()


# =============================================================================
# Validation: epsilon=0 should match previous analytical formula
# =============================================================================

def validate():
    print("=== Validation: epsilon=0 should match separable formula ===")
    
    theta = 0.3
    g_j = 1.0
    p = 0.5
    t_test = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    
    # Analytical from separable case (h_j=0)
    # B_j = sqrt(1 - sin^2(alpha) * sin^2(Omega*t))
    # with Omega = pi*g/2, sin^2(alpha) = cos^2(theta)  [for h=0]
    Omega = np.pi * g_j / 2
    sin2_alpha = np.cos(theta)**2
    B_separable = np.sqrt(1 - sin2_alpha * np.sin(Omega * t_test)**2)
    
    # Numerical with epsilon=0
    B_numerical = B_j_numerical_array(t_test, g_j, 0.0, theta, p)
    
    # Note: the separable formula B_j = sqrt(1 - cos^2(theta) sin^2(pi g t/2))
    # is for |phi_E> = |0> and does NOT depend on p (as we showed earlier).
    # The 4x4 numerical should agree.
    
    print(f"  theta={theta}, g={g_j}, p={p}, epsilon=0")
    all_ok = True
    for i, t_val in enumerate(t_test):
        err = abs(B_separable[i] - B_numerical[i])
        status = "OK" if err < 1e-10 else "FAIL"
        if err >= 1e-10:
            all_ok = False
        print(f"    t={t_val:.1f}: separable={B_separable[i]:.10f}, "
              f"4x4={B_numerical[i]:.10f}, err={err:.2e} [{status}]")
    
    if all_ok:
        print("  ALL PASSED\n")
    else:
        print("  SOME FAILURES — check formula/convention\n")
    
    # Also check: does B_j depend on p when epsilon=0?
    print("=== Check: B_j independence of p at epsilon=0 ===")
    t_check = 1.0
    for p_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        B = B_j_numerical(t_check, g_j, 0.0, theta, p_val)
        print(f"    p={p_val}: B_j={B:.10f}")
    
    print("\n=== Check: B_j DEPENDS on p at epsilon!=0 ===")
    eps_check = 0.5
    for p_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        B = B_j_numerical(t_check, g_j, eps_check, theta, p_val)
        print(f"    p={p_val}: B_j={B:.10f}")
    
    print()


# =============================================================================
# Structure analysis
# =============================================================================

def analyze_structure():
    """Check symmetries and block structure of H."""
    print("=== Structure analysis at theta=0 ===")
    eigenvalues, eigvecs = analyze_theta0_structure(g_j=1.0, epsilon=0.5)
    
    print(f"\n=== Structure analysis at theta=pi/4 ===")
    H = build_H_pair(1.0, 0.5, theta=np.pi/4)
    P_zz = np.kron(sz, sz)
    comm = H @ P_zz - P_zz @ H
    print(f"  [H, sigma_z x sigma_z] norm = {np.linalg.norm(comm):.2e}")
    
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"  Eigenvalues: {eigenvalues}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    validate()
    analyze_structure()
    
    print("\n--- Generating plots ---")
    
    for theta in [0.0, np.pi/4, np.pi/2]:
        print(f"  theta = {theta:.2f}")
        plot_B_vs_time_yy(theta=theta, p=0.5)
        plot_B_heatmap_yy(theta=theta, p=0.5, t_val=1.0)
        plot_perturbative_comparison(theta=theta, p=0.5)
        plot_pointer_basis_fidelity(theta=theta, p=0.5)
    
    print(f"\nDone. Plots saved to {OUT_DIR}")

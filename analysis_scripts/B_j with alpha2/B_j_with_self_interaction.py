"""
B_j(t) with environment self-interaction h_j * sigma_z^(j).

Analytical formula for |phi_E> = |0> initial environment state:

  B_j(t) = sqrt(1 - sin^2(alpha) * sin^2(Omega_j * t))

where:
  Omega_j = sqrt((pi*g_j/2 * cos(theta))^2 + (pi*g_j/2 * sin(theta) + h_j)^2)
  sin^2(alpha) = (pi*g_j/2 * cos(theta))^2 / Omega_j^2

For general environment initial state |phi> = cos(beta/2)|0> + sin(beta/2)|1>:
  B_j(t) = |<phi| U0^dag(t) U1(t) |phi>|
  where U0 = exp(-i h_j sigma_z t), U1 = exp(-i (g_j*A + h_j*sigma_z) t)

Also computes B_j numerically via explicit 2x2 matrix exponentiation for validation.

Author: [Olek] — QD project
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import expm

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
# ANALYTICAL B_j(t)
# =============================================================================

def B_j_analytical(t, g_j, h_j, theta, beta=0.5):
    """
    Analytical B_j(t) for environment initial state 
    |phi> = cos(beta/2)|0> + sin(beta/2)|1>.
    
    For beta=0 (|phi>=|0>), reduces to:
      B_j = sqrt(1 - sin^2(alpha) * sin^2(Omega*t))
    
    For general beta, computes via explicit overlap of rotated states.
    
    Parameters
    ----------
    t : float or np.ndarray
        Time.
    g_j : float
        S-E coupling strength.
    h_j : float
        Environment self-interaction strength.
    theta : float
        Hamiltonian angle parameter.
    beta : float
        Environment initial state Bloch angle (beta=0 -> |0>).
    """
    # Interaction Hamiltonian components
    ax = np.pi * g_j / 2 * np.cos(theta)   # coefficient of sigma_x in H_1
    az = np.pi * g_j / 2 * np.sin(theta) + h_j  # coefficient of sigma_z in H_1
    
    Omega = np.sqrt(ax**2 + az**2)
    
    if Omega < 1e-15:
        return np.ones_like(t) if isinstance(t, np.ndarray) else 1.0
    
    # Rotation axis of H_1 = Omega * (sin(alpha) sx + cos(alpha) sz)
    sin_alpha = ax / Omega
    cos_alpha = az / Omega
    
    if np.abs(beta) < 1e-15:
        # Special case: |phi> = |0>
        # B_j = sqrt(1 - sin^2(alpha) * sin^2(Omega*t))
        return np.sqrt(1 - sin_alpha**2 * np.sin(Omega * t)**2)
    else:
        # General case: compute overlap explicitly
        # |phi> = cos(beta/2)|0> + sin(beta/2)|1>
        #
        # U0(t)|phi> = exp(-i h_j sz t)|phi>
        #   = cos(beta/2) exp(-i h_j t)|0> + sin(beta/2) exp(+i h_j t)|1>
        #
        # U1(t)|phi> = exp(-i H_1 t)|phi>
        #   where exp(-i Omega t (n.sigma)) = cos(Omega t) I - i sin(Omega t)(n.sigma)
        #
        # <phi|U0^dag U1|phi> computed explicitly:
        
        cb = np.cos(beta / 2)
        sb = np.sin(beta / 2)
        
        cos_Ot = np.cos(Omega * t)
        sin_Ot = np.sin(Omega * t)
        cos_ht = np.cos(h_j * t)
        sin_ht = np.sin(h_j * t)
        
        # U1(t) matrix elements
        # U1_00 = cos(Ot) - i*cos_alpha*sin(Ot)
        # U1_01 = -i*sin_alpha*sin(Ot)
        # U1_10 = -i*sin_alpha*sin(Ot)
        # U1_11 = cos(Ot) + i*cos_alpha*sin(Ot)
        
        U1_00 = cos_Ot - 1j * cos_alpha * sin_Ot
        U1_01 = -1j * sin_alpha * sin_Ot
        U1_10 = -1j * sin_alpha * sin_Ot
        U1_11 = cos_Ot + 1j * cos_alpha * sin_Ot
        
        # U0^dag(t) = exp(+i h_j sz t)
        # U0dag_00 = exp(+i h_j t), U0dag_11 = exp(-i h_j t)
        
        # V = U0^dag @ U1
        # V_00 = exp(+i h t) * U1_00
        # V_01 = exp(+i h t) * U1_01
        # V_10 = exp(-i h t) * U1_10
        # V_11 = exp(-i h t) * U1_11
        
        eip = np.exp(1j * h_j * t)
        eim = np.exp(-1j * h_j * t)
        
        V_00 = eip * U1_00
        V_01 = eip * U1_01
        V_10 = eim * U1_10
        V_11 = eim * U1_11
        
        # <phi|V|phi> = cb^2 * V_00 + cb*sb * V_01 + sb*cb * V_10 + sb^2 * V_11
        overlap = cb**2 * V_00 + cb * sb * (V_01 + V_10) + sb**2 * V_11
        
        return np.abs(overlap)


# =============================================================================
# NUMERICAL B_j(t) via exact matrix exponentiation
# =============================================================================

def B_j_numerical(t_val, g_j, h_j, theta, beta=0.5):
    """
    Compute B_j at a single time t by explicit 2x2 matrix exponentiation.
    For validation of the analytical formula.
    """
    # Environment initial state
    phi = np.array([np.cos(beta / 2), np.sin(beta / 2)], dtype=complex)
    
    # H_0 = h_j * sigma_z (conditional on S=|0>)
    H0 = h_j * sz
    
    # H_1 = g_j * A + h_j * sigma_z (conditional on S=|1>)
    A = np.pi / 2 * (np.cos(theta) * sx + np.sin(theta) * sz)
    H1 = g_j * A + h_j * sz
    
    # Conditional evolved states
    U0 = expm(-1j * H0 * t_val)
    U1 = expm(-1j * H1 * t_val)
    
    phi_0 = U0 @ phi
    phi_1 = U1 @ phi
    
    # B_j = |<phi_0 | phi_1>|
    return np.abs(np.vdot(phi_0, phi_1))


def B_j_numerical_array(t_arr, g_j, h_j, theta, beta=0.0):
    """Vectorized over time array."""
    return np.array([B_j_numerical(t, g_j, h_j, theta, beta) for t in t_arr])


# =============================================================================
# PLOTTING
# =============================================================================

def plot_B_vs_time(theta=0.0, beta=0.0):
    """Plot B_j(t) for several (g_j, h_j) combinations."""
    
    t = np.linspace(0, 5, 500)
    
    cases = [
        (1.0, 0.0,  'g=1.0, h=0.0 (no self-int)'),
        (1.0, 0.5,  'g=1.0, h=0.5'),
        (1.0, 1.0,  'g=1.0, h=1.0'),
        (1.0, 2.0,  'g=1.0, h=2.0'),
        (1.0, 5.0,  'g=1.0, h=5.0'),
        (0.5, 1.0,  'g=0.5, h=1.0'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: analytical B_j(t)
    for g, h, label in cases:
        B_anal = B_j_analytical(t, g, h, theta, beta)
        ax1.plot(t, B_anal, label=label)
    
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$B_j(t)$')
    ax1.set_title(rf'Analytical $B_j(t)$, $\theta={theta:.2f}$, $\beta={beta:.2f}$')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: analytical vs numerical validation
    t_sparse = np.linspace(0, 5, 50)  # fewer points for numerical
    g_test, h_test = 1.0, 1.5
    
    B_anal = B_j_analytical(t, g_test, h_test, theta, beta)
    B_num = B_j_numerical_array(t_sparse, g_test, h_test, theta, beta)
    
    ax2.plot(t, B_anal, 'b-', label='Analytical')
    ax2.plot(t_sparse, B_num, 'ro', markersize=4, label='Numerical (expm)')
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$B_j(t)$')
    ax2.set_title(rf'Validation: $g={g_test}$, $h={h_test}$, $\theta={theta:.2f}$')
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'alpha2_B_j_vs_time_theta{theta:.3f}_beta{beta:.3f}.png', dpi=150)
    plt.close()
    print(f"Saved alpha2_B_j_vs_time_theta{theta:.3f}_beta{beta:.3f}.png")


def plot_B_heatmap(theta=0.0, beta=0.5, t_val=1.0):
    """
    Heatmap of B_j at fixed time t as a function of (g_j, h_j).
    Shows how self-interaction affects distinguishability.
    """
    g_range = np.linspace(0.01, 3.0, 200)
    h_range = np.linspace(-5.0, 5.0, 200)
    G, H = np.meshgrid(g_range, h_range)
    
    B = np.zeros_like(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            B[i, j] = B_j_analytical(t_val, G[i, j], H[i, j], theta, beta)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(G, H, B, cmap='RdYlBu_r', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax, label=r'$B_j(t)$')
    
    # Mark the line where sin^2(alpha) = 0 (no distinguishability)
    # sin^2(alpha) = 0 when cos(theta) = 0, i.e. theta = pi/2
    # Or more generally when the x-component vanishes: pi*g/2*cos(theta) = 0
    # For theta != pi/2, this only happens at g=0.
    
    ax.set_xlabel(r'$g_j$')
    ax.set_ylabel(r'$h_j$')
    ax.set_title(rf'$B_j$ at $t={t_val}$, $\theta={theta:.2f}$, $\beta={beta:.2f}$'
                 '\n' r'(low $B_j$ = good distinguishability)')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'alpha2_B_j_heatmap_theta{theta:.3f}_beta{beta:.3f}.png', dpi=150)
    plt.close()
    print(f"Saved alpha2_B_j_heatmap_theta{theta:.3f}_beta{beta:.3f}.png")


def plot_B_min_vs_h(theta=0.0, beta=0.5):
    """
    Plot the minimum achievable B_j (= sqrt(1 - sin^2(alpha))) as a function 
    of h_j/g_j ratio. This is the "best case" distinguishability.
    
    sin^2(alpha) = (pi*g/2 * cos(theta))^2 / Omega^2
    
    As h_j increases, Omega grows but the numerator stays fixed,
    so sin^2(alpha) -> 0 and B_j_min -> 1 (indistinguishable).
    """
    g_j = 1.0
    h_ratios = np.linspace(-10, 10, 500)
    
    B_min = np.zeros_like(h_ratios)
    Omega_vals = np.zeros_like(h_ratios)
    sin2_alpha = np.zeros_like(h_ratios)
    
    for i, ratio in enumerate(h_ratios):
        h_j = ratio * g_j
        ax_comp = np.pi * g_j / 2 * np.cos(theta)
        az_comp = np.pi * g_j / 2 * np.sin(theta) + h_j
        Omega = np.sqrt(ax_comp**2 + az_comp**2)
        s2a = ax_comp**2 / Omega**2 if Omega > 1e-15 else 0.0
        
        Omega_vals[i] = Omega
        sin2_alpha[i] = s2a
        B_min[i] = np.sqrt(1 - s2a)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    ax1.plot(h_ratios, B_min, 'b-', linewidth=2)
    ax1.set_ylabel(r'$B_j^{\min} = \sqrt{1 - \sin^2\alpha}$')
    ax1.set_title(rf'Minimum $B_j$ (best distinguishability) vs $h_j/g_j$, '
                  rf'$\theta={theta:.2f}$')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    ax2.plot(h_ratios, sin2_alpha, 'r-', linewidth=2)
    ax2.set_xlabel(r'$h_j / g_j$')
    ax2.set_ylabel(r'$\sin^2\alpha$')
    ax2.set_title(r'Effective coupling strength $\sin^2\alpha$')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'alpha2_B_j_min_vs_h_ratio_theta{theta:.3f}_beta{beta:.3f}.png', dpi=150)
    plt.close()
    print(f"Saved alpha2_B_j_min_vs_h_ratio_theta{theta:.3f}_beta{beta:.3f}.png")


def plot_frequency_and_amplitude(theta=0.0, beta=0.5):
    """
    Plot both the oscillation frequency Omega_j and amplitude sin^2(alpha)
    as functions of g_j and h_j. This decomposes the two effects of h_j:
      1. It changes the oscillation frequency
      2. It reduces the oscillation amplitude (makes states less distinguishable)
    """
    g_j = 1.0
    h_range = np.linspace(-5, 5, 300)
    
    ax_comp = np.pi * g_j / 2 * np.cos(theta)
    
    Omega = np.array([
        np.sqrt(ax_comp**2 + (np.pi * g_j / 2 * np.sin(theta) + h)**2)
        for h in h_range
    ])
    
    sin2_alpha = ax_comp**2 / Omega**2
    
    # Reference values at h=0
    Omega_0 = np.pi * g_j / 2
    sin2_alpha_0 = np.cos(theta)**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(h_range, Omega, 'b-', linewidth=2)
    ax1.axhline(y=Omega_0, color='gray', linestyle='--', alpha=0.5, 
                label=rf'$\Omega_0 = \pi g/2 = {Omega_0:.3f}$')
    ax1.set_xlabel(r'$h_j$')
    ax1.set_ylabel(r'$\Omega_j$')
    ax1.set_title(r'Oscillation frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(h_range, sin2_alpha, 'r-', linewidth=2)
    ax2.axhline(y=sin2_alpha_0, color='gray', linestyle='--', alpha=0.5,
                label=rf'$\sin^2\alpha_0 = \cos^2\theta = {sin2_alpha_0:.3f}$')
    ax2.set_xlabel(r'$h_j$')
    ax2.set_ylabel(r'$\sin^2\alpha$')
    ax2.set_title(r'Oscillation amplitude (distinguishability)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(rf'Effect of self-interaction $h_j$ on $B_j$ dynamics '
                 rf'($g_j={g_j}$, $\theta={theta:.2f}$)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'alpha2_B_j_frequency_amplitude_theta{theta:.3f}_beta{beta:.3f}.png', dpi=150)
    plt.close()
    print(f"Saved alpha2_B_j_frequency_amplitude_theta{theta:.3f}_beta{beta:.3f}.png")


# =============================================================================
# VALIDATION
# =============================================================================

def validate():
    """Check analytical vs numerical for several parameter combinations."""
    print("=== Validation: analytical vs numerical ===")
    
    test_cases = [
        # (g, h, theta, beta, t)
        (1.0, 0.0, 0.0, 0.0, 1.0),   # no self-interaction
        (1.0, 1.0, 0.0, 0.0, 1.0),   # with self-interaction
        (1.0, 2.0, 0.5, 0.0, 0.7),   # theta != 0
        (0.5, 1.5, 1.0, 0.0, 2.0),   # different parameters
        (1.0, 1.0, 0.0, 0.6, 1.0),   # nonzero beta (general env state)
        (1.0, 0.0, 0.0, 1.0, 1.5),   # beta = 1, no h
        (2.0, 3.0, 0.3, 0.8, 0.5),   # everything nonzero
    ]
    
    all_passed = True
    for g, h, theta, beta, t in test_cases:
        B_anal = B_j_analytical(t, g, h, theta, beta)
        B_num = B_j_numerical(t, g, h, theta, beta)
        err = abs(B_anal - B_num)
        status = "OK" if err < 1e-10 else "FAIL"
        if err >= 1e-10:
            all_passed = False
        print(f"  g={g}, h={h}, theta={theta:.1f}, beta={beta:.1f}, t={t}: "
              f"anal={B_anal:.10f}, num={B_num:.10f}, err={err:.2e} [{status}]")
    
    if all_passed:
        print("  ALL PASSED\n")
    else:
        print("  SOME FAILURES\n")
    
    return all_passed


# =============================================================================
# h_j at which QD is maximally impaired
# =============================================================================

def find_critical_h(theta=0.0, g_j=1.0):
    """
    Find the value of h_j that minimizes sin^2(alpha), i.e., most impairs QD.
    
    sin^2(alpha) = (pi*g/2*cos(theta))^2 / [(pi*g/2*cos(theta))^2 + (pi*g/2*sin(theta)+h)^2]
    
    This is minimized (sin^2 alpha -> 0) when |pi*g/2*sin(theta) + h| -> infinity,
    i.e., |h| -> infinity.
    
    It is MAXIMIZED when pi*g/2*sin(theta) + h = 0, i.e., h = -pi*g/2*sin(theta).
    At this point sin^2(alpha) = 1 and B_j oscillates fully to 0.
    """
    h_critical = -np.pi * g_j / 2 * np.sin(theta)
    sin2_alpha_max = 1.0 if np.abs(np.cos(theta)) > 1e-15 else 0.0
    
    print(f"For theta={theta:.4f}, g={g_j}:")
    print(f"  h_critical = {h_critical:.6f} (maximizes distinguishability)")
    print(f"  At h_critical: sin^2(alpha) = {sin2_alpha_max}")
    print(f"  At h=0:        sin^2(alpha) = {np.cos(theta)**2:.6f}")
    print(f"  Large |h|:     sin^2(alpha) -> 0 (QD impaired)")
    
    return h_critical


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    beta_val=0.5
    theta_val = np.pi/4  # Single theta for final output
    
    # Validate
    validate()
    
    # Critical h analysis
    print("=== Critical h_j analysis ===")
    for theta in [0.0, 0.3, np.pi/4, np.pi/2]:
        find_critical_h(theta)
        print()
    
    # Generate plots for several theta values
    for theta_val in [0.0, 0.3, np.pi/4, np.pi/2]:
        print(f"\n--- Generating plots for theta = {theta_val:.4f} ---")
        plot_B_vs_time(theta=theta_val, beta=beta_val)
        plot_B_heatmap(theta=theta_val, t_val=1.0, beta=beta_val)
        plot_B_min_vs_h(theta=theta_val, beta=beta_val)
        plot_frequency_and_amplitude(theta=theta_val, beta=beta_val)
    
    print(f"\nDone. All plots saved to {OUT_DIR}")

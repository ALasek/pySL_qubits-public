import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

OUT_DIR = pathlib.Path(__file__).parent.parent.parent / "analysis_plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
sigma_g = 1.0          # coupling disorder strength
N_g     = 100          # Monte Carlo samples over g_j ~ N(0, sigma_g^2)
N_theta = 80           # grid points in theta
N_p     = 80           # grid points in p
t_max   = 18.0          # max time
N_frames = 360         # animation frames

rng = np.random.default_rng(42)
g_samples = rng.normal(0, sigma_g, N_g)   # shape (N_g,)

# ── Grids ──────────────────────────────────────────────────────────────────────
theta_arr = np.linspace(0, np.pi / 2, N_theta)   # theta in [0, pi/2]
p_arr     = np.linspace(0.0, 0.5,    N_p)         # p in [0, 0.5]
THETA, P  = np.meshgrid(theta_arr, p_arr)         # (N_p, N_theta)

# Geometry: Lambda(p, theta) = 1 - (n(theta) dot r(p))^2.
R_X = 2.0 * np.sqrt(P * (1.0 - P))               # biased-state Bloch x component
R_Z = 2.0 * P - 1.0                              # biased-state Bloch z component
N_DOT_R = np.cos(THETA) * R_X + np.sin(THETA) * R_Z
LAMBDA = np.clip(1.0 - N_DOT_R**2, 0.0, 1.0)     # 1 - (n.r)^2

# ── B_j(t) averaged over g_j ──────────────────────────────────────────────────
# B_j(t) = sqrt(1 - Lambda * sin^2(pi*g_j*t/2))
# We Monte-Carlo average the unsquared Bhattacharyya coefficient over g samples.

def compute_B(t):
    """Returns <B_j(t)>_g, shape (N_p, N_theta)."""
    sin2 = np.sin(np.pi * g_samples * t / 2.0) ** 2
    B_samples = np.sqrt(np.clip(1.0 - sin2[:, None, None] * LAMBDA[None, :, :], 0.0, 1.0))
    return np.mean(B_samples, axis=0)

# ── Time array ────────────────────────────────────────────────────────────────
t_arr = np.linspace(0.0, t_max, N_frames)

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7), facecolor="#0d0d0d")
ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

plt.rcParams.update({
    "text.color":  "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

# Initial surface
B0   = compute_B(t_arr[0])
surf = [ax.plot_surface(
    THETA, P, B0,
    cmap=cm.plasma,
    vmin=0.0, vmax=1.0,
    linewidth=0, antialiased=True, alpha=0.92
)]

# Colorbar
mappable = cm.ScalarMappable(cmap=cm.plasma)
mappable.set_clim(0, 1)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label("$\\langle B_j(t) \\rangle_g$", color="white", fontsize=13)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Axis labels & style
ax.set_xlabel("$\\theta$",        color="white", fontsize=13, labelpad=10)
ax.set_ylabel("$p$",              color="white", fontsize=13, labelpad=10)
ax.set_zlabel("$\\langle B_j \\rangle$", color="white", fontsize=13, labelpad=10)
ax.set_xlim(0, np.pi / 2)
ax.set_ylim(0.0, 0.5)
ax.set_zlim(0.0, 1.0)

ax.set_xticks([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
ax.set_xticklabels(["0", "π/6", "π/4", "π/3", "π/2"], color="white", fontsize=9)
ax.tick_params(colors="white")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#333333")
ax.yaxis.pane.set_edgecolor("#333333")
ax.zaxis.pane.set_edgecolor("#333333")
ax.grid(True, color="#333333", linewidth=0.5)

title = ax.set_title(f"$t = {t_arr[0]:.2f}$", color="white", fontsize=15, pad=12)

# The full parallel-alignment curve lies at p >= 1/2; only its first point and
# the antiparallel endpoint lie in the simulated p <= 1/2 range.
blind_theta = np.array([0.0, np.pi / 2.0])
blind_p = np.array([0.5, 0.0])
ax.scatter(
    blind_theta, blind_p, np.ones_like(blind_p),
    color="#00ffcc", s=28, marker="x", label="$\\Lambda=0$ in plotted range",
    zorder=10
)
ax.legend(loc="upper left", fontsize=9,
          facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

# ── Update function ────────────────────────────────────────────────────────────
def update(frame):
    t = t_arr[frame]
    surf[0].remove()
    B = compute_B(t)
    surf[0] = ax.plot_surface(
        THETA, P, B,
        cmap=cm.plasma,
        vmin=0.0, vmax=1.0,
        linewidth=0, antialiased=True, alpha=0.92
    )
    title.set_text(f"$\\langle B_j(t) \\rangle_g$,   $t = {t:.2f}$   "
                   f"($\\sigma_g = {sigma_g}$)")
    return surf[0], title

# ── Animate ───────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    frames=N_frames,
    interval=50,          # ms between frames
    blit=False
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving animation (this may take ~30 s)…")
ani.save(
    OUT_DIR / "lambda_B_animation.gif",
    writer="pillow",
    fps=24,
    dpi=110
)
print("Saved: lambda_B_animation.gif")

plt.show()

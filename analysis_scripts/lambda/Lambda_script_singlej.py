import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.widgets import Slider

OUT_DIR = pathlib.Path(__file__).parent.parent.parent / "analysis_plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
g_j_default = 1.0      # default coupling for single qubit
N_theta     = 80
N_p         = 80
t_max       = 6.0
N_frames    = 120

# ── Grids ──────────────────────────────────────────────────────────────────────
theta_arr = np.linspace(0, np.pi / 2, N_theta)
p_arr     = np.linspace(0.0, 0.5,    N_p)
THETA, P  = np.meshgrid(theta_arr, p_arr)

R_X = 2.0 * np.sqrt(P * (1.0 - P))               # biased-state Bloch x component
R_Z = 2.0 * P - 1.0                              # biased-state Bloch z component
N_DOT_R = np.cos(THETA) * R_X + np.sin(THETA) * R_Z
LAMBDA = np.clip(1.0 - N_DOT_R**2, 0.0, 1.0)     # 1 - (n.r)^2

# ── B_j for a single qubit with fixed g_j ─────────────────────────────────────
# B_j(t) = sqrt(1 - Lambda(p,theta) * sin^2(pi * g_j * t / 2))
# No averaging — exact expression for one qubit

def compute_B_single(t, g_j):
    return np.sqrt(np.clip(1.0 - LAMBDA * np.sin(np.pi * g_j * t / 2.0) ** 2, 0.0, 1.0))

# ── Time array ────────────────────────────────────────────────────────────────
t_arr = np.linspace(0.0, t_max, N_frames)

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8), facecolor="#0d0d0d")

# Leave room at bottom for sliders
ax = fig.add_axes([0.05, 0.22, 0.90, 0.72], projection="3d")
ax.set_facecolor("#0d0d0d")

plt.rcParams.update({
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
})

# Initial surface
B0   = compute_B_single(t_arr[0], g_j_default)
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
cbar.set_label("$B_j(t)$", color="white", fontsize=13)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Axis style
ax.set_xlabel("$\\theta$", color="white", fontsize=13, labelpad=10)
ax.set_ylabel("$p$",       color="white", fontsize=13, labelpad=10)
ax.set_zlabel("$B_j$",     color="white", fontsize=13, labelpad=10)
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

title = ax.set_title(
    f"$B_j(t)$,   $t = {t_arr[0]:.2f}$,   $g_j = {g_j_default:.2f}$",
    color="white", fontsize=14, pad=12
)

# The full parallel-alignment curve lies at p >= 1/2; only its first point and
# the antiparallel endpoint lie in the simulated p <= 1/2 range.
blind_theta = np.array([0.0, np.pi / 2.0])
blind_p = np.array([0.5, 0.0])
ax.scatter(
    blind_theta, blind_p, np.ones_like(blind_p),
    color="#00ffcc", s=28, marker="x",
    label="$\\Lambda=0$ in plotted range",
    zorder=10
)
ax.legend(loc="upper left", fontsize=9,
          facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

# ── Sliders ───────────────────────────────────────────────────────────────────
slider_color = "#2a2a2a"

# g_j slider
ax_gj = fig.add_axes([0.15, 0.11, 0.70, 0.03], facecolor=slider_color)
slider_gj = Slider(
    ax_gj, "$g_j$", 0.1, 3.0,
    valinit=g_j_default, valstep=0.05, color="#ff6b6b"
)
slider_gj.label.set_color("white")
slider_gj.valtext.set_color("white")

# t slider (manual frame control — pauses animation)
ax_t = fig.add_axes([0.15, 0.05, 0.70, 0.03], facecolor=slider_color)
slider_t = Slider(
    ax_t, "$t$", 0.0, t_max,
    valinit=0.0, color="#4ecdc4"
)
slider_t.label.set_color("white")
slider_t.valtext.set_color("white")

# State: track whether animation is running or paused by slider
state = {"paused": False, "frame": 0}

def redraw(t, g_j):
    surf[0].remove()
    B = compute_B_single(t, g_j)
    surf[0] = ax.plot_surface(
        THETA, P, B,
        cmap=cm.plasma,
        vmin=0.0, vmax=1.0,
        linewidth=0, antialiased=True, alpha=0.92
    )
    title.set_text(
        f"$B_j(t)$,   $t = {t:.2f}$,   $g_j = {g_j:.2f}$"
    )
    fig.canvas.draw_idle()

def on_gj_change(val):
    state["paused"] = True
    ani.event_source.stop()
    redraw(slider_t.val, slider_gj.val)

def on_t_change(val):
    state["paused"] = True
    ani.event_source.stop()
    redraw(slider_t.val, slider_gj.val)

slider_gj.on_changed(on_gj_change)
slider_t.on_changed(on_t_change)

# Click on figure to resume animation
def on_click(event):
    if event.inaxes not in [ax_gj, ax_t]:
        state["paused"] = False
        ani.event_source.start()

fig.canvas.mpl_connect("button_press_event", on_click)

# ── Animation update ───────────────────────────────────────────────────────────
def update(frame):
    if state["paused"]:
        return surf[0], title
    t = t_arr[frame]
    slider_t.set_val(t)          # keep t-slider in sync
    state["frame"] = frame
    redraw(t, slider_gj.val)
    return surf[0], title

ani = animation.FuncAnimation(
    fig, update,
    frames=N_frames,
    interval=50,
    blit=False
)

# ── Instructions ──────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "Drag $g_j$ or $t$ sliders to explore  •  Click plot to resume animation",
    ha="center", color="#888888", fontsize=9
)

plt.show()

# ── Optional: also save a GIF with a few g_j values for reference ──────────────
print("\nSaving reference GIF (g_j = 1.0)…")
fig2 = plt.figure(figsize=(10, 7), facecolor="#0d0d0d")
ax2  = fig2.add_subplot(111, projection="3d", facecolor="#0d0d0d")

surf2 = [ax2.plot_surface(THETA, P, compute_B_single(0, 1.0),
                           cmap=cm.plasma, vmin=0, vmax=1,
                           linewidth=0, antialiased=True, alpha=0.92)]

ax2.set_xlabel("$\\theta$", color="white", fontsize=12, labelpad=8)
ax2.set_ylabel("$p$",       color="white", fontsize=12, labelpad=8)
ax2.set_zlabel("$B_j$",     color="white", fontsize=12, labelpad=8)
ax2.set_xlim(0, np.pi/2); ax2.set_ylim(0, 0.5); ax2.set_zlim(0, 1)
ax2.set_xticks([0, np.pi/4, np.pi/2])
ax2.set_xticklabels(["0", "π/4", "π/2"], color="white")
ax2.tick_params(colors="white")
ax2.xaxis.pane.fill = False; ax2.yaxis.pane.fill = False; ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor("#333"); ax2.yaxis.pane.set_edgecolor("#333"); ax2.zaxis.pane.set_edgecolor("#333")
ax2.grid(True, color="#333", linewidth=0.5)
ax2.scatter(blind_theta, blind_p, np.ones_like(blind_p),
            color="#00ffcc", s=28, marker="x")

title2 = ax2.set_title("", color="white", fontsize=13)
cbar2  = fig2.colorbar(cm.ScalarMappable(cmap=cm.plasma), ax=ax2, shrink=0.5, pad=0.1)
cbar2.set_label("$B_j$", color="white"); cbar2.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
fig2.patch.set_facecolor("#0d0d0d")

def update2(frame):
    surf2[0].remove()
    t = t_arr[frame]
    surf2[0] = ax2.plot_surface(THETA, P, compute_B_single(t, 1.0),
                                 cmap=cm.plasma, vmin=0, vmax=1,
                                 linewidth=0, antialiased=True, alpha=0.92)
    title2.set_text(f"$B_j(t)$,  $g_j = 1.0$,  $t = {t:.2f}$")
    return surf2[0], title2

ani2 = animation.FuncAnimation(fig2, update2, frames=N_frames, interval=50, blit=False)
ani2.save(OUT_DIR / "lambda_singlej_B_single_gj1.gif", writer="pillow", fps=24, dpi=110)
print("Saved: lambda_singlej_B_single_gj1.gif")
plt.close(fig2)

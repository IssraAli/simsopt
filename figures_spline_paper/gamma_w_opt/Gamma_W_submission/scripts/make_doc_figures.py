#!/usr/bin/env python3
r"""Generate the two explanatory figures for Gamma_W_doc (physicist-friendly schematics).

Fig 1  anatomy_of_a_well.png   -- |B| along a field line: the trapped well, the turning
                                  field B*, the turning points, and the occupied branch.
                                  Turning points / well come from the ACTUAL engine
                                  (find_wells / integrate_well), so the picture is faithful.
Fig 2  drift_and_detrapping.png -- (left) the (s,alpha) drift plane: J level sets and the
                                  bounce-centre characteristic, with s_dot ~ dJ/dalpha;
                                  (right) a de-trapping event, keep (stop) vs branch_continue
                                  (follow into the descendant well) -- the current default.

Pure numpy/matplotlib + the standalone engine's pure-numpy core (no firm3d). Run:
    python make_doc_figures.py            # writes ../figures/*.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# the engine's pure-numpy core (find the standalone next to this script tree)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Gamma_W_final import find_wells, integrate_well          # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY = "#2c6fbb", "#e08214", "#2ca25f", "#c0392b", "#7f7f7f"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})


# ----------------------------------------------------------------- Fig 1
def fig_anatomy():
    B0 = 1.0
    zeta = np.linspace(-np.pi, np.pi, 1200)
    B = B0 * (1 + 0.55 * (1 - np.cos(zeta)) + 0.05 * (1 - np.cos(3 * zeta)))
    B_star = 1.0 * 1.65                                      # turning field E/mu

    wells = find_wells(zeta, B, B_star)
    (i0, i1, kmin) = min(wells, key=lambda w: abs(zeta[w[2]]))   # occupied = nearest zeta=0
    gII = 1.0
    wi = integrate_well(zeta, B, gII, B_star, i0, i1, kmin)
    zL, zR, zmin = wi.zeta_L, wi.zeta_R, wi.zeta_min

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(zeta, B, color=BLUE, lw=2.2, label=r"$|B|(\zeta)$ along the field line")
    ax.axhline(B_star, color=RED, lw=1.6, ls="--")
    ax.text(np.pi, B_star + 0.02, r"$B_\ast = E/\mu$  (turning field)",
            color=RED, ha="right", va="bottom", fontsize=10)

    inwell = (zeta >= zL) & (zeta <= zR)
    ax.fill_between(zeta[inwell], B[inwell], B_star, color=ORANGE, alpha=0.22)
    ax.text(zmin, B_star - 0.28, "trapped\n(occupied) well", color=ORANGE,
            ha="center", va="center", fontsize=10)

    for zt in (zL, zR):
        ax.plot([zt], [B_star], "o", color=RED, ms=6, zorder=5)
    ax.annotate(r"turning point $\zeta_L$", (zL, B_star), (zL - 0.1, B_star + 0.33),
                color=RED, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.annotate(r"turning point $\zeta_R$", (zR, B_star), (zR + 0.1, B_star + 0.33),
                color=RED, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.plot([zmin], [wi.B_min], "o", color=BLUE, ms=6, zorder=5)
    ax.annotate(r"$B_{\min}$", (zmin, wi.B_min), (zmin, wi.B_min - 0.22),
                color=BLUE, fontsize=10, ha="center")

    # a neighbouring (secondary) well, above B* here -> not occupied
    ax.text(np.pi - 0.15, B[np.argmin(np.abs(zeta - (np.pi - 0.15)))] + 0.05,
            "barrier\n" + r"($B>B_\ast$: passing here)", color=GREY, fontsize=9,
            ha="right", va="bottom")

    ax.set_xlabel(r"toroidal angle along the field line  $\zeta$")
    ax.set_ylabel(r"$|B|$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0.9, 2.35)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.legend(loc="upper center", frameon=False, fontsize=9.5)
    ax.set_title(r"Anatomy of a trapped orbit: the well sets $J_a=2v_0(G{+}\iota I)\!\int_{\rm well}\!"
                 r"\sqrt{1-B/B_\ast}\,/B\,d\zeta$ and $\tau_b$", fontsize=10.5)
    fig.tight_layout()
    p = os.path.join(OUT, "anatomy_of_a_well.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


# ----------------------------------------------------------------- Fig 2
def fig_drift_detrap():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # -- left: (s, alpha) drift plane, J level sets + a characteristic --------
    s = np.linspace(0.05, 1.0, 300)
    al = np.linspace(0, 2*np.pi, 300)
    S, A = np.meshgrid(s, al, indexing="ij")
    # synthetic J with a weak alpha-dependence (the quasisymmetry-breaking residual)
    J = (1.0 - 0.6*S) * (1 + 0.16*np.cos(A))
    cs = axL.contour(A, S, J, levels=12, colors=GREY, linewidths=0.8)
    axL.clabel(cs, inline=True, fontsize=7, fmt="")

    # a bounce-centre trajectory = a level set of J (drift conserves J); launch at
    # alpha=pi (modulation minimum) and drift out toward the edge -> illustrates reach.
    a_traj = np.linspace(np.pi, 0.42*np.pi, 200)
    J0 = (1.0 - 0.6*0.30) * (1 + 0.16*np.cos(np.pi))
    s_traj = 1.0/0.6 * (1 - J0/(1 + 0.16*np.cos(a_traj)))
    axL.plot(a_traj, s_traj, color=BLUE, lw=2.4, zorder=5,
             label=r"bounce-centre drift (a level set of $J$)")
    axL.plot([a_traj[0]], [s_traj[0]], "o", color=GREEN, ms=8, zorder=6)
    axL.text(a_traj[0]+0.12, s_traj[0]-0.03, "launch $s_0$", color=GREEN, ha="left", fontsize=9)
    axL.add_patch(FancyArrowPatch((a_traj[90], s_traj[90]), (a_traj[120], s_traj[120]),
                  arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=2))
    axL.text(1.08*np.pi, 0.30, r"$\dot s = +\dfrac{\kappa}{\tau_b}\dfrac{\partial J}{\partial\alpha}$",
             fontsize=13, color=BLUE)
    axL.text(1.08*np.pi, 0.17, r"$\dot\alpha = -\dfrac{\kappa}{\tau_b}\dfrac{\partial J}{\partial s}$",
             fontsize=13, color=BLUE)
    axL.set_xlabel(r"field-line label  $\alpha=\theta-\iota\zeta$")
    axL.set_ylabel(r"normalized flux  $s$  (0 axis, 1 edge)")
    axL.set_title(r"Drift in the $(s,\alpha)$ plane: reach $R_W=\frac{\max s - s_0}{1-s_0}$",
                  fontsize=10.5)
    axL.set_xlim(0, 2*np.pi); axL.set_ylim(0.05, 1.0)
    axL.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axL.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axL.axhline(1.0, color=RED, lw=1.2, ls=":")
    axL.text(2*np.pi-0.1, 0.965, "edge (lost)", color=RED, ha="right", fontsize=9)
    axL.legend(loc="lower right", frameon=False, fontsize=8.5)

    # -- right: de-trapping, keep vs branch_continue --------------------------
    z = np.linspace(-np.pi, np.pi, 800)
    B_prim = 1 + 0.5*(1 - np.cos(z))                         # primary well (shrinking)
    ripple = 0.06*(1 - np.cos(6*z))                          # secondary ripple wells
    B_now = 1 + 0.13*(1 - np.cos(z)) + ripple               # primary nearly gone
    Bs = 1.14
    axR.plot(z, B_prim, color=GREY, lw=1.4, ls="--", label="earlier: deep primary well")
    axR.plot(z, B_now, color=BLUE, lw=2.2, label="now: primary de-trapped, ripple wells remain")
    axR.axhline(Bs, color=RED, lw=1.4, ls="--")
    axR.text(np.pi, Bs+0.008, r"$B_\ast$", color=RED, ha="right", va="bottom", fontsize=10)

    # descendant ripple well nearest zeta~0
    inr = (B_now < Bs)
    axR.fill_between(z[inr], B_now[inr], Bs, color=ORANGE, alpha=0.22)
    axR.annotate("keep: STOP\n(unresolved, in band)", (0.0, 1.02), (-2.9, 1.17),
                 color=RED, fontsize=9, ha="left",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axR.annotate("branch\\_continue: FOLLOW\ninto descendant well",
                 (0.52, B_now[np.argmin(np.abs(z-0.52))]), (1.1, 1.19),
                 color=GREEN, fontsize=9, ha="left",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    axR.set_xlabel(r"$\zeta$"); axR.set_ylabel(r"$|B|$")
    axR.set_xlim(-np.pi, np.pi); axR.set_ylim(0.95, 1.32)
    axR.set_xticks([-np.pi, 0, np.pi]); axR.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    axR.set_title("De-trapping: the current default follows the orbit\n(symmetry-agnostic)",
                  fontsize=10.5)
    axR.legend(loc="upper center", frameon=False, fontsize=8)

    fig.tight_layout()
    p = os.path.join(OUT, "drift_and_detrapping.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_anatomy()
    fig_drift_detrap()

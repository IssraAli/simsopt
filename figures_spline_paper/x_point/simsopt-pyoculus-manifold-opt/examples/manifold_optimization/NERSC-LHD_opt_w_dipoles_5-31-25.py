#!/usr/bin/env python

# Standard library imports
import os
import time
import logging
import json
import random
from datetime import datetime

# Third-party imports
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch jobs
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import ceil, sqrt

# Simsopt imports
import simsoptpp as sopp
from simsopt.field import (
    BiotSavart, Coil, Current, InterpolatedField, SurfaceClassifier, DipoleField,
    particles_to_vtk, LevelsetStoppingCriterion, load_coils_from_makegrid_file,
    MinRStoppingCriterion, MaxRStoppingCriterion, MinZStoppingCriterion, MaxZStoppingCriterion
)
from simsopt.geo import (
    SurfaceRZFourier, Curve, RotatedCurve, CurveXYZFourierSymmetries, CurveXYZFourier,
    plot, curves_to_vtk, create_equally_spaced_curves,
    LinkingNumber, CurveLength, CurveCurveDistance, MeanSquaredCurvature,
    LpCurveCurvature, CurveSurfaceDistance
)
from simsopt.util import proc0_print, comm_world
from simsopt._core.util import parallel_loop_bounds
from simsopt import load
from simsopt.mhd import field_topology_optimizables
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve

# Pyoculus imports
from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint

datadir = os.environ.get('MANIFOLD_OPT_DATA', os.path.dirname(os.path.abspath(__file__)))

# Setup output directory and logging
myString = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = os.path.join(os.environ.get('SCRATCH', datadir), f'output_{myString}')
# OUTPUT_DIR = os.path.join('/home/telder/pyoculus/examples/simsopt_lhd_like/', f'output_{myString}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'run.log'), level=logging.INFO)

# Plot settings
plt.rc('font', size=15)
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('figure', titlesize=20)

def get_fp_type(res):
    """Determine fixed point type based on residue."""
    if res > 1:  # Alternating hyperbolic fixed point
        return "s"
    elif res < 0:  # Hyperbolic fixed point
        return "x"
    elif res < 1:  # Elliptic fixed point
        return "o"

def compute_fieldlines(
    field,
    xyz_inits,
    tmax=200,
    tol=1e-7,
    phis=[],
    stopping_criteria=[],
    comm=None
):
    nlines = xyz_inits.shape[0]
    res_tys = []
    res_phi_hits = []
    first, last = parallel_loop_bounds(comm, nlines)
    for i in range(first, last):
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field,
            xyz_inits[i, :],
            tmax,
            tol,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0] / len(res_ty)
        logging.info(f"{i+1:3d}/{nlines}, t_final={res_ty[-1][0]}, average timestep {dtavg:.10f}s")
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    return res_tys, res_phi_hits

def plot_poincare_data(
    fieldlines_phi_hits,
    phis,
    filename,
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=None,
    ylims=None,
    s=2,
    marker="o"
):
    plt.rc("font", size=7)
    nrowcol = 2
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for i in range(4):
        row = i // nrowcol
        col = i % nrowcol
        for surf in [surf_classifier]:
            cross_section = surf.cross_section(phi=phis[i])
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            g = 0.0
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c=(g, g, g))
        axs[row, col].grid(True, linewidth=0.5)
        if i != 4 - 1:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc="left", y=0.0)
        else:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc="right", y=0.0)
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$R$")
        if col == 0:
            axs[row, col].set_ylabel("$Z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            if fieldlines_phi_hits[j].size == 0:
                continue
            if mark_lost:
                lost = fieldlines_phi_hits[j][-1, 1] < 0
                color = "r" if lost else "g"
            phi_hit_codes = fieldlines_phi_hits[j][:, 1]
            condition = np.logical_and(phi_hit_codes >= 0, np.mod(phi_hit_codes, 4) == i)
            indices = np.where(condition)[0]
            data_this_phi = fieldlines_phi_hits[j][indices, :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
            color = colors[j % len(colors)]
            axs[row, col].scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)
            new_row = row
            if i == 1 or i == 3:
                new_row = 1 - row
            axs[new_row, col].scatter(r, -data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)
    plt.figtext(0.5, 0.005, bottom_str, ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=dpi)
    plt.close()
    return fig, axs

def plot_poincare_data_1(
    fieldlines_phi_hits,
    phi,
    filename,
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=None,
    ylims=None,
    s=2,
    marker="o"
):
    fig, ax = plt.subplots()
    for surf in []:
        cross_section = surf.cross_section(phi=phi)
        r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
        z_interp = cross_section[:, 2]
        g = 0.0
        ax.plot(r_interp, z_interp, linewidth=1, c=(g, g, g))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax.grid(True, linewidth=0.5)
    ax.set_title(f"$\\phi = {phi/np.pi:.3f}\\pi$ ", loc="right", y=0.0)
    ax.set_xlabel("$R$")
    ax.set_ylabel("$Z$")
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    for j in range(len(fieldlines_phi_hits)):
        if fieldlines_phi_hits[j].size == 0:
            continue
        if mark_lost:
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            color = "r" if lost else "g"
        phi_hit_codes = fieldlines_phi_hits[j][:, 1]
        condition = np.logical_and(phi_hit_codes >= 0, np.mod(phi_hit_codes, 4) == 2)
        indices = np.where(condition)[0]
        data_this_phi = fieldlines_phi_hits[j][indices, :]
        if data_this_phi.size == 0:
            continue
        r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
        color = colors[j % len(colors)]
        ax.scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=dpi)
    # plt.close()
    return fig, ax

def trace_fieldlines(bfield, label, nfp):
    t1 = time.time()
    phis = [(i / 4) * (2 * np.pi / nfp) for i in range(4 * nfp)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        initial_conditions,
        tmax=tmax_fl,
        tol=tol,
        comm=comm_world,
        phis=phis,
        stopping_criteria=stopping_criteria
    )
    t2 = time.time()
    logging.info(
        f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}"
    )
    if comm_world is None or comm_world.rank == 0:
        particles_to_vtk(fieldlines_tys, os.path.join(OUTPUT_DIR, f'fieldlines_{label}'))
        fig, axs = plot_poincare_data(
            fieldlines_phi_hits,
            phis,
            f"poincare_{label}.png",
            dpi=300,
            s=1
        )
    return fig, axs

def plot_dipole_field(
    dipole_field,
    fig=None,
    ax=None,
    show=False,
    arrow_scale=1.0,
    point_kwargs={},
    arrow_kwargs={}
):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    x = dipole_field.dipole_grid[:, 0]
    y = dipole_field.dipole_grid[:, 1]
    z = dipole_field.dipole_grid[:, 2]
    mx = dipole_field.m_vec[:, 0] * arrow_scale
    my = dipole_field.m_vec[:, 1] * arrow_scale
    mz = dipole_field.m_vec[:, 2] * arrow_scale
    ax.scatter(x, y, z, **point_kwargs)
    ax.quiver(x, y, z, mx, my, mz, **arrow_kwargs)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dipole_field.png'))
    plt.close()
    return fig, ax

def generate_dipole_grid(
    surface,
    ntheta=None,
    nphi=None,
    moment_magnitude=0.0,
    scale_by_area=False,
    extent="full torus"
):
    ntheta = surface.ntheta if ntheta is None else ntheta
    nphi = surface.nphi if nphi is None else nphi
    new_surface = SurfaceRZFourier.from_nphi_ntheta(
        nphi=nphi,
        ntheta=ntheta,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        range=extent
    )
    new_surface.set_dofs(surface.get_dofs())
    points = new_surface.gamma().reshape(-1, 3)
    normals = new_surface.normal().reshape(-1, 3)
    n_dipoles = nphi * ntheta
    positions = points.reshape(n_dipoles, 3)
    normals = normals.reshape(n_dipoles, 3)
    if scale_by_area:
        dgamma_dphi = new_surface.dgamma_by_dphi().reshape(n_dipoles, 3)
        dgamma_dtheta = new_surface.dgamma_by_dtheta().reshape(n_dipoles, 3)
        area_elements = np.linalg.norm(np.cross(dgamma_dphi, dgamma_dtheta), axis=1)
        moments = moment_magnitude * area_elements[:, np.newaxis] * normals
    else:
        moments = moment_magnitude * normals
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    mx = moments[:, 0]
    my = moments[:, 1]
    mz = moments[:, 2]
    return DipoleField(np.vstack([x, y, z]).T, np.vstack([mx, my, mz]).T, stellsym=surface.stellsym, nfp=surface.nfp)

def generate_circular_coils_on_surface(surf, r, nquad=100):
    nphi = len(surf.quadpoints_phi)
    ntheta = len(surf.quadpoints_theta)
    gamma = surf.gamma()
    normals = surf.unitnormal()
    dgamma_dphi = surf.gammadash1()
    coils = []
    for i in range(nphi):
        for j in range(ntheta):
            R = gamma[i, j]
            N = normals[i, j]
            T_phi = dgamma_dphi[i, j]
            U = T_phi / np.linalg.norm(T_phi)
            V = np.cross(N, U)
            dofs = [
                R[0], r*U[0], r*V[0],
                R[1], r*U[1], r*V[1],
                R[2], r*U[2], r*V[2]
            ]
            curve = CurveXYZFourier(nquad, order=1)
            curve.set_dofs(dofs)
            curve.fix_all()
            current = Current(0.0e4)
            coil = Coil(curve, current)
            coils.append(coil)
    return coils

from numpy import array as array
# Configuration
CONFIG = {
    'R_major': 3.63,
    'r_minor': 0.975,
    'nfp': 10,
    'prob_tol': 1e-10,
    'FP_tol': 1e-10,
    'int_tol': 1e-15,
    'rtol': 1e-10,
    'nfieldlines': 5,
    'tmax_fl': 1,
    'test_dofs': 0,
    'loadFixedPoints': True,
    'show_loaded_fixed_points': True,
    'root_dir': os.path.join(datadir, 'FixedPoint_lists/'),
    'filename_for_fixed_points': 'LHD_fixed_points_change_dofs_0-minM_2-maxM_20.txt',
    'justTrace': 0,
    'opt': 1,
    'axis_guess': [3.568, 0],
    'initial_phi': 0.1 * np.pi,
    'degree': 4,
    'interpolant_n': 10,
    'classifier_aminor': 1.5,
    'classifier_elongation': 1.5,
    'bracketL': [2.6, 2.72],
    'bracketR': [4.63, 5.0],
    'OneD_finder': False
}
    # 'root_dir': '/home/telder/pyoculus/examples/simsopt_lhd_like/FixedPoint_lists/',

# Load fixed points
if CONFIG['loadFixedPoints']:
    with open(CONFIG['root_dir'] + CONFIG['filename_for_fixed_points'], 'r') as f:
        tuples = eval(f.read())
    FP_ms, FP_guesses, FP_residues, FP_sides, FP_dims = [], [], [], [], []
    GR_cutoff = 1e1
    for tuple_ in tuples:
        (m, guess, residue, side, dimension) = tuple_
        if m > 5 or m == 2:
            continue
        FP_ms.append(m)
        FP_guesses.append(guess)
        FP_residues.append(residue)
        FP_sides.append(side)
        FP_dims.append(dimension)
    logging.info(f'Targeting {FP_ms} {FP_guesses}')

# Finder arguments
if CONFIG['OneD_finder']:
    finder_args_L_list = [
        {"method": "scipy.1D", "bracket": CONFIG['bracketL'], "bracket_infinity": +1, "xtol": 1e-10, "scipy_method": "brentq"}
        for m, guess in zip(FP_ms_L, FP_guesses_L)
    ]
    finder_args_R_list = [
        {"method": "scipy.1D", "bracket": CONFIG['bracketR'], "bracket_infinity": +1, "xtol": 1e-10, "scipy_method": "brentq"}
        for m, guess in zip(FP_ms_R, FP_guesses_R)
    ]
else:
    factor = 1e-10
    xtol = 1e-15
    finder_args_list = [
        {"method": "scipy.root", "options": {"factor": factor, "xtol": xtol}}
        for m, guess in zip(FP_ms, FP_guesses)
    ]
integration_args = {"nsteps": 10**5, "rtol": CONFIG['rtol']}
axis_finder_args = {
    "method": "scipy.1D",
    "bracket": [3.0, 4.0],
    "bracket_infinity": +1,
    "xtol": 1e-10,
    "scipy_method": "brentq"
}

# Load LHD-like coils
order = 8
coils_hc = load(os.path.join(datadir, 'LHD_coilset', f'LHD_HCs_order_{order}.json'))
# coils_hc = load('/home/telder/pyoculus/examples/simsopt_lhd_like/LHD_coilset/' + f'LHD_HCs_order_{order}.json')
if CONFIG['test_dofs']:
    new_dofs = [3.8499743700893947, 0.9987961344685833, 0.049694380159080516, 0.0012728589050715283, 1.2653963459632173e-05, 0.00034827787743771344, -0.00021666333258971892, 8.101962566116475e-05, 4.3639258430786706e-05, 0.0002859247777239665, 4.4489351621177704e-05, -2.1651509221672777e-06, -0.0002518915719730465, 0.0004203148323810962, -5.676846454102374e-05, 4.676253665361815e-05, -4.6863950021309005e-05, -0.9965607320605306, -0.04977824859255296, -0.0008851647911581579, 2.7778873306964658e-05, -0.0003928882372798328, 0.00022528798432989332, 4.1938365446212256e-05, 6.019290886319615e-05, 3.8501666698588872, -0.9986278782508725, 0.0499421937368488, -0.0015022592219236916, 0.0003142282382486341, -0.00022372919878639727, 2.3355361689491158e-05, 7.647987737316935e-05, -6.448019228932428e-05, -8.698645649930155e-05, -0.0002833030285815724, 0.00024129467963971392, -0.0003495315537841296, 1.3071782520658959e-05, 0.00011589244560994964, -0.00010009304136499923, -8.207498061271198e-05, 0.9957171293088645, -0.049926325050629404, 0.0016376214613644713, -0.00048478710751876796, 3.492321161654002e-05, -8.241917543129414e-05, -0.0001653706229826647, -0.00010967482604539258]
    npts = 1000
    qp = np.linspace(0, 1, npts, endpoint=False)
    o = order
    nfp = 5
    SS = True
    base_curve = CurveXYZFourierSymmetries(quadpoints=qp, order=o, nfp=nfp, stellsym=SS, ntor=1)
    rota_curve = CurveXYZFourierSymmetries(quadpoints=qp, order=o, nfp=nfp, stellsym=SS, ntor=1)
    base_curve.set_dofs(new_dofs[0:25])
    rota_curve.set_dofs(new_dofs[25:])
    coils_hc = [Coil(base_curve, coils_hc[0].current), Coil(rota_curve, coils_hc[1].current)]

coils_vf = load(os.path.join(datadir, 'LHD_coilset', 'LHD_VCs.json'))
# coils_vf = load('/home/telder/pyoculus/examples/simsopt_lhd_like/LHD_coilset/' + f'LHD_VCs.json')
for coil in coils_vf:
    coil.fix_all()
for coil in coils_hc:
    coil.curve.fix_all()

# Dipole coilset
MakeDipoleCoilset = True
if MakeDipoleCoilset:
    r = 0.15
    nphi = 2
    ntheta = 2
    winding = SurfaceRZFourier.from_nphi_ntheta(
        mpol=1, ntor=1, nfp=CONFIG['nfp'], nphi=nphi, ntheta=ntheta, range="half period"
    )
    classifier_elongation_inv = 1 / CONFIG['classifier_elongation']
    b = 1.35
    classifier_elongation_inv = 0.4
    winding.set_rc(0, 0, CONFIG['R_major'])
    winding.set_rc(1, 0, 0.5 * b * (classifier_elongation_inv + 1))
    winding.set_zs(1, 0, -0.5 * b * (classifier_elongation_inv + 1))
    winding.set_rc(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
    winding.set_zs(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
    dips = generate_circular_coils_on_surface(winding, r=r, nquad=10)
    from simsopt.field import coils_via_symmetries
    dips_ss = coils_via_symmetries([dip.curve for dip in dips], [dip.current for dip in dips], winding.nfp, winding.stellsym)

if 1:
    dipole_field = generate_dipole_grid(winding, ntheta=2, nphi=2, extent="half period")

coils = coils_hc + coils_vf + dips_ss
bs = BiotSavart(coils)
logging.info('Running . . .')

# Field line tracing setup
tol = CONFIG['prob_tol']
degree = CONFIG['degree']
interpolant_n = CONFIG['interpolant_n']
nfieldlines = CONFIG['nfieldlines']
tmax_fl = CONFIG['tmax_fl']
Rs = np.linspace(4.0, 4.8, nfieldlines)
Zs = 0 * Rs
initial_conditions = np.array([
    [r * np.cos(CONFIG['initial_phi']), r * np.sin(CONFIG['initial_phi']), z]
    for r, z in zip(Rs, Zs)
])

bottom_str = (
    f"tol:{tol} interpolant_n:{interpolant_n} tmax:{tmax_fl} nfieldlines:{nfieldlines} degree:{degree}"
)

surf_classifier = SurfaceRZFourier.from_nphi_ntheta(
    mpol=1, ntor=1, nfp=CONFIG['nfp'], nphi=100, ntheta=30, range="full torus"
)
classifier_elongation_inv = 1 / CONFIG['classifier_elongation']
b = CONFIG['classifier_aminor'] / np.sqrt(classifier_elongation_inv)
surf_classifier.set_rc(0, 0, CONFIG['R_major'] + 0.25 * CONFIG['r_minor'])
surf_classifier.set_rc(1, 0, 0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_zs(1, 0, -0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_rc(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.set_zs(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
XYZ = surf_classifier.gamma().reshape(-1, 3)
R = (XYZ[:, 0]**2 + XYZ[:, 1]**2)**0.5
Zmax = np.max(np.abs(XYZ[:, 2]))
Rmax = np.max(R)
Rmin = np.min(np.abs(R))
sc_fieldline = SurfaceClassifier(surf_classifier, h=0.2, p=2)
stopping_criteria = [
    MaxZStoppingCriterion(Zmax), MinZStoppingCriterion(-Zmax),
    MaxRStoppingCriterion(Rmax), MinRStoppingCriterion(Rmin),
    LevelsetStoppingCriterion(sc_fieldline.dist)
]

label = "bsh_" + datetime.now().strftime("%Y%m%d-%H%M")
phis = [(i / 4) * (2 * np.pi / CONFIG['nfp']) for i in range(4 * CONFIG['nfp'])]
t1 = time.time()
fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
    bs,
    initial_conditions,
    tmax=tmax_fl,
    tol=tol,
    phis=phis,
    stopping_criteria=stopping_criteria,
    comm=comm_world
)
t2 = time.time()
logging.info(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}")
particles_to_vtk(fieldlines_tys, os.path.join(OUTPUT_DIR, f'fieldlines_{myString}'))

fig, ax = plot_poincare_data_1(
    fieldlines_phi_hits,
    CONFIG['initial_phi'],
    'poincare_initial.png',
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=[2.5, 5],
    ylims=[-1.0, 1.0],
    s=10,
    marker="o"
)
if CONFIG['show_loaded_fixed_points']:
    for m, guess, res in zip(FP_ms, FP_guesses, FP_residues):
        o = 0.05
        plt.annotate((r'$%d$' % m), xy=[guess[0], guess[1] - o], rotation=0, fontsize=10,color='r')
        plt.annotate((r'$%2.1le$' % res), xy=[guess[0], guess[1] + o], rotation=90, fontsize=10, color='b')
        ax.plot(
            guess[0], guess[1], ls='', marker=get_fp_type(res), markersize=10,
            color='g', markeredgecolor='k' if get_fp_type(res) != 'x' else None, mew=3
        )
    plt.title('Poincare with initial fixed points')
plt.savefig(os.path.join(OUTPUT_DIR, 'poincare_initial_with_fp.png'))
plt.close()

if CONFIG['justTrace']:
    logging.info("Field line tracing completed.")
else:
    # Fixed point search
    logging.info("Setting up the problem")
    simsoptfield = SimsoptBfield(CONFIG['nfp'], bs)
    pyoproblem = CylindricalBfieldSection.without_axis(
        simsoptfield,
        phi0=CONFIG['initial_phi'],
        guess=CONFIG['axis_guess'],
        rtol=CONFIG['prob_tol'],
        finderargs=axis_finder_args
    )
    logging.info("Searching fixed points")
    fps = []
    for guess, m, finder_args in zip(FP_guesses, FP_ms, finder_args_list):
        logging.info(f'Looking for m={m} at point {guess} . . . ')
        fp = field_topology_optimizables.PyOculusFixedPoint(
            bs,
            guess,
            CONFIG['nfp'],
            pyoproblem.phi0,
            m,
            None,
            finder_tol=CONFIG['FP_tol'],
            finder_args=finder_args,
            integration_args=integration_args
        )
        if fp._fixed_point.successful:
            fps.append(fp)
        else:
            logging.warning(f'Unable to find fixed point m={m} at point {guess}!!!!')

    fps_coords = [fp._current_location for fp in fps]
    ax.plot(
        initial_conditions[:, 0], initial_conditions[:, 1],
        marker='.', ls='', ms=3, color='k'
    )
    colors = ["red", "green", "blue", "yellow"]
    for i, fp in enumerate(fps):
        ax.plot(
            fp._fixed_point.coords[0, 0], fp._fixed_point.coords[0, 1],
            ls='', marker=get_fp_type(fp._fixed_point.GreenesResidue), markersize=10,
            color=['r', 'g', 'b', 'm', 'y', 'c'][np.mod(i, 6)],
            markeredgecolor='k' if get_fp_type(fp._fixed_point.GreenesResidue) != 'x' else None,
            mew=3
        )
    plt.savefig(os.path.join(OUTPUT_DIR, 'poincare_with_fps.png'))
    plt.close()

    logging.info("m#, Greene's residue")
    init_residues = []
    for fp in fps:
        try:
            logging.info(f"{fp._fixed_point.m}, {fp._fixed_point.GreenesResidue}")
            init_residues.append((fp._fixed_point.m, fp._fixed_point.GreenesResidue))
        except:
            logging.warning('Resonance not found')

    if CONFIG['opt']:
        INITCOILS = coils
        coils = INITCOILS
        base_curves = [coil.curve for coil in coils]
        base_currents = [coil.current for coil in coils]
        curves = base_curves
        currents = base_currents
        for i, current in enumerate(base_currents):
            if i < 9:
                current.fix_all()

        MS = [fp._fixed_point.m for fp in fps]
        weights = [0 if m == 2 else 1000 for m in MS]
        tuples = [(fp.residue, 0, w) for fp, w in zip(fps, weights)]
        prob = LeastSquaresProblem.from_tuples(tuples)
        logging.info(f'Objective before: {prob.objective()}')
        INIT_DOFS = np.copy(prob.x)
        logging.info(f'dofs before: {INIT_DOFS}')
        init_obj = np.copy(prob.objective())

        t1 = time.time()
        kwargs = {'max_nfev': 10, 'x_scale': 1000}
        least_squares_serial_solve(prob, grad=True, rel_step=0, abs_step=1e-06, **kwargs)
        fin_obj = np.copy(prob.objective())
        t2 = time.time()
        dt = t2 - t1
        logging.info(f'Solving took {dt} seconds or {dt//60} minutes or {dt/3600} hours')
        logging.info(f'Objective before: {init_obj}')
        logging.info(f'Objective after: {fin_obj}')
        logging.info(f'dofs names: {prob.dof_names}')
        logging.info(f'dofs after: {prob.x}')
        logging.info(f'new_dofs = {list(prob.x)}')
        # Removed undefined original_curve_dofs references

        logging.info('Final residues:')
        for i, fp in enumerate(fps):
            res_str = f"m={fp.fp_order} went from residue {init_residues[i][1]:3.1f} to final residue {fp.residue():3.1f}"
            if abs(fp.residue()) > 0.1:
                res_str += ' which is too large!'
            logging.info(res_str)

        # Iota computation
        compute_Iota = True
        iotas = []
        iota_nmaps = 10
        for i, fp in enumerate(fps):
            logging.info(f'Computing iota for fixed point {i}')
            try:
                iota = fp._map.winding(iota_nmaps, fp._fixed_point.coords[0], pyoproblem.axis)
                iotas.append(iota)
            except Exception as e:
                logging.error(f"Error computing iota: {e.args[0]}")

        IOTAS_R_VALUES = []
        IOTAS = []
        dphi = iota_nmaps * (2 * np.pi / CONFIG['nfp'])
        for fp, iota in zip(fps, iotas):
            if abs(fp._fixed_point.coords[0][1]) < 0.01:
                dtheta = iota[1]
                IOTAS.append(abs(dtheta/dphi))
                IOTAS_R_VALUES.append(fp._fixed_point.coords[0][0])

        fig, ax = plot_poincare_data_1(
            fieldlines_phi_hits,
            CONFIG['initial_phi'],
            'poincare_final.png',
            mark_lost=False,
            aspect="equal",
            dpi=300,
            xlims=[2.5, 5],
            ylims=[-1.0, 1.0],
            s=10,
            marker="o"
        )
        for i, fp in enumerate(fps):
            fp = fp._fixed_point
            o, coords, m, res = 0.05, fp.coords[0], fp.m, fp.GreenesResidue
            plt.annotate((r'$%d$' % m), xy=[coords[0], coords[1] - o], rotation=0, fontsize=10)
            plt.annotate((r'$%2.1le$' % res), xy=[coords[0], coords[1] + o], rotation=90, fontsize=10, color='g')
            m = get_fp_type(res)
            ax.plot(
                coords[0], coords[1],
                markersize=10, ls='', color=None,
                markeredgecolor='r', marker=m, markerfacecolor=None
            )
        for R, iota in zip(IOTAS_R_VALUES, IOTAS):
            plt.annotate(
                (r'$\iota=%2.1f$' % iota),
                xy=[R, -3 * o],
                rotation=-45,
                fontsize=10,
                color='k'
            )
        plt.savefig(os.path.join(OUTPUT_DIR, 'poincare_final_with_iota.png'))
        plt.close()
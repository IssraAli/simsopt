from simsopt.mhd.vmec import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.util import proc0_print
from simsopt.util import comm_world
import time
import numpy as np
from pathlib import Path
import sys
from simsopt.util.permanent_magnet_helper_functions import make_qfm
from simsopt.geo import (
    SurfaceRZFourier)
from simsopt import load


OUTDIR = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QH_total_length_4coils_setup/'
mpi = MpiPartition(ngroups=8)
comm = comm_world
print(
    'Script requires specifying two command line arguments -- '
    'the configuration name (QA, QH, QASH, CSX) and assumes that the biotsavart.json '
    'file containing the coil solution is in the passive_coils_<config_name> directory.'
)
nphi = 128 #256
ntheta = 128 #256
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
nfieldlines = 10 #35
tmax_fl = 10000
Z0 = np.zeros(nfieldlines)
aa = 0.06
if str(sys.argv[1]) == 'QA':
    input_name = 'input.LandremanPaul2021_QA_lowres'
    R0 = np.linspace(12.25*0.098749, 13.2*0.098749, nfieldlines)
    coilname = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QA_total_length_force_threshold12kNm/biot_savart_total_length.json'
elif str(sys.argv[1]) == 'QH':
    input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
    R0 = np.linspace(16.9, 17.8, nfieldlines)
    coilname = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QH_total_length_4coils_setup/biot_savart_qh.json'
    # coilname = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/LP_QH_total_length_5coils_setup/biot_savart_qh.json'
filename = TEST_DIR / input_name
Bfield = load(coilname)
coils = Bfield.coils
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
curves = [c.curve for c in coils]
base_curves = curves[:len(curves) // (s.nfp * 2)]
base_coils = coils[:len(coils) // (s.nfp * 2)]
ncoils = len(base_curves)
a_list = np.ones(len(base_curves)) * aa
b_list = np.ones(len(base_curves)) * aa
eval_points = s.gamma().reshape(-1, 3)
Bfield.set_points(s.gamma().reshape((-1, 3)))

BdotN = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
BdotN_over_B = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                        ) / np.mean(Bfield.AbsB())
Bn_plasma = None

print(BdotN, BdotN_over_B)

# # Make the QFM surfaces
qfm_surf = make_qfm(s, Bfield)
qfm_surf = qfm_surf.surface

# VMEC does NOT like the CSX plasma because it's very compact
# Also probably should be including the plasma current contributions
# Although Antoine Baillod has had the same issues.
vmec_input = str(filename)
equil = Vmec(vmec_input, mpi)
equil.boundary = qfm_surf
equil.run()

# Configure quasisymmetry objective:
if str(sys.argv[1]) == 'QH':
    helicity_n = -1
else:
    helicity_n = 0

qs = QuasisymmetryRatioResidual(equil,
                                np.arange(0, 1.01, 1.01/48),  # Radii to target
                                helicity_m=1, helicity_n=helicity_n)  # (M, N) you want in |B|
proc0_print("Quasisymmetry objective:", qs.total())
proc0_print("Quasisymmetry objective profile:", qs.profile())


import booz_xform as bx
from matplotlib import pyplot as plt

b2 = bx.Booz_xform()
b2.read_wout(equil.output_file)
b2.run()

## PLOT BOOZER (QFM AND TARGET)
plt.figure(1)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc("font", size=15)
bx.surfplot(b2, js = 48, fill=False)
plt.tight_layout()
plt.savefig(OUTDIR+"Boozer_VMEC.png", dpi=500)

plt.figure(2)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc("font", size=15)
plt.plot(np.arange(0, 1.01, 1.01/48), qs.profile())
plt.xlabel('Normalized toroidal flux')
plt.ylabel('Two-term quasisymmetry error')
plt.savefig(OUTDIR+"QS_profile.png", dpi=500)


# Need to load a wout file to get the proper iota profile!
plt.figure()
plt.grid()
psi_s = np.linspace(0, len(equil.wout.iotas[1::]) * equil.ds, len(equil.wout.iotas[1::]))
if str(sys.argv[1]) == 'QH':
    sign = -1
else:
    sign = 1
plt.rcParams["font.family"] = "Times New Roman"
plt.rc("font", size=15)
plt.plot(psi_s, sign*equil.wout.iotas[1::], 'rx')
plt.ylabel(r'rotational transform $\iota$')
plt.xlabel('Normalized toroidal flux s')
plt.savefig(OUTDIR+'iota_profile.png', dpi=500)
from simsopt.field.magneticfieldclasses import InterpolatedField
from simsopt.geo import SurfaceClassifier

sc_fieldline = SurfaceClassifier(s, h=0.02, p=2)

def skip(rs, phis, zs):
    # Reused function from examples/1_Simple/fieldline_tracing_QA.py
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip

n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / s.nfp, n * 2)
zrange = (0, np.max(zs), n // 2)
degree = 4  # 2 is sufficient sometimes
Bfield.set_points(s.gamma().reshape((-1, 3)))
bsh = InterpolatedField(
    Bfield, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym, skip=skip
)
bsh.set_points(s.gamma().reshape((-1, 3)))
from simsopt.util import proc0_print

phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]
print(rrange, zrange, phirange)
print(R0, Z0)

t1 = time.time()
# compute the fieldlines from the initial locations specified above

# fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
#     bsh, R0, Z0, tmax=tmax_fl, tol=1e-10, comm=comm,
#     phis=phis,
#     stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
# t2 = time.time()
# proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)

# # make the poincare plots
# if comm is None or comm.rank == 0:
#     plot_poincare_data(fieldlines_phi_hits, phis, OUTDIR+'poincare_fieldline' + str(sys.argv[1]) + '.png', dpi=300, surf=None, aspect='auto')

# plt.show()

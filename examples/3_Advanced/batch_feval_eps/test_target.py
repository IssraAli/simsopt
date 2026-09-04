import numpy as np
import torch
from bo_utils import from_unit_cube
from eps_eff6 import EffectiveRipple
from mpi4py import MPI
from simsopt._core import make_optimizable
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.util.mpi import MpiPartition

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

INVALID_PENALTY = np.array([-100])
default_fev_var = np.array([1e-3])
failed_fev_var = np.array([5e2])

mpi = MpiPartition()


def parallel_batch_target(candidates, spline_kwargs, lb, ub, stopp):
    stopp[0] = comm.bcast(stopp[0], root=0)
    if stopp[0] == 0:
        x = comm.scatter(candidates, root=0)
        val, var = target(x.flatten(), spline_kwargs, lb, ub)
        gathered_val = comm.gather(val)
        gathered_var = comm.gather(var)
        if rank == 0:
            print("batch complete. ")
            Y_cand = np.array(gathered_val)
            Yvar_cand = np.array(gathered_var)
            return torch.Tensor(Y_cand).reshape(-1, 1), torch.Tensor(
                Yvar_cand
            ).reshape(-1, 1)


def target(X, spline_kwargs, lb, ub):
    dofs = from_unit_cube(X, lb, ub)
    # print(dofs)

    surf = SurfaceBSpline(**spline_kwargs)
    surf.axis.fix("r_axis_0")
    assert len(surf.x) == len(dofs), (
        f"len(surf.x): {len(surf.x)}, len(dofs): {len(dofs)}"
    )
    surf.x = np.array(dofs)
    # try:
    vmec = Vmec.vmec_from_surf(
        nfp=surf.nfp, surf=surf, mpi=mpi, ns=50, M=12, N=12, ftol=1e-8
    )
    vmec.run()

    def eps_eff_callable(vmec):
        ripple = EffectiveRipple(vmec, np.linspace(1e-3, 1, 10))
        results = ripple.compute()
        return results.eps_eff_32

    e32 = make_optimizable(eps_eff_callable, vmec)

    prob = LeastSquaresProblem.from_tuples(
        [
            (e32.J, 0, 1),
            (vmec.aspect, 4, 10),
            (vmec.mean_iota, 0.42, 10),
            # (vmec.iota_edge(), 0.42, 10)
        ]
    )
    return np.maximum(prob.objective(), INVALID_PENALTY), default_fev_var
    # except Exception as e:
    #     print(f"Failed with exception {e}, appending invalid penalty")
    #     rz_surf = surf.to_RZFourier()
    #     rz_surf.plot(engine='plotly')
    #     return INVALID_PENALTY, failed_fev_var


def ar_target(vmec, target):
    val = vmec.aspect()
    return val - target


def iota_target(vmec, target):
    val = vmec.mean_iota()  # np.abs(vmec.mean_iota())
    return val - target

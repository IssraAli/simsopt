"""
Bounds, unit-cube transforms, and warm-start utilities for composite BO.

Bounds match the LSQ reference (`spline_surface.py:160-171`); the gauge fix on
`r_axis_0` matches `spline_surface.py:312`.
"""

import fnmatch

import numpy as np
import torch
from simsopt.geo import SurfaceBSpline


def write_doflist_maxlist_minlist(spline_kwargs):
    """
    Build a template surface, apply the same gauge fix the LSQ reference uses,
    and return (dof_names, ub, lb) over the *free* DOFs after the fix.
    """
    template_surf = SurfaceBSpline(**spline_kwargs)
    template_surf.axis.fix('r_axis_0')

    doflist = template_surf.dof_names
    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    cs_r_indices = [fnmatch.fnmatch(d, 'CrossSectionFixedZeta*r*') for d in doflist]
    r_axis_indices = [fnmatch.fnmatch(d, 'PseudoAxis*r_axis*') for d in doflist]
    z_axis_indices = [fnmatch.fnmatch(d, 'PseudoAxis*z_axis*') for d in doflist]

    lb[cs_r_indices] = 0.01
    ub[cs_r_indices] = 0.8

    lb[r_axis_indices] = 0.7
    ub[r_axis_indices] = 2.2

    lb[z_axis_indices] = -0.5
    ub[z_axis_indices] = 0.5

    return doflist, ub, lb


def to_unit_cube(x, lb, ub):
    """Project from physical hypercube [lb, ub] to [0, 1]^d."""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1
    return (x - lb) / (ub - lb)


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d back to the physical hypercube [lb, ub]."""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1, f'lb: {lb}, ub: {ub}'
    return x * (ub - lb) + lb


def make_warm_start(spline_kwargs, lb, ub, n_perturb=4, sigma=0.02, seed=0,
                    default_r=0.4, dtype=torch.double):
    """
    Build a small batch of warm-start unit-cube points anchored at the LSQ
    reference initial condition: a circular cross-section surface with
    `default_r=0.4`. Returns a tensor of shape (1 + n_perturb, dims) where
    row 0 is the center and the remaining rows are Gaussian perturbations
    (clipped to [0, 1]).
    """
    surf = SurfaceBSpline(**spline_kwargs, default_r=default_r)
    surf.axis.fix('r_axis_0')

    x_phys = np.asarray(surf.x, dtype=np.float64)
    assert x_phys.shape == lb.shape, (
        f'warm-start dim {x_phys.shape} disagrees with bounds dim {lb.shape}; '
        'check that gauge fix matches between bounds and warm-start construction'
    )

    center_unit = to_unit_cube(x_phys, lb, ub)
    # If default_r happens to fall outside the bounds box for any DOF, clip
    # so the warm-start is at least inside the unit cube.
    center_unit = np.clip(center_unit, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    perturbations = rng.normal(scale=sigma, size=(n_perturb, len(center_unit)))
    perturbed = np.clip(center_unit[None, :] + perturbations, 0.0, 1.0)

    warm = np.vstack([center_unit[None, :], perturbed])  # (1 + n_perturb, dims)
    return torch.tensor(warm, dtype=dtype)

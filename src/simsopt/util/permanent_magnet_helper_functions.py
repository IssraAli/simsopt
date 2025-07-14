"""
This module contains the a number of useful functions for using 
the permanent magnets functionality in the SIMSOPT code.
"""
__all__ = [
           'initialize_coils_for_pm_optimization',
           'make_optimization_plots', 'run_Poincare_plots',
           'initialize_default_kwargs'
           ]

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def initialize_coils_for_pm_optimization(config_flag, TEST_DIR, s, out_dir=''):
    """
    Initializes coils for each of the target configurations that are
    used for permanent magnet optimization. The total current is chosen
    to achieve a desired average field strength along the major radius.

    Args:
        config_flag: String denoting the stellarator configuration 
          being initialized.
        TEST_DIR: String denoting where to find the input files.
        out_dir: Path or string for the output directory for saved files.
        s: plasma boundary surface.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk
    from simsopt.util.coil_optimization_helper_functions import read_focus_coils

    out_dir = Path(out_dir)
    if 'muse' in config_flag:
        # Load in pre-optimized coils
        coils_filename = TEST_DIR / 'muse_tf_coils.focus'

        # Slight difference from older MUSE initialization. 
        # Here we fix the total current summed over all coils and 
        # older MUSE opt. fixed the first coil current. 
        base_curves, base_currents0, ncoils = read_focus_coils(coils_filename)
        total_current = np.sum([curr.get_value() for curr in base_currents0])
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = []
        for i in range(ncoils):
            coils.append(Coil(base_curves[i], base_currents[i]))

    elif config_flag == 'qh':
        # generate planar TF coils
        ncoils = 4
        R0 = s.get_rc(0, 0)
        R1 = s.get_rc(1, 0) * 4
        order = 2

        # QH reactor scale needs to be 5.7 T average magnetic field strength
        total_current = 48712698  # Amperes
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True,
                                                   R0=R0, R1=R1, order=order, numquadpoints=64)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    elif config_flag == 'qa':
        # generate planar TF coils
        ncoils = 8
        R0 = 1.0
        R1 = 0.65
        order = 5

        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        total_current = 187500
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    
    # fix all the coil shapes so only the currents are optimized
    for i in range(ncoils):
        base_curves[i].fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir / "curves_init")
    return base_curves, curves, coils


def make_optimization_plots(RS_history, m_history, m_proxy_history, pm_opt, out_dir=''):
    """
    Make line plots of the algorithm convergence and make histograms
    of m, m_proxy, and if available, the FAMUS solution.

    Args:
        RS_history: List of the relax-and-split optimization progress.
        m_history: List of the permanent magnet solutions generated
          as the relax-and-split optimization progresses.
        m_proxy_history: Same but for the 'proxy' variable in the
          relax-and-split optimization.
        pm_opt: PermanentMagnetGrid class object that was optimized.
        out_dir: Path or string for the output directory for saved files.
    """
    out_dir = Path(out_dir)

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(out_dir / 'RS_objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    m0_abs = np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    mproxy_abs = np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima

    # get FAMUS rho values for making comparison histograms
    if hasattr(pm_opt, 'famus_filename'):
        m0, p = np.loadtxt(
            pm_opt.famus_filename, skiprows=3,
            usecols=[7, 8],
            delimiter=',', unpack=True
        )
        # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
        momentq = np.loadtxt(pm_opt.famus_filename, skiprows=1, max_rows=1, usecols=[1])
        rho = p ** momentq
        rho = rho[pm_opt.Ic_inds]
        x_multi = [m0_abs, mproxy_abs, abs(rho)]
    else:
        x_multi = [m0_abs, mproxy_abs]
        rho = None
    plt.hist(x_multi, bins=np.linspace(0, 1, 40), log=True, histtype='bar')
    plt.grid(True)
    plt.legend(['m', 'w', 'FAMUS'])
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(out_dir / 'm_histograms.png')

    # If relax-and-split was used with l0 norm,
    # make a nice histogram of the algorithm progress
    if len(RS_history) != 0:

        m_history = np.array(m_history)
        m_history = m_history.reshape(m_history.shape[0] * m_history.shape[1], pm_opt.ndipoles, 3)
        m_proxy_history = np.array(m_proxy_history).reshape(m_history.shape[0], pm_opt.ndipoles, 3)

        for i, datum in enumerate([m_history, m_proxy_history]):
            # Code from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
            def prepare_animation(bar_container):
                def animate(frame_number):
                    plt.title(frame_number)
                    data = np.sqrt(np.sum(datum[frame_number, :] ** 2, axis=-1)) / pm_opt.m_maxima
                    n, _ = np.histogram(data, np.linspace(0, 1, 40))
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    return bar_container.patches
                return animate

            # make histogram animation of the dipoles at each relax-and-split save
            fig, ax = plt.subplots()
            data = np.sqrt(np.sum(datum[0, :] ** 2, axis=-1)) / pm_opt.m_maxima
            if rho is not None:
                ax.hist(abs(rho), bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='g')
            _, _, bar_container = ax.hist(data, bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='r')
            ax.set_ylim(top=1e5)  # set safe limit to ensure that all data is visible.
            plt.grid(True)
            if rho is not None:
                plt.legend(['FAMUS', np.array([r'm$^*$', r'w$^*$'])[i]])
            else:
                plt.legend([np.array([r'm$^*$', r'w$^*$'])[i]])

            plt.xlabel('Normalized magnitudes')
            plt.ylabel('Number of dipoles')
            animation.FuncAnimation(
                fig, prepare_animation(bar_container),
                range(0, m_history.shape[0], 2),
                repeat=False, blit=True
            )


def run_Poincare_plots(s_plot, bs, b_dipole, comm, filename_poincare, out_dir=''):
    """
    Wrapper function for making Poincare plots.

    Args:
        s_plot: plasma boundary surface with range = 'full torus'.
        bs: MagneticField class object.
        b_dipole: DipoleField class object.
        comm: MPI COMM_WORLD object for using MPI for tracing.
        filename_poincare: Filename for the output poincare picture.
        out_dir: Path or string for the output directory for saved files.
    """
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.util.coil_optimization_helper_functions import make_Bnormal_plots, trace_fieldlines
    # from simsopt.objectives import SquaredFlux

    out_dir = Path(out_dir)

    n = 32
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    r_margin = 0.05
    rrange = (np.min(rs) - r_margin, np.max(rs) + r_margin, n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 4  # 2 is sufficient sometimes
    # nphi = len(s_plot.quadpoints_phi)
    # ntheta = len(s_plot.quadpoints_theta)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    # Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    # Bnormal_dipole = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    # f_B = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_pre_poincare_check")
    make_Bnormal_plots(b_dipole, s_plot, out_dir, "dipole_pre_poincare_check")
    make_Bnormal_plots(bs + b_dipole, s_plot, out_dir, "total_pre_poincare_check")

    bsh = InterpolatedField(
        bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    trace_fieldlines(bsh, 'bsh_PMs_' + filename_poincare, s_plot, comm, out_dir)

def initialize_default_kwargs(algorithm='RS'):
    """
    Keywords to the permanent magnet optimizers are now passed
    by the kwargs dictionary. Default dictionaries are initialized
    here.

    Args:
        algorithm: String denoting which algorithm is being used
          for permanent magnet optimization. Options are 'RS'
          (relax and split), 'GPMO' (greedy placement), 
          'backtracking' (GPMO with backtracking), 
          'multi' (GPMO, placing multiple magnets each iteration),
          'ArbVec' (GPMO with arbitrary dipole orientiations),
          'ArbVec_backtracking' (GPMO with backtracking and 
          arbitrary dipole orientations).

    Returns:
        kwargs: Dictionary of keywords to pass to a permanent magnet 
          optimizer.
    """

    kwargs = {}
    kwargs['verbose'] = True   # print out errors every few iterations
    if algorithm == 'RS':
        kwargs['nu'] = 1e100  # Strength of the "relaxation" part of relax-and-split
        kwargs['max_iter'] = 100  # Number of iterations to take in a convex step
        kwargs['reg_l0'] = 0.0
        kwargs['reg_l1'] = 0.0
        kwargs['alpha'] = 0.0
        kwargs['min_fb'] = 0.0
        kwargs['epsilon'] = 1e-3
        kwargs['epsilon_RS'] = 1e-3
        kwargs['max_iter_RS'] = 2  # Number of total iterations of the relax-and-split algorithm
        kwargs['reg_l2'] = 0.0
    elif 'GPMO' in algorithm or 'ArbVec' in algorithm:
        kwargs['K'] = 1000
        kwargs["reg_l2"] = 0.0
        kwargs['nhistory'] = 500  # K > nhistory and nhistory must be divisor of K
    return kwargs


def initialize_coils_simple(s, out_dir='', target_B=5.7, ncoils=4, order=16, nturns=256):
    """
    Initializes four coils with order=16 and total current set to produce 
    a target B-field on-axis. The coil centers and radii are scaled by 
    the plasma surface major radius. The function iteratively adjusts the
    total current until the field strength along the major radius averages
    to the target value.

    Args:
        s: plasma boundary surface.
        out_dir: Path or string for the output directory for saved files.
        target_B: Target magnetic field strength in Tesla (default: 5.7).
    Returns:
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries, BiotSavart
    from simsopt.field.coil import coils_to_vtk

    out_dir = Path(out_dir)
    
    # Get the major radius from the surface and scale coil parameters
    R0 = s.get_rc(0, 0)  # Major radius
    R1 = s.get_rc(1, 0) * 2.5  # Scale the minor radius component
    
    # Initial guess for total current (using QH configuration as reference)
    total_current = 5e7  # 50 MA initial guess is not bad for reactor-scale
    
    # Create equally spaced curves with the specified parameters
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256)
    
    # print(f"Target B-field: {target_B} T")
    # print(f"Major radius: {R0:.3f} m")
    # print(f"Minor radius component: {s.get_rc(1, 0):.3f} m")
    # print(f"NFP: {s.nfp}")
    
    # Iterative current adjustment
    max_iterations = 20
    tolerance = 1e-2
    for iteration in range(max_iterations):
        # print(f"Iteration {iteration + 1}/{max_iterations}")
        # print(f"  Current total current: {total_current:.0f} A")
        
        # Distribute current among coils
        base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
        total_current_obj = Current(total_current)
        total_current_obj.fix_all()
        base_currents += [total_current_obj - sum(base_currents)]
        
        # Create coils using symmetries
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        
        # Create BiotSavart object to evaluate field
        bs = BiotSavart(coils)
        
        # Calculate field strength along major radius
        B_avg = calculate_modB_on_major_radius(bs, s)
        
        # print(f"  Achieved B-field: {B_avg:.3f} T")
        # print(f"  Difference: {B_avg - target_B:.3f} T")
        
        # Check convergence
        if abs(B_avg - target_B) / target_B < tolerance:
            # print(f"  ✓ Converged! B-field within {tolerance*100:.1f}% of target")
            break
        
        # Adjust current based on field difference
        # Use simple linear scaling: new_current = current * (target_B / achieved_B)
        current_scale_factor = target_B / B_avg
        total_current *= current_scale_factor
        
        # print(f"  New total current: {total_current:.0f} A")
    
    else:
        print(f"  ⚠ Warning: Did not converge within {max_iterations} iterations")
        print(f"  Final B-field: {B_avg:.3f} T (target: {target_B} T)")
    
    # Save final coils to VTK
    coils_to_vtk(coils, out_dir / "coils_init", nturns=nturns)
    
    # print(f"Final configuration:")
    
    return coils


def optimize_coils_simple(s, target_B=5.7, out_dir='', max_iterations=1500, max_iter_lag=50, 
                         ncoils=4, order=16, nphi=32, ntheta=32, verbose=False, **kwargs):
    """
    Performs complete coil optimization including initialization and optimization.
    This function initializes coils with the target B-field and then optimizes
    them using the augmented Lagrangian method.

    Args:
        s: plasma boundary surface.
        target_B: Target magnetic field strength in Tesla (default: 5.7).
        out_dir: Path or string for the output directory for saved files.
        max_iterations: Maximum number of optimization iterations (default: 1500).
        max_iter_lag: Maximum number of Lagrangian iterations (default: 50).
        ncoils: Number of base coils to create (default: 4).
        order: Fourier order for coil curves (default: 16).
        nphi: Number of phi points for surface discretization (default: 32).
        ntheta: Number of theta points for surface discretization (default: 32).
        verbose: Print out progress and results (default: False).
        **kwargs: Additional keyword arguments for constraint thresholds.
    Returns:
        coils: List of optimized Coil class objects.
        results: Dictionary containing optimization results and metrics.
    """
    import numpy as np
    import time
    from pathlib import Path
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo import LinkingNumber, CurveLength, CurveCurveDistance
    from simsopt.geo import LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
    from simsopt.objectives import SquaredFlux, QuadraticPenalty, Weight
    from simsopt.solve import augmented_lagrangian_method
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.field.force import LpCurveForce, LpCurveTorque, coil_force

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set default constraint thresholds if not provided
    # Defaults here are reasonable for 10 m major radius
    #  reactor-scale device with 5.7 T target B-field
    length_target = kwargs.get('length_target', 210.0)
    flux_threshold = kwargs.get('flux_threshold', 1e-8)
    cc_threshold = kwargs.get('cc_threshold', 1.0)
    cs_threshold = kwargs.get('cs_threshold', 1.5)
    msc_threshold = kwargs.get('msc_threshold', 1.0)
    curvature_threshold = kwargs.get('curvature_threshold', 1.0)

    # 1 MN/m on each of an assumed 256 turns of the coil
    nturns = kwargs.get('nturns', 256)
    force_threshold = kwargs.get('force_threshold', 1.0) * nturns
    torque_threshold = kwargs.get('torque_threshold', 1.0) * nturns

    # Rescale all the length thresholds by the plasma major radius
    # divided by the 10m assumption for the major radius
    R0 = 10.0 / s.get_rc(0, 0)
    length_target /= R0
    length_target *= (ncoils / 7.0) ** 0.5
    cc_threshold /= R0
    cs_threshold /= R0 
    curvature_threshold *= R0
    msc_threshold *= R0

    # print(f"Starting coil optimization for target B-field: {target_B} T")
    # print(f"Surface major radius: {s.get_rc(0, 0):.3f} m")
    # print(f"Surface minor radius component: {s.get_rc(1, 0):.3f} m")
    # print(f"Number of base coils: {ncoils}")
    # print(f"Fourier order: {order}")

    # Step 1: Initialize coils with target B-field
    # print("Step 1: Initializing coils with target B-field...")
    coils = initialize_coils_simple(s, out_dir=out_dir, target_B=target_B, ncoils=ncoils, order=order, nturns=nturns)

    # Rescale force_threshold
    total_current = sum([c.current.get_value() for c in coils[:ncoils]]) / (s.stellsym + 1) / s.nfp
    coils_backup = initialize_coils_simple(s, out_dir=out_dir, ncoils=ncoils, order=order, nturns=nturns)
    total_current_reactor_scale = sum([c.current.get_value() for c in coils_backup[:ncoils]]) / (s.stellsym + 1) / s.nfp
    force_threshold *= (total_current / total_current_reactor_scale) ** 2
    torque_threshold *= (total_current / total_current_reactor_scale) ** 2

    # Extract base curves and currents from the initialized coils
    base_curves = [coil.curve for coil in coils[:ncoils]]
    
    # print(f"Initialized {len(coils)} coils (including symmetries)")

    # Step 2: Create plotting surface for visualization
    # print("Step 2: Setting up plotting surface...")
    qphi = 4 * nphi
    qtheta = 4 * ntheta
    quadpoints_phi = np.linspace(0, 1, qphi)
    quadpoints_theta = np.linspace(0, 1, qtheta)
    
    # Create a plotting surface (full torus)
    # Handle case where surface was created manually (no filename)
    if hasattr(s, 'filename') and s.filename is not None:
        s_plot = SurfaceRZFourier.from_vmec_input(
            s.filename,
            range="full torus",
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            nfp=s.nfp,
            stellsym=s.stellsym
        )
    else:
        # Create surface manually with same parameters
        s_plot = SurfaceRZFourier(
            nfp=s.nfp,
            stellsym=s.stellsym,
            mpol=s.mpol,
            ntor=s.ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta
        )
    
    # Copy the surface coefficients
    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            if s.get_rc(m, n) != 0:
                s_plot.set_rc(m, n, s.get_rc(m, n))
            if s.get_zs(m, n) != 0:
                s_plot.set_zs(m, n, s.get_zs(m, n))

    # Step 3: Create BiotSavart object and save initial state
    # print("Step 3: Creating BiotSavart object and saving initial state...")
    bs = BiotSavart(coils)
    B_avg = calculate_modB_on_major_radius(bs, s)
    print(f"  Total current: {total_current:.0f} A")
    print(f"  B-field averaged along major radius: {B_avg:.3f} T")
    print(f"  Number of coils: {len(coils)}")
    curves = [c.curve for c in coils]
    
    # Save initial coils
    coils_to_vtk(coils, out_dir / "coils_initial", nturns=nturns)
    
    # Calculate and display initial B-field
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    B_initial = calculate_modB_on_major_radius(bs, s_plot)
    # print(f"Initial B-field on-axis: {B_initial:.3f} T")
    
    # Save initial surface data
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                          s_plot.unitnormal(), axis=2)[:, :, None] / 
                    bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))
    }
    s_plot.to_vtk(out_dir / "surface_initial", extra_data=pointData)

    # Step 4: Define objective function and constraints
    # print("Step 4: Setting up optimization objectives and constraints...")
    bs.set_points(s.gamma().reshape((-1, 3)))
    
    # Main objective: Squared flux
    Jf = SquaredFlux(s, bs, definition="normalized", threshold=flux_threshold)
    
    # Constraint terms
    Jls = [CurveLength(c) for c in base_curves]
    Jl = sum(QuadraticPenalty(jj, length_target, "max") for jj in Jls)
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jcs = [LpCurveCurvature(c, 2, curvature_threshold) for c in base_curves]
    Jlink = LinkingNumber(curves, downsample=2)
    Jforce = LpCurveForce(coils[:ncoils], coils, p=2.0, threshold=force_threshold, downsample=2)
    Jtorque = LpCurveTorque(coils[:ncoils], coils, p=2.0, threshold=torque_threshold, downsample=2)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Print initial constraint values
    print("Initial thresholds:")
    print(f" Flux Threshold: {flux_threshold:.2e}")
    print(f" Length Target: {length_target:.2e}")
    print(f" CC Threshold: {cc_threshold:.2e}")
    print(f" CS Threshold: {cs_threshold:.2e}")
    print(f" MSC Threshold: {msc_threshold:.2e}")
    print(f" Curvature Threshold: {curvature_threshold:.2e}")
    print(f" Force Threshold: {force_threshold:.2e}")
    print(f" Torque Threshold: {torque_threshold:.2e}")

    # Step 5: Run optimization
    # print("Step 5: Running optimization...")
    start_time = time.time()
    
    # Constraint list for augmented Lagrangian method
    c_list = [
        Jf,
        Jccdist,
        Weight(1e3) * Jcsdist,
        QuadraticPenalty(sum(Jls), length_target, "max"),
        sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs),
        sum(Jcs),
        Jlink,
        Jforce,
        Jtorque
    ]
    
    # Run optimization
    _, _, lag_mul = augmented_lagrangian_method(
        f=None,  # No main objective function
        equality_constraints=c_list,
        MAXITER=max_iterations,
        MAXITER_lag=max_iter_lag,
        verbose=verbose,
    )
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.1f} seconds")

    # Step 6: Save results and final state
    # print("Step 6: Saving results and final state...")
    
    # Save optimized coils
    coils_to_vtk(coils, out_dir / "coils_optimized", nturns=nturns)
    bs.save(out_dir / "biot_savart_optimized.json")
    
    # Calculate and display final B-field
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    B_final = calculate_modB_on_major_radius(bs, s_plot)
    print(f"Final B-field on-axis: {B_final:.3f} T")
    
    # Save final surface data
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {
        "B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                     s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                         s_plot.unitnormal(), axis=2)[:, :, None] /
                   bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))
    }
    s_plot.to_vtk(out_dir / "surface_optimized", extra_data=pointData)
    
    # Print final constraint values
    bs.set_points(s.gamma().reshape((-1, 3)))
    print("Final constraint values:")
    print(f"  Normalized flux: {Jf.J():.2e}")
    print(f"  CS separation: {Jcsdist.J():.2e} (min distance: {Jcsdist.shortest_distance():.3f})")
    print(f"  CC separation: {Jccdist.J():.2e} (min distance: {Jccdist.shortest_distance():.3f})")
    print(f"  Length constraint: {Jl.J():.2e}")
    print(f"  Curvature constraint: {sum(Jcs).J():.2e}")
    print(f"  Linking number: {Jlink.J():.2e}")
    print(f"  Force constraint: {Jforce.J():.2e}")
    print(f"  Max curvatures: {[np.max(c.kappa()) for c in base_curves]}")
    print(f"  Lengths: {[CurveLength(c).J() for c in base_curves]}")
    
    # Calculate final forces
    force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in coils[:ncoils]]
    print(f"  Forces: {[f'{f:.2e}' for f in force]}")
    
    # Calculate final B_N metrics
    bs.set_points(s.gamma().reshape((-1, 3)))
    nphi_s = len(s.quadpoints_phi)
    ntheta_s = len(s.quadpoints_theta)
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi_s, ntheta_s, 3)) * s.unitnormal(), axis=2)))
    avg_BdotN_over_B = BdotN / bs.AbsB().mean()
    
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    nphi_plot = len(s_plot.quadpoints_phi)
    ntheta_plot = len(s_plot.quadpoints_theta)
    max_BdotN_overB = np.max(np.abs(np.sum(bs.B().reshape((nphi_plot, ntheta_plot, 3)) *
                                          s_plot.unitnormal(), axis=2)) /
                            bs.AbsB().reshape((nphi_plot, ntheta_plot, 1)))
    
    print(f"  <B_N>/<|B|> = {avg_BdotN_over_B:.2e}")
    print(f"  Max |B_N|/|B| = {max_BdotN_overB:.2e}")    
    print("Optimization completed successfully!")
    print(f"Results saved to: {out_dir}")
    
    # Prepare results dictionary
    bs.set_points(s.gamma().reshape((-1, 3)))
    results = {
        'initial_B_field': B_initial,
        'final_B_field': B_final,
        'target_B_field': target_B,
        'optimization_time': end_time - start_time,
        'final_flux': Jf.J(),
        'final_cs_separation': Jcsdist.J(),
        'final_cc_separation': Jccdist.J(),
        'final_length_constraint': Jl.J(),
        'final_curvature_constraint': sum(Jcs).J(),
        'final_linking_number': Jlink.J(),
        'final_force_constraint': Jforce.J(),
        'final_forces': force,
        'final_lengths': [CurveLength(c).J() for c in base_curves],
        'final_max_curvatures': [np.max(c.kappa()) for c in base_curves],
        'avg_BdotN_over_B': avg_BdotN_over_B,
        'max_BdotN_over_B': max_BdotN_overB,
        'lagrange_multipliers': lag_mul,
        'output_directory': str(out_dir)
    }
    
    return coils, results

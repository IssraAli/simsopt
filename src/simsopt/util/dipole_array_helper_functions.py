"""
This module contains the a number of useful functions for
optimizing dipole arrays in the SIMSOPT code.
"""

__all__ = [
    "remove_inboard_dipoles",
    "remove_interlinking_dipoles_and_TFs",
    "align_dipoles_with_plasma",
    "initialize_coils",
    "dipole_array_optimization_function",
    "save_coil_sets",
    "quaternion_from_axis_angle",
    "quaternion_multiply",
    "rotate_vector",
    "compute_quaternion",
    "compute_fourier_coeffs",
    "rho_fourier",
    "generate_even_arc_angles",
    "generate_even_arc_angles_from_surface",
    "generate_windowpane_array",
    "generate_windowpane_wedge_array",
    "generate_windowpane_ring_array",
    "generate_windowpane_metric_ring_array",
    "generate_tf_array",
    "generate_curves",
    "rho",
    "a_m",
    "b_m",
]

import numpy as np


def remove_inboard_dipoles(plasma_surf, base_curves, eps=-0.4):
    """
    Remove all the dipole coils on the inboard side. Useful if the desired plasma
    configuration is fairly compact, so no room for dipoles on inboard side.

    Args:
        plasma_surf: Surface object
            The plasma boundary surface.
        base_curves:
            The curve objects for the dipoles in the array.
        eps: float
            Controls how inboard a dipole can be without being removed.

    Returns:
        base_curves:
            The same objects, minus any dipoles on the inboard side.
    """
    import warnings

    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :]) ** 2))
            conflict_bool = dij < (1.0 + eps) * plasma_surf.get_rc(0, 0)
            if conflict_bool:
                print("bad index = ", i, dij, plasma_surf.get_rc(0, 0))
                warnings.warn(
                    "There is a PSC coil initialized such that it is within a radius"
                    "of a TF coil. Deleting these PSCs now."
                )
                counter += 1
                break
        if counter == 0:
            keep_inds.append(ii)
    return np.array(base_curves)[keep_inds]


def remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF, eps=0.05):
    """
    Similar process to remove any dipole coils that are initialized
    intersecting with the TF coils, to avoid interlinking at the very beginning of optimization.

    Args:
        base_curves:
            The curve objects for the dipoles in the array.
        base_curves_TF:
            The curve objects for the TF (modular) coils in the array.
        eps: float
            controls how close a TF-dipole coil distance can be before being removed.

    Returns:
        base_curves:
            The same objects, minus any dipoles that were interlinking the TF coils.
    """
    import warnings

    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            for j in range(len(base_curves_TF)):
                for k in range(base_curves_TF[j].gamma().shape[0]):
                    dij = np.sqrt(
                        np.sum(
                            (
                                base_curves[ii].gamma()[i, :]
                                - base_curves_TF[j].gamma()[k, :]
                            )
                            ** 2
                        )
                    )
                    conflict_bool = dij < (1.0 + eps) * base_curves[0].x[0]
                    if conflict_bool:
                        # print('bad indices = ', i, j, dij, base_curves[0].x[0])
                        warnings.warn(
                            "There is a PSC coil initialized such that it is within a radius"
                            "of a TF coil. Deleting these PSCs now."
                        )
                        counter += 1
                        break
        if counter == 0:
            keep_inds.append(ii)
    return np.array(base_curves)[keep_inds]


def align_dipoles_with_plasma(plasma_surf, base_curves):
    """
    Initialize a set of dipole coils in base_curves
    to point in the same direction as the nearest plasma surface normal.

    Args:
        plasma_surf: Surface object
            The plasma boundary surface.
        base_curves: list of Curve objects
            The dipole coils to align with the plasma normals.

    Returns:
        alphas: np.ndarray
            One of the unique angles of the planar dipole coils.
        deltas: np.ndarray
            The other unique Euler angle of the planar dipole coils.
    """
    ncoils = len(base_curves)
    coil_normals = np.zeros((ncoils, 3))
    plasma_points = plasma_surf.gamma().reshape(-1, 3)
    plasma_unitnormals = plasma_surf.unitnormal().reshape(-1, 3)
    for i in range(ncoils):
        point = base_curves[i].get_dofs()[-3:]
        dists = np.sum((point - plasma_points) ** 2, axis=-1)
        min_ind = np.argmin(dists)
        coil_normals[i, :] = plasma_unitnormals[min_ind, :]
    coil_normals = coil_normals / np.linalg.norm(coil_normals, axis=-1)[:, None]
    alphas = np.arcsin(
        -coil_normals[:, 1],
    )
    deltas = np.arctan2(coil_normals[:, 0], coil_normals[:, 2])
    return alphas, deltas


def initialize_coils(s, configuration, regularization):
    """
    Initializes appropriate coils for the Schuett-Henneberg 2-field-period QA,
    Landreman-Paul QA, Landreman-Paul QH, etc., usually for purposes
    of finding a dipole coil array solution.

    Args:
        s: plasma boundary surface.
        configuration: String denoting the stellarator name.
        regularization: Regularization object for RegularizedCoil objects.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
        base_currents: List of Current class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries

    # parameters for the TF coils, increase order for a better solution
    # Total current scaled to give B ~ 5.7 T on axis (actually averaged over the major radius)
    if configuration == "LandremanPaulQA":
        ncoils = 3
        R0 = s.get_rc(0, 0) * 1
        R1 = s.get_rc(1, 0) * 3.5
        order = 8
        total_current = 66237606
    elif configuration == "LandremanPaulQH":
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1
        R1 = s.get_rc(1, 0) * 4
        order = 8
        total_current = 45642162
    elif configuration == "SchuettHennebergQAnfp2":
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1.4
        R1 = s.get_rc(1, 0) * 4
        order = 16
        total_current = 35000000
    else:
        raise ValueError("Stellarator configuration not recognized.")
    print("Total current = ", total_current)

    # Create the initial coils
    base_curves = create_equally_spaced_curves(
        ncoils,
        s.nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
        numquadpoints=256,
    )
    base_currents = [
        (Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)
    ]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        s.nfp,
        s.stellsym,
        regularizations=[regularization for _ in range(ncoils)],
    )
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents


def dipole_array_optimization_function(dofs, obj_dict, weight_dict, psc_array=None):
    """
    Wrapper function for performing dipole array optimization.

    Args:
        dofs: np.ndarray
            The degrees of freedom for the optimization.
        obj_dict: dict
            Dictionary of objective functions.
        weight_dict: dict
            Dictionary of weights for the objective functions.

    Returns:
        J: float
            The objective function value.
        grad: np.ndarray
            The gradient of the objective function.
    """
    from simsopt.objectives import QuadraticPenalty

    # unpack all the dictionary objects
    btot = obj_dict["btot"]
    s = obj_dict["s"]
    base_curves_TF = obj_dict["base_curves_TF"]
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    JF = obj_dict["JF"]
    Jf = obj_dict["Jf"]
    Jlength = obj_dict["Jlength"]
    Jlength2 = obj_dict["Jlength2"]
    Jcs = obj_dict["Jcs"]
    Jmscs = obj_dict["Jmscs"]
    Jls = obj_dict["Jls"]
    Jls_TF = obj_dict["Jls_TF"]
    Jccdist = obj_dict["Jccdist"]
    Jccdist2 = obj_dict["Jccdist2"]
    Jcsdist = obj_dict["Jcsdist"]
    linkNum = obj_dict["linkNum"]
    Jforce = obj_dict["Jforce"]
    Jforce2 = obj_dict["Jforce2"]
    Jtorque = obj_dict["Jtorque"]
    Jtorque2 = obj_dict["Jtorque2"]
    length_weight = weight_dict["length_weight"]
    curvature_weight = weight_dict["curvature_weight"]
    msc_weight = weight_dict["msc_weight"]
    msc_threshold = weight_dict["msc_threshold"]
    cc_weight = weight_dict["cc_weight"]
    cs_weight = weight_dict["cs_weight"]
    link_weight = weight_dict["link_weight"]
    force_weight = weight_dict["force_weight"]
    net_force_weight = weight_dict["net_force_weight"]
    torque_weight = weight_dict["torque_weight"]
    net_torque_weight = weight_dict["net_torque_weight"]

    # Set the overall objective degrees of freedom
    JF.x = dofs
    if psc_array is not None:
        # absolutely essential line that updates the PSC currents even though they are not directly optimized.
        psc_array.recompute_currents()
        # absolutely essential line if the PSCs do not have any dofs
        btot.Bfields[0].invalidate_cache()
    J = JF.J()
    grad = JF.dJ()

    # Get all the individual objective values and print them
    jf = Jf.J()
    length_val = length_weight * Jlength.J()
    length_val2 = length_weight * Jlength2.J()
    cc_val = cc_weight * Jccdist.J()
    cc_val2 = cc_weight * Jccdist.J()
    cs_val = cs_weight * Jcsdist.J()
    curvature_val = curvature_weight * sum(Jcs).J()
    msc_val = (
        msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs).J()
    )
    link_val = link_weight * linkNum.J()
    forces_val = Jforce.J()
    forces_val2 = Jforce2.J()
    torques_val = Jtorque.J()
    torques_val2 = Jtorque2.J()
    BdotN = np.mean(
        np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
    )
    BdotN_over_B = np.mean(
        np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
    ) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    cl_string2 = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    outstr += f", LenDipoles=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    valuestr += f", LenDipoles={length_val2:.2e}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    valuestr += f", ccObj2={cc_val2:.2e}"
    valuestr += f", csObj={cs_val:.2e}"
    valuestr += f", curvatureObj={curvature_val:.2e}"
    valuestr += f", mscObj={msc_val:.2e}"
    valuestr += f", Lk1Obj={link_val:.2e}"
    valuestr += f", forceObj={force_weight * forces_val:.2e}"
    valuestr += f", forceObj2={net_force_weight * forces_val2:.2e}"
    valuestr += f", torqueObj={torque_weight * torques_val:.2e}"
    valuestr += f", torqueObj2={net_torque_weight * torques_val2:.2e}"
    outstr += f", F={forces_val:.2e}"
    outstr += f", Fnet={forces_val2:.2e}"
    outstr += f", T={torques_val:.2e}"
    outstr += f", Tnet={torques_val2:.2e}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-C-Sep2={Jccdist2.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link Number = {linkNum.J()}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    print(valuestr, "\n")
    return J, grad


def save_coil_sets(btot, OUT_DIR, file_suffix, compute_forces_torques=True):
    """
    Save TF and dipole array coil sets together for a dipole array solution.
    Coils are saved together so that coils_to_vtk can compute forces and torques
    from all coils together.

    Args:
        btot: BiotSavart object
            The BiotSavart object containing (both) the coil sets.
        OUT_DIR: str
            The output directory.
        file_suffix: str
            The suffix for the output files.
        compute_forces_torques: bool
            Forwarded to coils_to_vtk. Force/torque computation is O(n_coils^2), so
            for large windowpane arrays (e.g. from a small inboard_radius in
            generate_curves) it can dominate the save time; set to False to skip it
            and just write coil geometry and currents (O(n_coils)).
    """
    from simsopt.field import coils_to_vtk

    coils_to_vtk(
        btot.Bfields[0].coils + btot.Bfields[1].coils,
        OUT_DIR + "coils" + file_suffix,
        close=True,
        compute_forces_torques=compute_forces_torques,
    )


def quaternion_from_axis_angle(axis, theta):
    """
    Compute a quaternion from a rotation axis and angle.
    Parameters:
        axis (array-like): A 3-element array representing the rotation axis.
        theta (float): The rotation angle in radians.
    Returns:
        np.array: The resulting quaternion as a 4-element array [q0, qi, qj, qk].
    """
    axis = axis / np.linalg.norm(axis)
    q0 = np.cos(theta / 2)
    q_vec = axis * np.sin(theta / 2)
    return np.array([q0, *q_vec])


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q = q1 * q2.
    Parameters:
        q1 (array-like): A 4-element array representing the first quaternion [q0, qi, qj, qk].
        q2 (array-like): A 4-element array representing the second quaternion [q0, qi, qj, qk].
    Returns:
        np.array: The resulting quaternion as a 4-element array [q0, qi, qj, qk].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rotate_vector(v, q):
    """
    Rotate a vector v using quaternion q.
    Parameters:
        v (array-like): A 3-element array representing the vector to rotate.
        q (array-like): A 4-element array representing the quaternion [q0, qi, qj, qk].
    Returns:
        np.array: The rotated vector.
    """
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])  # q* (conjugate)
    v_quat = np.array([0, *v])  # Convert v to quaternion form
    v_rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conjugate)
    return v_rotated[1:]  # Return the vector part


def compute_quaternion(normal, tangent):
    """
    Compute the quaternion to rotate the upward direction [0, 0, 1] to the given unit normal vector.
    Then, compute the change needed to align the x direction [1, 0, 0] to the desired tangent after the first rotation.

    Parameters:
        normal (array-like): A 3-element array representing the unit normal vector [n_x, n_y, n_z].
        tangent (array-like): A 3-element array representing the unit tangent vector [t_x, t_y, t_z].

    Returns:
        np.array: Quaternion as [q0, qi, qj, qk].
    """
    # Ensure input vectors are numpy arrays and normalized
    normal = np.asarray(normal)
    tangent = np.asarray(tangent)

    if not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("The input normal must be a unit vector.")
    if not np.isclose(np.linalg.norm(tangent), 1.0):
        raise ValueError("The input tangent must be a unit vector.")

    # Step 1: Rotate +z (upward) to normal
    upward = np.array([0.0, 0.0, 1.0])
    cos_theta1 = np.dot(upward, normal)
    axis1 = np.cross(upward, normal)

    if np.allclose(axis1, 0):
        q1 = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if np.allclose(normal, upward)
            else np.array([0.0, 1.0, 0.0, 0.0])
        )
    else:
        axis1 /= np.linalg.norm(axis1)
        theta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
        q1 = quaternion_from_axis_angle(axis1, theta1)

    # Step 2: Rotate initial x-direction using q1
    initial_x = np.array([1.0, 0.0, 0.0])
    rotated_x = rotate_vector(initial_x, q1)

    # Step 3: Rotate rotated_x to match desired tangent
    axis2 = np.cross(rotated_x, tangent)
    if np.linalg.norm(axis2) < 1e-8:
        q2 = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation needed
    else:
        axis2 /= np.linalg.norm(axis2)
        theta2 = np.arccos(np.clip(np.dot(rotated_x, tangent), -1.0, 1.0))
        q2 = quaternion_from_axis_angle(axis2, theta2)

    # Final quaternion (q2 * q1)
    q_final = quaternion_multiply(q2, q1)
    return q_final


# These functions compute the Fourier series coefficients for the superellipse


def rho(theta, a, b, n):
    """
    Compute the radius of a superellipse at angle theta.
    Parameters:
        theta: angle in radians
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        rho: radius at angle theta
    """
    return 1 / (abs(np.cos(theta) / a) ** (n) + abs(np.sin(theta) / b) ** (n)) ** (
        1 / (n)
    )


# Define Fourier coefficient integrals


def a_m(m, a, b, n):
    """
    Compute the Fourier coefficient a_m for a superellipse.
    Parameters:
        m: order of the Fourier coefficient
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        a_m: Fourier coefficient a_m
    """
    from scipy import integrate as spi

    def integrand(theta):
        return rho(theta, a, b, n) * np.cos(m * theta)

    if m == 0:
        return (1 / (2 * np.pi)) * spi.quad(integrand, 0, 2 * np.pi)[0]
    else:
        return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]


def b_m(m, a, b, n):
    """
    Compute the Fourier coefficient b_m for a superellipse.
    Parameters:
        m: order of the Fourier coefficient
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        b_m: Fourier coefficient b_m
    """
    from scipy import integrate as spi

    def integrand(theta):
        return rho(theta, a, b, n) * np.sin(m * theta)

    return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]


# Compute Fourier coefficients up to a given order


def compute_fourier_coeffs(max_order, a, b, n):
    """
    Compute Fourier coefficients for a superellipse.

    Parameters:
        max_order: maximum order of the Fourier series
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        coeffs: dictionary of Fourier coefficients
    """
    coeffs = {"a_m": [], "b_m": []}
    for m in range(max_order + 1):
        coeffs["a_m"].append(a_m(m, a, b, n))
        coeffs["b_m"].append(b_m(m, a, b, n))
    return coeffs


# Reconstruct the Fourier series approximation


def rho_fourier(theta, coeffs, max_order):
    """
    Reconstruct the Fourier series approximation of the superellipse.

    Parameters:
        theta: angle in radians
        coeffs: dictionary of Fourier coefficients
        max_order: maximum order of the Fourier series
    Returns:
        rho_approx: approximate radius at angle theta
    """
    rho_approx = coeffs["a_m"][0]
    for m in range(1, max_order + 1):
        rho_approx += coeffs["a_m"][m] * np.cos(m * theta) + coeffs["b_m"][m] * np.sin(
            m * theta
        )
    return rho_approx


# use this to evenly space coils on elliptical grid
# since evenly spaced in poloidal angle won't work
# must use quadrature for elliptic integral


def generate_even_arc_angles(a, b, ntheta):
    """
    Generate ntheta evenly spaced angles along the arc length of an ellipse.
    Parameters:
        a: semi-major axis of the ellipse
        b: semi-minor axis of the ellipse
        ntheta: number of angles to generate
    Returns:
        thetas: array of angles corresponding to the arc length
    """
    from scipy.integrate import quad
    from scipy.optimize import root_scalar

    def arc_length_diff(theta):
        return np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

    # Total arc length of the ellipse
    total_arc_length, _ = quad(arc_length_diff, 0, 2 * np.pi)
    arc_lengths = np.linspace(0, total_arc_length, ntheta, endpoint=False)

    def arc_length_to_theta(theta, s_target):
        s, _ = quad(arc_length_diff, 0, theta)
        return s - s_target

    # Solve for theta corresponding to each arc length
    thetas = np.zeros(ntheta)
    for i, s in enumerate(arc_lengths):
        if i != 0:
            result = root_scalar(
                arc_length_to_theta, args=(s,), bracket=[thetas[i - 1], 2 * np.pi]
            )
            thetas[i] = result.root
    return thetas


def generate_even_arc_angles_from_surface(
    winding_surface, phi_vals, ntheta, ndense=2000, theta_range=(0.0, 2 * np.pi)
):
    """
    Generate ntheta evenly spaced (by real 3D arc length) poloidal angles, using the
    actual winding_surface geometry rather than assuming an idealized elliptical cross
    section (as generate_even_arc_angles does). This matters for winding surfaces that
    are not axisymmetric, where the poloidal cross section's shape and size can vary
    with the toroidal angle.

    phi_vals may be a single toroidal angle or several. When several are given, a
    single cross section is not used on its own -- instead, at each point around the
    poloidal loop, whichever of the sampled cross sections has covered the least arc
    length so far is used, before continuing. This "worst-of-all-samples" arc-length
    function is always at least as tight as any individual cross section, since a
    winding surface with a non-axisymmetric cross section can pinch inward locally at
    some interior toroidal angle even if its *total* poloidal perimeter there is not
    the smallest of the sampled cross sections (a pinch in one region has to be offset
    by extra length somewhere else around the same loop). Using only the smallest-total
    cross section, or only the two ends phi=0 and phi=pi/nfp, can therefore still miss a
    local pinch elsewhere and place adjacent poloidal rings too close together; sampling
    several cross sections and taking this pointwise minimum guards against that.

    Parameters:
        winding_surface: surface to sample; its poloidal cross section(s) at phi_vals
                         are used to build the arc-length parameterization.
        phi_vals: toroidal angle (radians), or array of them, at which to take the
                  poloidal cross section(s).
        ntheta: number of angles to generate.
        ndense: number of points used to densely sample each cross section for building
                the arc-length parameterization.
        theta_range: (theta_min, theta_max) in radians over which theta_locs are
                     equispaced by arc length. Defaults to the full poloidal loop,
                     (0, 2*pi); a strict subrange is treated as an open arc (no
                     wraparound segment closing theta_max back to theta_min).
    Returns:
        theta_locs: array of ntheta poloidal angles (radians), evenly spaced by the
                    (possibly worst-of-all-samples) real arc length over theta_range.
        total_arc_length: the (possibly worst-of-all-samples) real arc length, in the
                          same units as winding_surface.gamma(), spanned by theta_range.
    """
    from scipy.interpolate import RegularGridInterpolator

    phi_vals = np.atleast_1d(phi_vals)
    phi_min = np.min(winding_surface.quadpoints_phi)
    phi_max = np.max(winding_surface.quadpoints_phi)
    theta_max = np.max(winding_surface.quadpoints_theta)

    theta_min_range, theta_max_range = theta_range
    if theta_min_range >= theta_max_range:
        raise ValueError(
            f"theta_range must satisfy theta_min < theta_max; got {theta_range}"
        )
    # ndense segments (ndense + 1 points), matching the resolution of the closed-loop
    # full-range case (ndense points wrapped around into ndense segments).
    theta_dense = np.linspace(theta_min_range, theta_max_range, ndense + 1, endpoint=True)
    # theta_range need not start at 0 (or even be nonnegative), so wrap into [0, 1)
    # via mod (a true periodic wrap) before clipping to the grid's own upper bound --
    # clipping alone would incorrectly map negative theta to the grid's theta=0
    # sample instead of its true periodic-equivalent location.
    theta_dense_norm = np.clip(np.mod(theta_dense / (2 * np.pi), 1.0), 0.0, theta_max)
    gamma_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gamma()[..., i],
            method="linear",
        )
        for i in range(3)
    ]

    # For each sampled cross section, get the segment lengths between consecutive
    # dense theta points along the open arc from theta_min to theta_max.
    seg_lengths_per_phi = []
    for phi_val in phi_vals:
        phi_norm = np.clip(phi_val / (2 * np.pi), phi_min, phi_max)
        pts = np.stack(
            [
                interp((np.full(theta_dense.shape[0], phi_norm), theta_dense_norm))
                for interp in gamma_interpolators
            ],
            axis=-1,
        )
        seg_lengths_per_phi.append(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    # Conservative (pointwise-minimum) segment lengths, then their cumulative sum.
    seg_lengths = np.min(seg_lengths_per_phi, axis=0)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_arc_length = cum_length[-1]

    arc_targets = np.linspace(0, total_arc_length, ntheta, endpoint=False)
    theta_locs = np.interp(arc_targets, cum_length, theta_dense)
    return theta_locs, total_arc_length


def generate_windowpane_array(
    winding_surface,
    inboard_radius,
    wp_fil_spacing,
    half_per_spacing,
    wp_n,
    numquadpoints=32,
    order=12,
    square=False,
    verbose=False,
):
    """
    Initialize an array of nwps_poloidal x nwps_toroidal planar windowpane coils on a winding surface
    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent
    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally tangent to the surface normal
                         assumed to be an elliptical cross section
        inboard_radius: radius of dipoles at inboard midplane - constant poloidally, will increase toroidally
        wp_fil_spacing: spacing wp filaments
        half_per_spacing: spacing between half period segments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle
               (see CurvePlanarFourier documentation), more for ellipse approximation
    Returns:
        base_wp_curves: list of initialized curves (half field period)
    """
    from simsopt.geo import CurvePlanarFourier
    from scipy.special import ellipe
    from scipy.interpolate import RegularGridInterpolator

    # Identify locations of windowpanes
    VV_a = winding_surface.get_rc(1, 0)
    VV_b = winding_surface.get_zs(1, 0)
    VV_R0 = winding_surface.get_rc(0, 0)
    arc_length = 4 * VV_a * ellipe(1 - (VV_b / VV_a) ** 2)
    nwps_poloidal = int(
        arc_length / (2 * inboard_radius + wp_fil_spacing)
    )  # figure out how many poloidal dipoles can fit for target radius
    Rpol = (
        arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2
    )  # adjust the poloidal length based off npol to fix filament distance
    theta_locs = generate_even_arc_angles(VV_a, VV_b, nwps_poloidal)
    nwps_toroidal = int(
        (
            np.pi / winding_surface.nfp * (VV_R0 - VV_a)
            - half_per_spacing
            + wp_fil_spacing
        )
        / (2 * inboard_radius + wp_fil_spacing)
    )
    if verbose:
        print(f"     Number of Toroidal Dipoles: {nwps_toroidal}")
        print(f"     Number of Poroidal Dipoles: {nwps_poloidal}")
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    # Get the actual bounds of the interpolation grid to avoid out-of-bounds errors
    phi_min = np.min(winding_surface.quadpoints_phi)
    phi_max = np.max(winding_surface.quadpoints_phi)
    theta_max = np.max(winding_surface.quadpoints_theta)
    unitn_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.unitnormal()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    gamma_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gamma()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    dgammadtheta_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gammadash2()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    # Initialize curves
    base_wp_curves = []
    for ii in range(nwps_poloidal):
        for jj in range(nwps_toroidal):
            theta_coil = theta_locs[ii]
            r = (
                VV_a
                * VV_b
                / np.sqrt(
                    (VV_b * np.cos(theta_coil)) ** 2 + (VV_a * np.sin(theta_coil)) ** 2
                )
            )
            if square:
                Rtor = r
            else:
                Rtor = (
                        np.pi / winding_surface.nfp * (VV_R0 + r * np.cos(theta_coil))
                        - half_per_spacing
                        - (nwps_toroidal - 1) * wp_fil_spacing
                    ) / (2 * nwps_toroidal)
            # Calculate toroidal angle of center of coil
            dphi = (half_per_spacing / 2 + Rtor) / (
                VV_R0 + r * np.cos(theta_coil)
            )  # need to add buffer in phi for gaps in panels
            phi_coil = dphi + jj * (2 * Rtor + wp_fil_spacing) / (
                VV_R0 + r * np.cos(theta_coil)
            )
            # Normalize coordinates to [0, 1) for interpolation, clamping to grid bounds to avoid out-of-bounds errors
            phi_norm = np.clip(phi_coil / (2 * np.pi), phi_min, phi_max)
            theta_norm = np.clip(theta_coil / (2 * np.pi), 0.0, theta_max)
            # Interpolate coil center and rotation vectors
            unitn_interp = np.stack(
                [interp((phi_norm, theta_norm)) for interp in unitn_interpolators],
                axis=-1,
            )
            gamma_interp = np.stack(
                [interp((phi_norm, theta_norm)) for interp in gamma_interpolators],
                axis=-1,
            )
            dgammadtheta_interp = np.stack(
                [
                    interp((phi_norm, theta_norm))
                    for interp in dgammadtheta_interpolators
                ],
                axis=-1,
            )
            curve = CurvePlanarFourier(numquadpoints, order)
            # dofs stored as: [rc(0), rc(1), ..., rc(order), rs(1), ..., rs(order), q0, qi, qj, qk, X, Y, Z]
            # Compute fourier coefficients for given super-ellipse
            coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)
            # Set rc(0) (constant term)
            curve.set("rc(0)", coeffs["a_m"][0])
            # Set rc(m) and rs(m) for m=1 to order
            for m in range(1, order + 1):
                curve.set(f"rc({m})", coeffs["a_m"][m])
                curve.set(f"rs({m})", coeffs["b_m"][m])
            # Align the coil normal with the surface normal and Rpol axis with dgamma/dtheta
            # Renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(
                unitn_interp / np.linalg.norm(unitn_interp),
                dgammadtheta_interp / np.linalg.norm(dgammadtheta_interp),
            )
            curve.set("q0", quaternion[0])
            curve.set("qi", quaternion[1])
            curve.set("qj", quaternion[2])
            curve.set("qk", quaternion[3])
            # Align the coil center with the winding surface gamma
            curve.set("X", gamma_interp[0])
            curve.set("Y", gamma_interp[1])
            curve.set("Z", gamma_interp[2])
            base_wp_curves.append(curve)
    return base_wp_curves


def generate_windowpane_wedge_array(
    winding_surface,
    inboard_radius,
    wp_fil_spacing,
    half_per_spacing,
    wp_n,
    numquadpoints=32,
    order=12,
    fill_wedges=True,
    verbose=False,
):
    """
    Initialize an array of fixed-size (square) planar windowpane coils on a winding surface,
    arranged in toroidal "wedges": sectors bounded by two R-Z planes of constant toroidal
    angle. Each wedge contains a poloidal strip of coils, spaced evenly around the poloidal
    cross section.

    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent.

    Wedges are spaced evenly in toroidal angle, calibrated using the tightest (inboard)
    poloidal location so that coils never overlap. Each wedge therefore owns a toroidal
    angular slab wide enough to fit exactly one coil at the inboard ring; away from the
    inboard side that same angular slab spans more physical distance. If fill_wedges is
    True, each poloidal ring fills its own slab with as many same-size coils (spaced evenly
    about the wedge center) as physically fit there, rather than leaving that extra room
    empty. This never lets coils spill into a neighboring wedge's slab.

    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally tangent to the surface normal
                         assumed to be an elliptical cross section
        inboard_radius: half-width of the (square) dipole coils, constant over the whole array
        wp_fil_spacing: spacing wp filaments
        half_per_spacing: spacing between half period segments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle
               (see CurvePlanarFourier documentation), more for square approximation
        fill_wedges: if True, pack additional coils into each wedge's poloidal ring
                     wherever the local toroidal slab has room for more than one
                     (e.g. on the outboard side), instead of leaving that room empty.
    Returns:
        base_wp_curves: list of initialized curves (half field period)
    """
    from simsopt.geo import CurvePlanarFourier
    from scipy.special import ellipe
    from scipy.interpolate import RegularGridInterpolator

    # Identify locations of windowpanes
    VV_a = winding_surface.get_rc(1, 0)
    VV_b = winding_surface.get_zs(1, 0)
    VV_R0 = winding_surface.get_rc(0, 0)
    arc_length = 4 * VV_a * ellipe(1 - (VV_b / VV_a) ** 2)
    nwps_poloidal = int(
        arc_length / (2 * inboard_radius + wp_fil_spacing)
    )  # figure out how many poloidal dipoles can fit for target radius
    Rpol = (
        arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2
    )  # adjust the poloidal length based off npol to fix filament distance
    Rtor = Rpol  # square coil: same physical size in both directions, everywhere
    theta_locs = generate_even_arc_angles(VV_a, VV_b, nwps_poloidal)

    # Wedges are spaced using the tightest (inboard) circumference, so a fixed toroidal
    # angle spacing never lets coils overlap regardless of poloidal location. This same
    # angular step is the width of the slab each wedge owns.
    R_inboard = VV_R0 - VV_a
    half_period_length_ref = np.pi / winding_surface.nfp * R_inboard
    usable_length_ref = half_period_length_ref - half_per_spacing
    nwps_toroidal = max(
        1,
        int((usable_length_ref + wp_fil_spacing) / (2 * Rtor + wp_fil_spacing)),
    )
    if nwps_toroidal == 1:
        wedge_pitch = 0.0
    else:
        # Stretch the wedges to equispace across the entire usable half period at
        # the inboard reference (rather than packing tight at wp_fil_spacing and
        # leaving the floor-rounding leftover as an unused gap at the far end,
        # near the next symmetry sector); resulting wedge-to-wedge spacing may
        # exceed wp_fil_spacing, but is never smaller.
        wedge_spacing = (usable_length_ref - nwps_toroidal * 2 * Rtor) / (
            nwps_toroidal - 1
        )
        wedge_pitch = 2 * Rtor + wedge_spacing
    wedge_angular_width = wedge_pitch / R_inboard
    dphi = (half_per_spacing / 2 + Rtor) / R_inboard
    phi_locs = dphi + np.arange(nwps_toroidal) * wedge_angular_width

    if verbose:
        print(f"     Number of Wedges (Toroidal): {nwps_toroidal}")
        print(f"     Number of Poroidal Dipoles per wedge: {nwps_poloidal}")
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    # Get the actual bounds of the interpolation grid to avoid out-of-bounds errors
    phi_min = np.min(winding_surface.quadpoints_phi)
    phi_max = np.max(winding_surface.quadpoints_phi)
    theta_max = np.max(winding_surface.quadpoints_theta)
    unitn_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.unitnormal()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    gamma_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gamma()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    dgammadtheta_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gammadash2()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    # Every coil has the same size, so the superellipse Fourier coefficients only
    # need to be computed once.
    coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)

    # Initialize curves: outer loop over toroidal wedges, inner loop over the
    # poloidal strip within a wedge (all at the same nominal toroidal angle,
    # unless fill_wedges packs more than one coil into a ring's local slab).
    base_wp_curves = []
    for jj in range(nwps_toroidal):
        phi_center = phi_locs[jj]
        for ii in range(nwps_poloidal):
            theta_coil = theta_locs[ii]
            r = (
                VV_a
                * VV_b
                / np.sqrt(
                    (VV_b * np.cos(theta_coil)) ** 2 + (VV_a * np.sin(theta_coil)) ** 2
                )
            )
            local_circumference = VV_R0 + r * np.cos(theta_coil)
            local_slab_width = wedge_angular_width * local_circumference
            # Reserve half a filament spacing on each side of the slab facing a
            # neighboring wedge, so this wedge's coils never end up closer than
            # wp_fil_spacing to the neighbor's coils (which reserve the same
            # buffer on their own side of the shared boundary). But the near
            # side of the first wedge and the far side of the last wedge face
            # the global phi=0/half-period mirror boundary instead of another
            # wedge, and need the larger buffer used elsewhere for that
            # boundary (half_per_spacing/2 + Rtor), not just wp_fil_spacing/2.
            mirror_buffer = half_per_spacing / 2 + Rtor
            buffer_near = mirror_buffer if jj == 0 else wp_fil_spacing / 2
            buffer_far = (
                mirror_buffer if jj == nwps_toroidal - 1 else wp_fil_spacing / 2
            )
            lower_bound = -local_slab_width / 2 + buffer_near
            upper_bound = local_slab_width / 2 - buffer_far
            usable_slab_width = upper_bound - lower_bound
            if fill_wedges:
                # How many coils of this fixed size actually fit in this ring's
                # usable share of the wedge's angular slab.
                n_fill = max(
                    1,
                    int(
                        (usable_slab_width + wp_fil_spacing)
                        / (2 * Rtor + wp_fil_spacing)
                    ),
                )
            else:
                n_fill = 1
            if n_fill == 1:
                phi_offsets = np.array([(lower_bound + upper_bound) / 2]) / local_circumference
            else:
                # Spread all n_fill coils out to use the entire usable slab
                # width (rather than packing them tight and wasting the
                # leftover as unused edge margin), while still keeping at
                # least wp_fil_spacing between adjacent coils.
                coil_spacing = (usable_slab_width - n_fill * 2 * Rtor) / (n_fill - 1)
                offsets_phys = lower_bound + Rtor + np.arange(n_fill) * (
                    2 * Rtor + coil_spacing
                )
                phi_offsets = offsets_phys / local_circumference
            for kk, offset in enumerate(phi_offsets):
                phi_coil = phi_center + offset
                # Normalize coordinates to [0, 1) for interpolation, clamping to grid bounds to avoid out-of-bounds errors
                phi_norm = np.clip(phi_coil / (2 * np.pi), phi_min, phi_max)
                theta_norm = np.clip(theta_coil / (2 * np.pi), 0.0, theta_max)
                # Interpolate coil center and rotation vectors
                unitn_interp = np.stack(
                    [interp((phi_norm, theta_norm)) for interp in unitn_interpolators],
                    axis=-1,
                )
                gamma_interp = np.stack(
                    [interp((phi_norm, theta_norm)) for interp in gamma_interpolators],
                    axis=-1,
                )
                dgammadtheta_interp = np.stack(
                    [
                        interp((phi_norm, theta_norm))
                        for interp in dgammadtheta_interpolators
                    ],
                    axis=-1,
                )
                curve = CurvePlanarFourier(numquadpoints, order)
                # dofs stored as: [rc(0), rc(1), ..., rc(order), rs(1), ..., rs(order), q0, qi, qj, qk, X, Y, Z]
                # Set rc(0) (constant term)
                curve.set("rc(0)", coeffs["a_m"][0])
                # Set rc(m) and rs(m) for m=1 to order
                for m in range(1, order + 1):
                    curve.set(f"rc({m})", coeffs["a_m"][m])
                    curve.set(f"rs({m})", coeffs["b_m"][m])
                # Align the coil normal with the surface normal and Rpol axis with dgamma/dtheta
                # Renormalize the vector because interpolation can slightly modify its norm
                quaternion = compute_quaternion(
                    unitn_interp / np.linalg.norm(unitn_interp),
                    dgammadtheta_interp / np.linalg.norm(dgammadtheta_interp),
                )
                curve.set("q0", quaternion[0])
                curve.set("qi", quaternion[1])
                curve.set("qj", quaternion[2])
                curve.set("qk", quaternion[3])
                # Align the coil center with the winding surface gamma
                curve.set("X", gamma_interp[0])
                curve.set("Y", gamma_interp[1])
                curve.set("Z", gamma_interp[2])
                base_wp_curves.append(curve)
    return base_wp_curves


def _winding_surface_is_axisymmetric(winding_surface, tol=1e-10):
    """
    Check whether winding_surface's shape is independent of the toroidal angle, i.e.
    only its n=0 Fourier modes (the middle column of rc/zs/rs/zc, at column index
    winding_surface.ntor) are nonzero.
    """
    if winding_surface.ntor == 0:
        return True
    n0 = winding_surface.ntor
    non_axisym_coeffs = [
        np.delete(winding_surface.rc, n0, axis=1),
        np.delete(winding_surface.zs, n0, axis=1),
    ]
    for attr in ("rs", "zc"):
        coeffs = getattr(winding_surface, attr, None)
        if coeffs is not None and coeffs.size:
            non_axisym_coeffs.append(np.delete(coeffs, n0, axis=1))
    return all(np.all(np.abs(c) < tol) for c in non_axisym_coeffs)


def generate_windowpane_ring_array(
    winding_surface,
    inboard_radius,
    wp_fil_spacing,
    half_per_spacing,
    wp_n,
    numquadpoints=32,
    order=12,
    theta_range=(0.0, 2 * np.pi),
    verbose=False,
):
    """
    Initialize an array of fixed-size (square) planar windowpane coils on an
    axisymmetric winding surface (one whose shape does not depend on the toroidal
    angle -- raises ValueError otherwise; use generate_windowpane_metric_ring_array
    for a non-axisymmetric winding surface).

    Unlike generate_windowpane_wedge_array (which fixes a small set of toroidal angles
    shared by every poloidal ring, then tries to pack extra coils into each ring's share
    of that fixed slab), this builds the array ring by ring in the poloidal direction and
    lets each ring independently tile the toroidal direction:

    (a) The poloidal ring positions (theta_locs) and the fixed coil size (Rpol = Rtor,
        a square) are built from the actual winding surface geometry (not an idealized
        ellipse), using the real poloidal arc length of the surface's cross section (see
        generate_even_arc_angles_from_surface). Because the surface is axisymmetric,
        this cross section -- and so the arc length, Rpol, and theta_locs derived from
        it -- is the same at every toroidal angle, so a single reference cross section
        (phi=0) fully determines the poloidal ring layout. theta_locs are equispaced by
        real arc length -- over theta_range, which defaults to the full poloidal loop --
        with exactly the minimum required spacing, 2*inboard_radius + wp_fil_spacing,
        between every adjacent pair of rings.
    (b) For each ring, the first coil is placed so that its near edge sits at least
        half_per_spacing/2 from the phi=0 stellarator-symmetric mirror plane, using the
        ring's local circumference (the same at phi=0 and phi=pi/nfp, again because the
        surface is axisymmetric). From there, the ring is filled toroidally with as many
        further coils as fit (equispaced by real arc length, stretched to use the entire
        available length, so the spacing between neighbors is always at least
        wp_fil_spacing but may exceed it) before running into the same buffer at the far
        end of the half period. Since the local circumference varies with poloidal
        angle, different rings can and will end up with different toroidal coil counts
        (more on the outboard side, fewer on the inboard side).

    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent.

    Parameters:
        winding_surface: axisymmetric surface upon which to place the coils, with coil
                         plane locally tangent to the surface normal
        inboard_radius: half-width of the (square) dipole coils, constant over the whole array
        wp_fil_spacing: spacing wp filaments
        half_per_spacing: spacing between half period segments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle
               (see CurvePlanarFourier documentation), more for square approximation
        theta_range: (theta_min, theta_max) in radians defining the range of poloidal
                     angle over which the poloidal ring is equispaced by real arc
                     length. Defaults to the full poloidal loop, (0, 2*pi); a strict
                     subrange restricts the ring to that arc only (e.g. to skip the
                     inboard side).
    Returns:
        base_wp_curves: list of initialized curves (half field period)
    """
    from simsopt.geo import CurvePlanarFourier
    from scipy.interpolate import RegularGridInterpolator

    if not _winding_surface_is_axisymmetric(winding_surface):
        raise ValueError(
            "generate_windowpane_ring_array requires an axisymmetric winding_surface "
            "(one whose shape does not depend on the toroidal angle, i.e. all Fourier "
            "modes with n != 0 are zero). For a non-axisymmetric winding surface, use "
            "generate_windowpane_metric_ring_array instead."
        )

    # Identify locations of windowpanes. Since the surface is axisymmetric, its
    # poloidal cross section (and so the real poloidal arc length used to build
    # theta_locs, Rpol, and nwps_poloidal) is the same at every toroidal angle, so a
    # single reference cross section (phi=0) is exact -- no need to sample several
    # toroidal angles as generate_windowpane_metric_ring_array does for a general
    # surface.
    arc_length_ref_phi = 0.0
    _, arc_length = generate_even_arc_angles_from_surface(
        winding_surface, arc_length_ref_phi, 2, theta_range=theta_range
    )
    nwps_poloidal = int(
        arc_length / (2 * inboard_radius + wp_fil_spacing)
    )  # figure out how many poloidal dipoles can fit for target radius
    Rpol = (
        arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2
    )  # adjust the poloidal length based off npol to fix filament distance
    Rtor = Rpol  # square coil: same physical size in both directions, everywhere
    theta_locs, _ = generate_even_arc_angles_from_surface(
        winding_surface, arc_length_ref_phi, nwps_poloidal, theta_range=theta_range
    )

    if verbose:
        print(f"     Number of Poroidal Dipoles: {nwps_poloidal}")
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    # Get the actual bounds of the interpolation grid to avoid out-of-bounds errors
    phi_min = np.min(winding_surface.quadpoints_phi)
    phi_max = np.max(winding_surface.quadpoints_phi)
    theta_max = np.max(winding_surface.quadpoints_theta)
    unitn_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.unitnormal()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    gamma_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gamma()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    dgammadtheta_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gammadash2()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    # Every coil has the same size, so the superellipse Fourier coefficients only
    # need to be computed once.
    coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)

    # Initialize curves: for each poloidal ring, independently tile the toroidal
    # direction, starting a fixed physical buffer away from the phi=0 mirror plane.
    base_wp_curves = []
    toroidal_counts = []
    phi_start_norm = np.clip(0.0, phi_min, phi_max)
    for ii in range(nwps_poloidal):
        theta_coil = theta_locs[ii]
        # theta_coil may be negative when theta_range doesn't start at 0, so wrap
        # into [0, 1) via mod before clipping to the grid's own upper bound.
        theta_norm = np.clip(np.mod(theta_coil / (2 * np.pi), 1.0), 0.0, theta_max)
        # Since the surface is axisymmetric, the local circumference is the same at
        # phi=0, phi=pi/nfp, or anywhere else along this ring, so it can be read off
        # at a single reference toroidal angle.
        gamma_at_phi0 = np.array(
            [interp((phi_start_norm, theta_norm)) for interp in gamma_interpolators]
        )
        local_circumference = np.hypot(gamma_at_phi0[0], gamma_at_phi0[1])
        # (a) buffer the first coil in this ring from the phi=0 mirror plane; since
        # dphi * local_circumference = half_per_spacing/2 + Rtor for every ring, the
        # physical buffer is the same everywhere, not just at the inboard ring.
        dphi = (half_per_spacing / 2 + Rtor) / local_circumference
        # (b) how many coils (with at least wp_fil_spacing between them, and the same
        # buffer at the far/half-period-boundary end) fit toroidally in this ring
        local_half_period_length = np.pi / winding_surface.nfp * local_circumference
        usable_length = local_half_period_length - half_per_spacing
        nwps_toroidal_local = max(
            1,
            int((usable_length + wp_fil_spacing) / (2 * Rtor + wp_fil_spacing)),
        )
        toroidal_counts.append(nwps_toroidal_local)
        if nwps_toroidal_local == 1:
            coil_pitch = 0.0
        else:
            # Stretch the coils to equispace across the entire usable length
            # (rather than packing tight at wp_fil_spacing and leaving the
            # floor-rounding leftover as an unused gap at the far end); the
            # resulting spacing between coils may exceed wp_fil_spacing, but
            # is never smaller.
            coil_spacing = (
                usable_length - nwps_toroidal_local * 2 * Rtor
            ) / (nwps_toroidal_local - 1)
            coil_pitch = 2 * Rtor + coil_spacing
        for jj in range(nwps_toroidal_local):
            phi_coil = dphi + jj * coil_pitch / local_circumference
            # Normalize coordinates to [0, 1) for interpolation, clamping to grid bounds to avoid out-of-bounds errors
            phi_norm = np.clip(phi_coil / (2 * np.pi), phi_min, phi_max)
            # Interpolate coil center and rotation vectors
            unitn_interp = np.stack(
                [interp((phi_norm, theta_norm)) for interp in unitn_interpolators],
                axis=-1,
            )
            gamma_interp = np.stack(
                [interp((phi_norm, theta_norm)) for interp in gamma_interpolators],
                axis=-1,
            )
            dgammadtheta_interp = np.stack(
                [
                    interp((phi_norm, theta_norm))
                    for interp in dgammadtheta_interpolators
                ],
                axis=-1,
            )
            curve = CurvePlanarFourier(numquadpoints, order)
            # dofs stored as: [rc(0), rc(1), ..., rc(order), rs(1), ..., rs(order), q0, qi, qj, qk, X, Y, Z]
            # Set rc(0) (constant term)
            curve.set("rc(0)", coeffs["a_m"][0])
            # Set rc(m) and rs(m) for m=1 to order
            for m in range(1, order + 1):
                curve.set(f"rc({m})", coeffs["a_m"][m])
                curve.set(f"rs({m})", coeffs["b_m"][m])
            # Align the coil normal with the surface normal and Rpol axis with dgamma/dtheta
            # Renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(
                unitn_interp / np.linalg.norm(unitn_interp),
                dgammadtheta_interp / np.linalg.norm(dgammadtheta_interp),
            )
            curve.set("q0", quaternion[0])
            curve.set("qi", quaternion[1])
            curve.set("qj", quaternion[2])
            curve.set("qk", quaternion[3])
            # Align the coil center with the winding surface gamma
            curve.set("X", gamma_interp[0])
            curve.set("Y", gamma_interp[1])
            curve.set("Z", gamma_interp[2])
            base_wp_curves.append(curve)
    if verbose:
        print(
            f"     Number of Toroidal Dipoles per ring (min/max): "
            f"{min(toroidal_counts)}/{max(toroidal_counts)}"
        )
    return base_wp_curves


def generate_windowpane_metric_ring_array(
    winding_surface,
    inboard_radius,
    wp_fil_spacing,
    half_per_spacing,
    wp_n,
    numquadpoints=32,
    order=12,
    theta_range=(0.0, 2 * np.pi),
    verbose=False,
):
    """
    Initialize an array of fixed-size (square) planar windowpane coils on a winding
    surface, using the surface's first fundamental form (the local metric, built from
    gammadash1 and gammadash2) to size and space coils, instead of approximating with an
    idealized elliptical cross section or a cylindrical R*dphi relation.

    This is the most direct generalization of generate_windowpane_ring_array to
    genuinely non-axisymmetric winding surfaces, and reduces to the exact same result
    for an axisymmetric surface: there, |dX/dphi| is exactly the cylindrical radius
    R(theta) (independent of phi) and |dX/dtheta| is likewise independent of phi, so the
    multi-cross-section handling below becomes a no-op and everything collapses to the
    same circumference/arc-length formulas generate_windowpane_ring_array uses.

    (a) Poloidal ring positions (theta_locs) and coil size (Rpol = Rtor) are built from
        the poloidal-direction metric coefficient sqrt(G) = |dX/dtheta|, sampled at
        several toroidal angles across the half period and combined by taking the
        pointwise-tightest (smallest) local speed at each poloidal angle -- since a
        non-axisymmetric surface can pinch inward locally at some interior toroidal
        angle without that showing up as a smaller *total* perimeter at either mirror
        plane. This differs from generate_windowpane_ring_array only in using the exact
        analytic metric rather than finite-differencing sampled points.
    (b) For each ring, the toroidal-direction metric coefficient sqrt(E) = |dX/dphi| is
        integrated exactly along *that ring's own theta* from phi=0 to phi=pi/nfp,
        giving the true arc length of the curve the coils are actually placed along --
        rather than approximating it from the circumference at only the phi=0/phi=pi/nfp
        mirror planes. Coils are then placed at equal *arc length* intervals along this
        curve (stretched to use the full usable length, as in generate_windowpane_ring_array),
        buffered by half_per_spacing/2 + Rtor of real arc length from each mirror plane.

    As with generate_windowpane_ring_array, poloidal rings are equispaced by real arc
    length -- over theta_range, which defaults to the full poloidal loop -- with
    exactly the minimum required spacing (2*inboard_radius + wp_fil_spacing) between
    every adjacent pair; within a ring, toroidal coils are equispaced by real arc
    length with at least wp_fil_spacing between every adjacent pair (more, if the
    available length doesn't divide evenly).

    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent.

    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally
                         tangent to the surface normal.
        inboard_radius: half-width of the (square) dipole coils, constant over the whole array
        wp_fil_spacing: spacing wp filaments
        half_per_spacing: spacing between half period segments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle
               (see CurvePlanarFourier documentation), more for square approximation
        theta_range: (theta_min, theta_max) in radians defining the range of poloidal
                     angle over which the poloidal ring is equispaced by real arc
                     length. Defaults to the full poloidal loop, (0, 2*pi); a strict
                     subrange restricts the ring to that arc only (e.g. to skip the
                     inboard side).
    Returns:
        base_wp_curves: list of initialized curves (half field period)
    """
    from simsopt.geo import CurvePlanarFourier
    from scipy.interpolate import RegularGridInterpolator

    nfp = winding_surface.nfp
    phi_end = np.pi / nfp
    phi_min = np.min(winding_surface.quadpoints_phi)
    phi_max = np.max(winding_surface.quadpoints_phi)
    theta_max = np.max(winding_surface.quadpoints_theta)

    unitn_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.unitnormal()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    gamma_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gamma()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    # dX/dtheta: used both for the poloidal metric coefficient sqrt(G) = |dX/dtheta|
    # and (its direction) for aligning each coil's Rpol axis, as in the other functions.
    dgammadtheta_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gammadash2()[..., i],
            method="linear",
        )
        for i in range(3)
    ]
    # dX/dphi: only used for the toroidal metric coefficient sqrt(E) = |dX/dphi|.
    dgammadphi_interpolators = [
        RegularGridInterpolator(
            (winding_surface.quadpoints_phi, winding_surface.quadpoints_theta),
            winding_surface.gammadash1()[..., i],
            method="linear",
        )
        for i in range(3)
    ]

    def speed_theta(phi_val, theta_vals):
        # sqrt(G) = |dX/dtheta_radians| at fixed phi_val, over an array of theta
        # values. gammadash2 is dX/dtheta_normalized (quadpoints_theta runs over
        # [0, 1) for a full poloidal turn), so it must be divided by 2*pi to get the
        # derivative with respect to theta in radians.
        # theta_vals may be negative (a caller-supplied theta_range need not start
        # at 0), so wrap into [0, 1) via mod (a true periodic wrap) before clipping
        # to the grid's own upper bound -- clipping alone would incorrectly map
        # negative theta to the grid's theta=0 sample instead of its true
        # periodic-equivalent location.
        phi_norm = np.full(theta_vals.shape, np.clip(phi_val / (2 * np.pi), phi_min, phi_max))
        theta_norm = np.clip(np.mod(theta_vals / (2 * np.pi), 1.0), 0.0, theta_max)
        dxdtheta = np.stack(
            [interp((phi_norm, theta_norm)) for interp in dgammadtheta_interpolators],
            axis=-1,
        )
        return np.linalg.norm(dxdtheta, axis=-1) / (2 * np.pi)

    def speed_phi(theta_val, phi_vals):
        # sqrt(E) = |dX/dphi_radians| at fixed theta_val, over an array of phi values.
        # gammadash1 is dX/dphi_normalized, so likewise divide by 2*pi. theta_val (a
        # ring's theta_coil) may be negative -- see the mod comment in speed_theta.
        phi_norm = np.clip(phi_vals / (2 * np.pi), phi_min, phi_max)
        theta_norm = np.full(
            phi_vals.shape, np.clip(np.mod(theta_val / (2 * np.pi), 1.0), 0.0, theta_max)
        )
        dxdphi = np.stack(
            [interp((phi_norm, theta_norm)) for interp in dgammadphi_interpolators],
            axis=-1,
        )
        return np.linalg.norm(dxdphi, axis=-1) / (2 * np.pi)

    # (a) Poloidal ring layout, from the metric's theta-direction coefficient,
    # conservatively combined across several toroidal cross sections (a no-op for an
    # axisymmetric surface, where sqrt(G) doesn't depend on phi at all). theta_range
    # is treated as an open arc from theta_min to theta_max (no wraparound segment
    # closing theta_max back to theta_min); the default (0, 2*pi) covers the same
    # physical loop as a closed ring, since theta=0 and theta=2*pi are the same point.
    theta_min_range, theta_max_range = theta_range
    if theta_min_range >= theta_max_range:
        raise ValueError(
            f"theta_range must satisfy theta_min < theta_max; got {theta_range}"
        )
    ndense_theta = 2000
    # ndense_theta segments (ndense_theta + 1 points), matching the resolution of the
    # closed-loop full-range case (ndense_theta points wrapped around into
    # ndense_theta segments).
    theta_dense = np.linspace(
        theta_min_range, theta_max_range, ndense_theta + 1, endpoint=True
    )
    n_phi_samples = 17
    phi_samples = np.linspace(0, phi_end, n_phi_samples)
    speeds = np.array([speed_theta(phi_val, theta_dense) for phi_val in phi_samples])
    conservative_speed_theta = np.min(speeds, axis=0)
    seg_lengths = (
        0.5
        * (conservative_speed_theta[:-1] + conservative_speed_theta[1:])
        * np.diff(theta_dense)
    )
    cum_theta_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    arc_length = cum_theta_length[-1]
    nwps_poloidal = int(
        arc_length / (2 * inboard_radius + wp_fil_spacing)
    )  # figure out how many poloidal dipoles can fit for target radius
    Rpol = (
        arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2
    )  # adjust the poloidal length based off npol to fix filament distance
    Rtor = Rpol  # square coil: same physical size in both directions, everywhere
    arc_targets_theta = np.linspace(0, arc_length, nwps_poloidal, endpoint=False)
    theta_locs = np.interp(arc_targets_theta, cum_theta_length, theta_dense)

    if verbose:
        print(f"     Number of Poroidal Dipoles: {nwps_poloidal}")
    # Every coil has the same size, so the superellipse Fourier coefficients only
    # need to be computed once.
    coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)

    # Initialize curves: for each poloidal ring, integrate the metric exactly along
    # that ring's own theta to tile the toroidal direction.
    ndense_phi = 2000
    phi_dense = np.linspace(0, phi_end, ndense_phi)
    base_wp_curves = []
    toroidal_counts = []
    for ii in range(nwps_poloidal):
        theta_coil = theta_locs[ii]
        speeds_phi = speed_phi(theta_coil, phi_dense)
        seg_phi = 0.5 * (speeds_phi[:-1] + speeds_phi[1:]) * np.diff(phi_dense)
        cum_phi_length = np.concatenate([[0.0], np.cumsum(seg_phi)])
        total_ring_arc_length = cum_phi_length[-1]

        # (b) how many coils (with at least wp_fil_spacing between them, and a
        # half_per_spacing/2 + Rtor buffer of real arc length at each mirror plane)
        # fit toroidally in this ring
        usable_length = total_ring_arc_length - half_per_spacing
        nwps_toroidal_local = max(
            1,
            int((usable_length + wp_fil_spacing) / (2 * Rtor + wp_fil_spacing)),
        )
        toroidal_counts.append(nwps_toroidal_local)
        if nwps_toroidal_local == 1:
            arc_targets_phi = np.array([half_per_spacing / 2 + Rtor])
        else:
            # Stretch the coils to equispace across the entire usable length
            # (rather than packing tight at wp_fil_spacing and leaving the
            # floor-rounding leftover as an unused gap at the far end); the
            # resulting spacing between coils may exceed wp_fil_spacing, but
            # is never smaller.
            coil_spacing = (
                usable_length - nwps_toroidal_local * 2 * Rtor
            ) / (nwps_toroidal_local - 1)
            coil_pitch = 2 * Rtor + coil_spacing
            arc_targets_phi = (half_per_spacing / 2 + Rtor) + np.arange(
                nwps_toroidal_local
            ) * coil_pitch
        phi_coils = np.interp(arc_targets_phi, cum_phi_length, phi_dense)

        # theta_coil may be negative when theta_range doesn't start at 0, so wrap
        # into [0, 1) via mod before clipping to the grid's own upper bound.
        theta_norm = np.clip(np.mod(theta_coil / (2 * np.pi), 1.0), 0.0, theta_max)
        for phi_coil in phi_coils:
            # Normalize coordinates to [0, 1) for interpolation, clamping to grid bounds to avoid out-of-bounds errors
            phi_norm = np.clip(phi_coil / (2 * np.pi), phi_min, phi_max)
            # Interpolate coil center and rotation vectors
            unitn_interp = np.stack(
                [interp((phi_norm, theta_norm)) for interp in unitn_interpolators],
                axis=-1,
            )
            gamma_interp = np.stack(
                [interp((phi_norm, theta_norm)) for interp in gamma_interpolators],
                axis=-1,
            )
            dgammadtheta_interp = np.stack(
                [
                    interp((phi_norm, theta_norm))
                    for interp in dgammadtheta_interpolators
                ],
                axis=-1,
            )
            curve = CurvePlanarFourier(numquadpoints, order)
            # dofs stored as: [rc(0), rc(1), ..., rc(order), rs(1), ..., rs(order), q0, qi, qj, qk, X, Y, Z]
            # Set rc(0) (constant term)
            curve.set("rc(0)", coeffs["a_m"][0])
            # Set rc(m) and rs(m) for m=1 to order
            for m in range(1, order + 1):
                curve.set(f"rc({m})", coeffs["a_m"][m])
                curve.set(f"rs({m})", coeffs["b_m"][m])
            # Align the coil normal with the surface normal and Rpol axis with dgamma/dtheta
            # Renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(
                unitn_interp / np.linalg.norm(unitn_interp),
                dgammadtheta_interp / np.linalg.norm(dgammadtheta_interp),
            )
            curve.set("q0", quaternion[0])
            curve.set("qi", quaternion[1])
            curve.set("qj", quaternion[2])
            curve.set("qk", quaternion[3])
            # Align the coil center with the winding surface gamma
            curve.set("X", gamma_interp[0])
            curve.set("Y", gamma_interp[1])
            curve.set("Z", gamma_interp[2])
            base_wp_curves.append(curve)
    if verbose:
        print(
            f"     Number of Toroidal Dipoles per ring (min/max): "
            f"{min(toroidal_counts)}/{max(toroidal_counts)}"
        )
    return base_wp_curves


def generate_tf_array(
    winding_surface,
    ntf,
    TF_R0,
    TF_a,
    TF_b,
    fixed_geo_tfs=False,
    planar_fourier_tfs=True,
    elliptical_tfs=False,
    order=6,
    numquadpoints=32,
):
    """
    Initialize an array of planar toroidal field coils over a half field period
    Parameters:
        winding_surface: surface upon which to obtain field periodicity/stellarator symmetry for the coils
        ntf: number of TF coils per half field period
        TF_R0: major radius of the TF coils
        TF_a: minor radius of the TF coils (in R direction)
        TF_b: minor radius of the TF coils (in Z direction)
        fixed_geo_tfs: whether to fix the geometric degrees of freedom of the TF coils
        planar_tfs: whether to use planar TF coils
        order: order of the Fourier series for the TF coils
        numquadpoints: number of quadrature points representing each coil
    Returns:
        base_tf_curves: list of initialized curves (half field period)
    """
    from simsopt.geo import create_equally_spaced_curves

    if not fixed_geo_tfs:
        assert not (planar_fourier_tfs and elliptical_tfs), "Only one of planar and elliptical tfs can be true. "
        if planar_fourier_tfs:
            # try:
            from simsopt.geo import create_equally_spaced_planar_curves

            base_tf_curves = create_equally_spaced_planar_curves(
                ntf,
                winding_surface.nfp,
                stellsym=winding_surface.stellsym,
                R0=TF_R0,
                R1=TF_a,
                order=order,
                numquadpoints=numquadpoints,
            )
            # except ImportError:
            #     raise ImportError(
            #         "Need to be on the windowpane branch with the correct TF curve class to unfix TF geometry"
            #     )
        elif elliptical_tfs:
            try:
                from simsopt.geo import create_equally_spaced_cylindrical_curves

                base_tf_curves = create_equally_spaced_cylindrical_curves(
                    ntf,
                    winding_surface.nfp,
                    stellsym=winding_surface.stellsym,
                    R0=TF_R0,
                    a=TF_a,
                    b=TF_b,
                    numquadpoints=numquadpoints,
                )
            except ImportError:
                raise ImportError(
                    "Need to be on the windowpane branch with the correct TF curve class to unfix TF geometry"
                )
        else:
            base_tf_curves = create_equally_spaced_curves(
                ncurves=ntf,
                nfp=winding_surface.nfp,
                stellsym=winding_surface.stellsym,
                R0=TF_R0,
                R1=TF_a,
                order=order,
                numquadpoints=numquadpoints,
            )
    else:
        # order=1 is fine for elliptical
        base_tf_curves = create_equally_spaced_curves(
            ncurves=ntf,
            nfp=winding_surface.nfp,
            stellsym=winding_surface.stellsym,
            R0=TF_R0,
            R1=TF_a,
            order=order,
            numquadpoints=numquadpoints,
        )
        # add this for elliptical TF coils - keep same ellipticity as VV
        for c in base_tf_curves:
            c.fix_all()
            c.set(
                "zs(1)", -TF_b
            )  # see create_equally_spaced_curves doc for minus sign info

    return base_tf_curves


def generate_curves(
    surf,
    VV,
    planar_fourier_tfs=False,
    ellptical_tfs=False,
    outdir="",
    inboard_radius=0.8,
    wp_fil_spacing=0.75,
    half_per_spacing=0.75,
    wp_n=2,
    numquadpoints=32,
    order_wp=12,
    order_tf=6,
    verbose=True,
    fixed_geo_tfs=False,
    wp_layout="metric_ring",
    theta_range=None,
    tf_init_fac=4,
    ntf=3,
):
    """
    Generate the curves for the winding surface and TF coils.

    Args:
        surf: SurfaceRZFourier object
            The plasma boundary surface.
        VV: SurfaceRZFourier object
            The vacuum vessel surface to put the dipole coils on.
        planar_tfs: bool
            Whether to use planar TF coils.
        outdir: str
            The output directory for the generated curves.
        inboard_radius: float
            The radius of the inboard midplane of the dipole coils.
        wp_fil_spacing: float
            The spacing between the filaments of the dipole coils.
        half_per_spacing: float
            The spacing between the half period segments of the dipole coils.
        wp_n: float
            The value of n for the superellipse.
        numquadpoints: int
            The number of quadrature points for the dipole coils.
        order: int
            The order of the Fourier series for the dipole coils.
        verbose: bool
            Whether to print verbose output.
        fixed_geo_tfs: bool
            Whether to fix the geometric degrees of freedom of the TF coils.
        wp_layout: str
            Which windowpane array generator to use for the dipole coils:
                "jake": generate_windowpane_array with square=False -- coils sized
                    to snugly tile the surface, elliptical/superellipse cross section.
                "wedge": generate_windowpane_wedge_array -- fixed-size square coils in
                    toroidal wedges bounded by constant-toroidal-angle R-Z planes, with
                    a poloidal strip of coils per wedge.
                "ring": generate_windowpane_ring_array -- fixed-size square coils, built
                    ring by ring in the poloidal direction with each ring independently
                    tiling the toroidal direction (correctly buffers every ring from the
                    phi=0/half-period mirror planes). Only supports axisymmetric winding
                    surfaces (raises ValueError otherwise) -- use "metric_ring" for a
                    non-axisymmetric one.
                "metric_ring": generate_windowpane_metric_ring_array (default) -- same
                    ring-by-ring approach as "ring", but sizes and spaces coils from the
                    surface's exact first fundamental form, so it also works for
                    non-axisymmetric winding surfaces (and reduces to exactly the same
                    result as "ring" for an axisymmetric one).
        theta_range: (theta_min, theta_max) in radians, only valid when wp_layout is
            "ring" or "metric_ring" (raises ValueError otherwise). Defines the range
            of poloidal angle over which the poloidal ring is equispaced by real arc
            length; ``None`` (default) uses the full poloidal loop, (0, 2*pi). Pass a
            strict subrange to restrict dipole coils to that poloidal arc only (e.g.
            to skip the inboard side).
        tf_init_fac: float
            The factor by which to scale the TF coils.
        ntf: int
            The number of TF coils per half field period.
    """
    from simsopt.geo import curves_to_vtk

    if theta_range is not None and wp_layout not in ("ring", "metric_ring"):
        raise ValueError(
            f"theta_range is only valid when wp_layout is 'ring' or 'metric_ring'; "
            f"got wp_layout={wp_layout!r} with theta_range={theta_range!r}."
        )

    # choose some reasonable parameters for array initialization
    if wp_layout == "jake":
        base_wp_curves = generate_windowpane_array(
            winding_surface=VV,
            inboard_radius=inboard_radius,
            wp_fil_spacing=wp_fil_spacing,
            half_per_spacing=half_per_spacing,
            wp_n=wp_n,  # elliptical coils
            numquadpoints=numquadpoints,
            order=order_wp,  # want high order to approximate ellipse
            verbose=verbose,
        )
    elif wp_layout == "wedge":
        base_wp_curves = generate_windowpane_wedge_array(
            winding_surface=VV,
            inboard_radius=inboard_radius,
            wp_fil_spacing=wp_fil_spacing,
            half_per_spacing=half_per_spacing,
            wp_n=wp_n,  # square coils
            numquadpoints=numquadpoints,
            order=order_wp,  # want high order to approximate square
            verbose=verbose,
        )
    elif wp_layout == "ring":
        base_wp_curves = generate_windowpane_ring_array(
            winding_surface=VV,
            inboard_radius=inboard_radius,
            wp_fil_spacing=wp_fil_spacing,
            half_per_spacing=half_per_spacing,
            wp_n=wp_n,  # square coils
            numquadpoints=numquadpoints,
            order=order_wp,  # want high order to approximate square
            theta_range=theta_range if theta_range is not None else (0.0, 2 * np.pi),
            verbose=verbose,
        )
    elif wp_layout == "metric_ring":
        base_wp_curves = generate_windowpane_metric_ring_array(
            winding_surface=VV,
            inboard_radius=inboard_radius,
            wp_fil_spacing=wp_fil_spacing,
            half_per_spacing=half_per_spacing,
            wp_n=wp_n,  # square coils
            numquadpoints=numquadpoints,
            order=order_wp,  # want high order to approximate square
            theta_range=theta_range if theta_range is not None else (0.0, 2 * np.pi),
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unrecognized wp_layout '{wp_layout}'; must be one of "
            "'jake', 'wedge', 'ring', or 'metric_ring'."
        )
    # generate TFs of the class CurvePlanarEllipticalCylindrical (fixed_geo_TFs=False)
    base_tf_curves = generate_tf_array(
        winding_surface=VV,
        ntf=ntf,  # 3 TF coils
        TF_R0=surf.major_radius(),
        TF_a=surf.minor_radius() * tf_init_fac,
        TF_b=surf.minor_radius() * tf_init_fac,
        fixed_geo_tfs=fixed_geo_tfs,
        planar_fourier_tfs=planar_fourier_tfs,
        elliptical_tfs=ellptical_tfs,
        order=order_tf,
        numquadpoints=numquadpoints,
    )
    if ellptical_tfs:
        # unfix the relevant TF dofs
        for c in base_tf_curves:
            c.fix_all()
            c.unfix("R0")
            c.unfix("r_rotation")

    # export curves
    curves_to_vtk(base_wp_curves + base_tf_curves, outdir + "test_curves")
    surf.to_vtk(outdir + "test_winding_surf")
    VV.to_vtk(outdir + "test_vessel")
    return base_wp_curves, base_tf_curves

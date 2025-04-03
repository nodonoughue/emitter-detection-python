import utils
import numpy as np


_deg2rad = utils.unit_conversions.convert(1, 'deg', 'rad')

# TODO: Test and Debug!!
# TODO: Convert filterGrid.m?  Not actually used in MATLAB code.

# **********************************************************************************************************************
# Altitude Constraints
# **********************************************************************************************************************
def fixed_alt(alt_val: float, geo_type: str = 'ellipse', is_upper_bound=True):
    """
    Define the cost function and gradient for a fixed altitude constraint. The form of this cost function and its
    gradient depend on the Earth model in use, either flat Earth (altitude is the cartesian z-coordinate), spherical
    Earth (altitude is the norm of the cartesian coordinates, minus the radius of the Earth), or an ellipsoidal
    Earth (altitude is dependent on Latitude in addition to the norm of the cartesian coordinates).

    Ported from MATLAB code
    Nicholas O'Donoughue
    26 March 2025

    :param alt_val:         Altitude constraint [in meters]
    :param geo_type:        String, either 'ellipse', 'sphere', or 'flat'.
    :param is_upper_bound:  Boolean, determines whether the altitude constraint is upper-bounded (True) or lower-bounded
                            (false). Used for inequality constraints. If it is lower-bounded, then the sign on epsilon
                            and epsilon_gradient are reversed.
    :return a:              Function handle that accepts a (3,n) numpy array of coordinates and returns a (2, ) tuple.
                            The first output is an (n, ) numpy array of cost (the Euclidean distance between each point
                            and the desired altitude). The second output is a (3,n) numpy array of valid x-coordinates
                            that satisfy the constraint but are as close as possible to the input coordinates.
    :return a_gradient:     Function handle that accepts a (3,n) numpy array of coordinates and returns a (3,n) numpy
                            array with the gradient of the cost function (a) with respect to each spatial coordinate.
    """

    eps_multiplier = 1 if is_upper_bound else -1

    if geo_type.lower() == 'flat':
        def a(x):
            eps, x_valid = fixed_alt_constraint_flat(x, alt_val)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_alt_gradient_flat(x)

    elif geo_type.lower() == 'sphere':
        def a(x):
            eps, x_valid = fixed_alt_constraint_sphere(x, alt_val)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_alt_gradient_sphere(x)
    elif geo_type.lower() == 'ellipse':
        def a(x):
            eps, x_valid = fixed_alt_constraint_ellipse(x, alt_val)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_alt_gradient_ellipse(x)
    else:
        raise ValueError('Invalid geo_type')

    return a, a_gradient


def bounded_alt(geo_type: str, alt_min:float =None, alt_max:float =None):

    bounds = list([])

    # Upper Bound
    if alt_max is not None:
        a_upper, _ = fixed_alt(alt_max, geo_type, is_upper_bound=True)

        bounds.append(a_upper)

    # Lower Bound
    if alt_min is not None:
        a_lower, _ = fixed_alt(alt_min, geo_type, is_upper_bound=False)

        bounds.append(a_lower)

    return bounds


def fixed_alt_constraint_flat(x: np.ndarray, alt: float):
    """
    Implement the flat Earth altitude constraint; altitude is simply the 3rd cartesian dimension.
    Ported from MATLAB code.

    :param x:           numpy array (3,n) of coordinates
    :param alt:         desired altitude (same units as x); must be scalar
    :return epsilon:    Euclidean distance between each point and the desired altitude
    :return x_valid:    Valid x-coordinates that satisfy the altitude constraint while being as close as possible to the
                        input coordinates.
    """
    verify_3d_input(x)

    # Flat Earth; altitude is the third dimension
    epsilon = x[2] - alt

    # To make x valid, replace the altitude with alt
    x_valid = x.copy()
    x_valid[2] = alt

    return epsilon, x_valid


def fixed_alt_gradient_flat(x: np.ndarray):
    """
    Implement the gradient of the flat Earth altitude constraint; altitude is simply the 3rd cartesian dimension.
    Ported from MATLAB code.

    :param x:                   numpy array (3,n) of coordinates
    :return epsilon_gradient:   Gradient of the Euclidean distance between each point and the desired altitude, as a
                                function of the 3 spatial coordinates.
    """

    verify_3d_input(x)

    # For flat Earth, the gradient is simply a one in the vertical coordinate, and zero elsewhere
    epsilon_gradient = np.zeros_like(x)
    epsilon_gradient[2] = 1.

    return epsilon_gradient


def fixed_alt_constraint_sphere(x: np.ndarray, alt: float):
    """
    Implement the spherical Earth altitude constraint; altitude is simply the Euclidean norm of each coordinate. The
    radius of the Earth (utils.constants.radius_earth_true) will be subtracted from the norm before comparing to the
    altitude constraint.

    Ported from MATLAB code.

    :param x:           numpy array (3,n) of coordinates in ECEF
    :param alt:         desired altitude (must be in meters); must be scalar
    :return epsilon:    Euclidean distance between each point and the desired altitude
    :return x_valid:    Valid x-coordinates that satisfy the altitude constraint while being as close as possible to the
                        input coordinates.
    """

    verify_3d_input(x)

    # Implement equation 5.5, and the scale term defined in equation 5.9
    radius_target_square = np.sum(np.abs(x)**2, axis=0)
    epsilon = radius_target_square - (utils.constants.radius_earth_true + alt)**2      # eq 5.5
    scale = (utils.constants.radius_earth_true + alt) / np.sqrt(radius_target_square)  # eq 5.9, modified

    x_valid = scale @ x
    return epsilon, x_valid


def fixed_alt_gradient_sphere(x: np.ndarray):
    """
    Implement the gradient of the spherical Earth altitude constraint; altitude is simply the Euclidean norm of each
    coordinate.

    Ported from MATLAB code.

    :param x:                   numpy array (3,n) of coordinates in ECEF
    :return epsilon_gradient:   Gradient of the Euclidean distance between each point and the desired altitude, as a
                                function of the 3 spatial coordinates.
    """

    verify_3d_input(x)

    # Implement the gradient of equation 5.5, with respect to x, which is simply 2*x
    return 2*x


def fixed_alt_constraint_ellipse(x: np.ndarray, alt: float):
    """
    Implement the ellipsoidal Earth altitude constraint. Uses the coordinate conversion ecef_to_lla to compute the
    altitude of each point.

    Ported from MATLAB code.

    :param x:           numpy array (3,n) of coordinates in ECEF
    :param alt:         desired altitude (must be meters); must be scalar
    :return epsilon:    Euclidean distance between each point and the desired altitude
    :return x_valid:    Valid x-coordinates that satisfy the altitude constraint while being as close as possible to the
                        input coordinates.
    """

    verify_3d_input(x)

    # Convert ECEF to LLA
    [lat, lon, alt_coord] = utils.coordinates.ecef_to_lla(x[0], x[1], x[2])

    # Compare altitude to desired
    epsilon = alt_coord - alt

    # Find the nearest valid x -- replace computed altitude with desired
    xx, yy, zz = utils.coordinates.lla_to_ecef(lat, lon, alt * np.ones_like(alt_coord))
    x_valid = np.concatenate((xx[np.newaxis], yy[np.newaxis], zz[np.newaxis]), axis=0)

    return epsilon, x_valid


def fixed_alt_gradient_ellipse(x: np.ndarray):
    """
    Implement the gradient of the ellipsoidal Earth altitude constraint.

    Ported from MATLAB code.

    :param x:                   numpy array (3,n) of coordinates in ECEF
    :return epsilon_gradient:   Gradient of the Euclidean distance between each point and the desired altitude, as a
                                function of the 3 spatial coordinates.
    """

    verify_3d_input(x)

    # Implement equations 5.18 through 5.26

    # Load constants
    e1sq = utils.constants.first_ecc_sq
    a = utils.constants.semimajor_axis_km * 1e3

    # Break position into x/y/z components
    xx = x[0]
    yy = x[1]
    zz = x[2]

    # Compute geodetic latitude
    lat, _, _ = utils.coordinates.ecef_to_lla(xx, yy, zz)
    lat_rad = lat * _deg2rad

    # Pre-compute some repeated terms
    xy_len_sq = xx**2+yy**2
    xy_len = np.sqrt(xy_len_sq)
    zz_sq = zz**2

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Compute gradient of geodetic latitude, equations 5.24-5.26
    d_lat_dx = -xx*zz @ (1-e1sq) / (xy_len * (zz_sq + (1-e1sq)**2 @ xy_len_sq))
    d_lat_dy = -yy*zz @ (1-e1sq) / (xy_len * (zz_sq + (1-e1sq)**2 @ xy_len_sq))
    d_lat_dz = (1-e1sq) @ xy_len / (zz_sq + (1-e1sq) @ xy_len_sq)

    # Compute gradient of effective radius, equations 5.21-5.23
    d_r_dx = a*e1sq*sin_lat*cos_lat*d_lat_dx/((1-e1sq*sin_lat**2) ** 1.5)
    d_r_dy = a*e1sq*sin_lat*cos_lat*d_lat_dy/((1-e1sq*sin_lat**2) ** 1.5)
    d_r_dz = a*e1sq*sin_lat*cos_lat*d_lat_dz/((1-e1sq*sin_lat**2) ** 1.5)

    # Compute gradient of constraint (epsilon), equations 5.18-5.20
    d_eps_dx = (xx - yy**2 / xx)/(cos_lat*xy_len)-d_r_dx
    d_eps_dy = (xx + yy)/(cos_lat*xy_len)-d_r_dy
    d_eps_dz = -d_r_dz

    epsilon_grad = np.concatenate((d_eps_dx[np.newaxis],
                                   d_eps_dy[np.newaxis],
                                   d_eps_dz[np.newaxis]), axis=0)

    return epsilon_grad




# **********************************************************************************************************************
# Cartesian Constraints
# **********************************************************************************************************************
def fixed_cartesian(bound_type: str, bound_val: float = None, x0: np.ndarray = None, u_vec: np.ndarray = None,
                    is_upper_bound=True):
    """
    Generate a set of constraints and constraint gradient functions for fixed solution constraints on a cartesian grid.

    If type is 'x', 'y', or 'z', it creates a fixed bound along the specified axis. If type is 'linear', creates a
    linear bound defined by a point x0 and pointing vector u.

    Ported from MATLAB code
    Nicholas O'Donoughue
    26 March 2025

    :param bound_type:      String, either 'x', 'y', 'z', or 'linear'
    :param bound_val:       Double, the value of the bound (for 'x', 'y', or 'z')
    :param x0:              numpy (3, ) array specifying the starting point for a 'linear' bound
    :param u_vec:           numpy (3, ) array specifying the pointing vector for a 'linear' bound
    :param is_upper_bound:  Boolean, determines whether the altitude constraint is upper-bounded (True) or lower-bounded
                            (false). Used for inequality constraints. If it is lower-bounded, then the sign on epsilon
                            and epsilon_gradient are reversed.
    :return a:              Function handle that accepts a (3,n) numpy array of coordinates and returns a (2, ) tuple.
                            The first output is an (n, ) numpy array of cost (the Euclidean distance between each point
                            and the desired altitude). The second output is a (3,n) numpy array of valid x-coordinates
                            that satisfy the constraint but are as close as possible to the input coordinates.
    :return a_gradient:     Function handle that accepts a (3,n) numpy array of coordinates and returns a (3,n) numpy
                            array with the gradient of the cost function (a) with respect to each spatial coordinate.
    """

    eps_multiplier = 1 if is_upper_bound else -1

    if bound_type.lower() == 'linear':
        assert x0 is not None and u_vec is not None, 'Error parsing inputs. Linear bound requires x0 and u_vec.'
    else:
        assert bound_val is not None, 'Error parsing inputs. x/y/z bounds require bound_val to be defined.'

    if bound_type.lower() == 'x':
        def a(x):
            eps, x_valid = fixed_cartesian_xyz(x, bound_val, axis=0)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_cartesian_gradient_xyz(x, axis=0)

    elif bound_type.lower() == 'y':
        def a(x):
            eps, x_valid = fixed_cartesian_xyz(x, bound_val, axis=1)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_cartesian_gradient_xyz(x, axis=1)

    elif bound_type.lower() == 'z':
        def a(x):
            eps, x_valid = fixed_cartesian_xyz(x, bound_val, axis=2)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_cartesian_gradient_xyz(x, axis=2)

    elif bound_type.lower() == 'linear':
        def a(x):
            eps, x_valid = fixed_cartesian_linear(x, x0, u_vec)

            # Don't apply the sign change to x_valid, only to the computed error (eps)
            return eps_multiplier * eps, x_valid

        def a_gradient(x):
            return eps_multiplier * fixed_cartesian_gradient_linear(x, x0, u_vec)
    else:
        raise ValueError('Invalid bound_type')

    return a, a_gradient


def fixed_cartesian_xyz(x: np.ndarray, bound_val: float, axis: int):

    # Cartesian bounds work on 2D or 3D
    # verify_3d_input(x)

    # Compute the difference between each coordinate's value in the specified axis and the desired constraint
    epsilon = x[axis] - bound_val

    # Replace the specified axis with bound_val to generate coordinates that satisfy the constraint
    x_valid = x.copy()
    x_valid[axis] = bound_val

    return epsilon, x_valid

def fixed_cartesian_gradient_xyz(x: np.ndarray, axis: int):

    # Cartesian bounds work on 2D or 3D
    # verify_3d_input(x)

    # The gradient is all zeros, except for ones in the specified axis
    epsilon_gradient = np.zeros_like(x)
    epsilon_gradient[axis, :] = 1.0
    return epsilon_gradient


def fixed_cartesian_linear(x: np.ndarray, x0: np.ndarray, u_vec: np.ndarray):

    # Make sure all three inputs have the same number of spatial coordinates
    verify_common_dim(x, x0, u_vec)
    n_dim, _ = utils.safe_2d_shape(x)

    # Make sure the pointing vector is unit-norm and compute projection matrix
    u_vec = u_vec / np.linalg.norm(u_vec)
    proj_matrix = np.eye(n_dim) - u_vec @ u_vec.T

    # Find the component of x-x0 that is perpendicular to u_vec
    x_ortho = proj_matrix @ (x - x0)

    # The distance is the norm of x_ortho along the first axis
    epsilon = np.linalg.norm(x_ortho, axis=0)

    # A valid x will be each coordinate minus its orthogonal projection
    x_valid = x - x_ortho

    return epsilon, x_valid


def fixed_cartesian_gradient_linear(x: np.ndarray, x0: np.ndarray, u_vec: np.ndarray):
    # Make sure all three inputs have the same number of spatial coordinates
    verify_common_dim(x, x0, u_vec)
    n_dim, _ = utils.safe_2d_shape(x)

    # Make sure the pointing vector is unit-norm and compute projection matrix
    u_vec = u_vec / np.linalg.norm(u_vec)
    proj_matrix = np.eye(n_dim) - u_vec @ u_vec.T

    epsilon_gradient = 2 * proj_matrix @ (x - x0)
    return epsilon_gradient


# **********************************************************************************************************************
# Utilities Constraints
# **********************************************************************************************************************
def verify_3d_input(x: np.ndarray):
    """
    Ensure that the input x is a (3,n) numpy array. Assert an error if it is not.
    """
    n_dim, n_val = utils.safe_2d_shape(x)
    assert n_dim == 3, 'Unable to constrain altitude; input coordinates have unexpected shape.'
    return


def verify_common_dim(*args: np.ndarray):
    """
    Ensure that all inputs args have a common first dimension
    """
    dims = [np.shape(this_arg)[0] for this_arg in args]
    assert np.all(dims == dims[0]), 'Not all inputs have the same number of spatial dimensions'


def snap_to_equality_constraints(x: np.ndarray, eq_constraints: list, tol: float = 1e-6):
    """
    Apply the equality constraints in the function handle eq_constraints to the position
    x, subject to a tolerance tol.

    If multiple constraints are provided, then they are applied in order. In
    this manner, only the final one is guaranteed to be satisfied by the
    output x_valid, as application of later constraints may violate earlier
    ones.

    If the variable x has non-singleton dimensions (beyond the first), then
    this operation is repeated across them; the first dimensions is assumed
    to be used for vector inputs x (such as a 2D or 3D target position).

    The projection operation is carried out as discussed in equations 5.10
    and 5.11.  This is intended for use in iterative solvers.

    Ported from MATLAB code.

    :param x:               Input x-coordinates, numpy array of size (n_dim, n)
    :param eq_constraints:  List containing 1 or more function handles to equality constraints; each must return a
                            2-tuple with the error and a set of valid coordinates.
    :param tol:             Tolerance for equality constraints; default = 1e-6
    :return x_valid:        Valid x-coordinates, numpy array of size (n_dim, n)
    """

    x_valid = x.copy()
    for constraint in eq_constraints:
        # Apply the constraint
        this_eps, this_x_valid = constraint(x_valid)

        # Check against tolerance
        tol_mask = np.fabs(this_eps) <= tol

        if not np.all(tol_mask):
            # At least one point needs to be updated
            x_valid[:, not tol_mask] = this_x_valid[:, not tol_mask]

            # TODO: Ensure that the new points satisfy all old constraints, as well.

    return x_valid


def snap_to_inequality_constraints(x: np.ndarray, ineq_constraints: list):
    """
    Apply the inequality constraints in the function handle ineq_constraints to the position
    x.

    If multiple constraints are provided, then they are applied in order. In
    this manner, only the final one is guaranteed to be satisfied by the
    output x_valid, as application of later constraints may violate earlier
    ones.

    If the variable x has non-singleton dimensions (beyond the first), then
    this operation is repeated across them; the first dimensions is assumed
    to be used for vector inputs x (such as a 2D or 3D target position).

    The projection operation is carried out as discussed in equations 5.10
    and 5.11, and Figure 5.6.  This is intended for use in iterative solvers.

    Ported from MATLAB code.

    :param x:               Input x-coordinates, numpy array of size (n_dim, n)
    :param ineq_constraints:List containing 1 or more function handles to inequality constraints; each must return a
                            2-tuple with the error and a set of valid coordinates.
    :return x_valid:        Valid x-coordinates, numpy array of size (n_dim, n)
    """

    x_valid = x.copy()
    for constraint in ineq_constraints:
        # Apply the constraint
        this_eps, this_x_valid = constraint(x_valid)
        valid_mask = this_eps <= 0.  # all points with non-positive error are valid; no need to update

        if not np.all(valid_mask):
            # At least one point needs to be updated
            x_valid[:, not valid_mask] = this_x_valid[:, not valid_mask]

            # TODO: Ensure that the new points satisfy all old constraints, as well.

    return x_valid


def constrain_likelihood(ell, eq_constraints: list=None, ineq_constraints: list=None, tol: float = 1e-6):
    """
    Accepts a set of functions handles ell (likelihood), a (equality
    constraints), b (inequality constraints), and a tolerance to apply
    to the equality constraints.

    Returns a constrained likelihood function handle that will accept an
    nDim x N set of position vectors, and return a vector of N constrained
    likelihood values.  If either the abs(a(x))<= tol, or b(x)<=0 constraints
    are violated, then the likelihood is -Inf.

    Ported from MATLAB code.

    :param ell:                 Likelihood function handle
    :param eq_constraints:      list of equality constraint function handles (default=None)
    :param ineq_constraints:    list of inequality constraint function handles (default=None)
    :param tol:                 tolerance for equality constraints (default=1e-6)
    :return ell_constrained:    Function handle to a modified likelihood, that returns -Inf for any points that violate
                                one or more constraints.
    """

    # Make certain that eq_constraints and ineq_constraints are iterable
    if ineq_constraints is not None:
        utils.ensure_iterable(ineq_constraints, flatten=True)
    if eq_constraints is not None:
        utils.ensure_iterable(eq_constraints, flatten=True)

    def ell_constrained(x):
        valid_mask = True

        if eq_constraints is not None:
            for constraint in eq_constraints:
                eps, _ = constraint(x)
                valid_mask = valid_mask and eps <= tol

        if ineq_constraints is not None:
            for constraint in ineq_constraints:
                eps, _ = constraint(x)
                valid_mask = valid_mask and eps <= 0.

        result = -np.inf * np.ones_like(valid_mask)
        result[valid_mask] = ell(x[:, valid_mask])
        return result

    return ell_constrained

import numpy as np
from scipy.special import erfc, erf


def base_solution(x, t, c_boundary, v, D):
    """Base solution of the 1-D advection-dispersion-reaction equation with a continuous source boundary condition

    This corresponds to the term H in the above equation.
    """
    k = 0
    u = v * np.sqrt(1 + 4 * k * D / v**2)
    a = 2 * np.sqrt(D * t)
    b = x / (2 * D)
    minus_term = 1 / 2 * np.exp((v - u) * b) * erfc((x - u * t) / a)
    plus_term = 1 / 2 * np.exp((v + u) * b) * erfc((x + u * t) / a)
    log_plus_term = 1/2 + (v + u) * b + np.log(erfc((x + u * t) / a))
    return minus_term + np.exp(log_plus_term)


def advection_dispersion_reaction_step_function(x, t, c_boundary, v, D, k, T):
    "Solution of the 1-D advection-dispersion-reaction equation with a step function as boundary condition"
    t_shifted = np.maximum(t - T, 0)
    c_source = base_solution(x, t, c_boundary, v, D, k)
    c_flush = -base_solution(x, t_shifted, c_boundary, v, D, k)
    return c_source + c_flush


def advection_dispersion_reaction_retardation(x, t, c_boundary, v, D, k, T, R):
    """Solution of the 1-D advection-dispersion-reaction equation with linear equilibrium sorption

    - in a semi-infinite domain
    - with a constant concentration boundary condition described by a step function.
    """
    # Transport parameters are scaled by the retardation factor
    v_R = v / R
    D_R = D / R
    k_R = k / R
    # The analytical solution is called with the scaled parameters
    return advection_dispersion_reaction_step_function(
        x, t, c_boundary, v_R, D_R, k_R, T
    )

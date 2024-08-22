from desc.backend import jnp
from desc.optimize import register_optimizer
from pdfo import pdfo
from scipy.optimize import NonlinearConstraint, Bounds



@register_optimizer(
    name="pdfo",
    description="Powell's Derivative Free Global Optimizer, see https://pdfo.net/index.html",
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    hessian=False,
    stochastic=False,
    GPU=False
)
def _optimize_pdfo(objective, constraint, x0, method, x_scale, verbose, stoptol, options=None):
    '''
    Wrapper for pdfo global optimizer.
    '''
    options = {} if options is None else options
    options["quiet"] = not verbose
    options["maxfev"] = stoptol["max_nfev"]
    options["radius_final"] = stoptol["xtol"]
    #Assume minimum is at 0
    #options["ftarget"] = stoptol["ftol"]
    x_scale = 1 if x_scale == "auto" else x_scale

    fun = objective.compute_scalar

    if constraint is not None:
        constraint_wrapped = NonlinearConstraint(
            constraint.compute_scaled,
            constraint.bounds_scaled[0],
            constraint.bounds_scaled[1],
            constraint.jac_scaled
        )
    else:
        constraint_wrapped = None

    result = pdfo(fun, x0=x0, constraints=constraints_wrapped, options=options)

    return result





    
from scipy.optimize import NonlinearConstraint
from scipy.optimize import OptimizeResult

from desc.backend import jnp
import numpy as np

from .aug_lagrangian import fmin_auglag
from .aug_lagrangian_ls import lsq_auglag
from .utils import compute_hess_scale, compute_jac_scale
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .optimizer import register_optimizer
from .stochastic import sgd
#from turbo import DTurbo1, Turbo1, TurboM
from turbo import Turbo1, TurboM

import inspect


@register_optimizer(
    name=["fmin-auglag", "fmin-auglag-bfgs"],
    description=[
        "Augmented Lagrangian trust region method for minimizing scalar valued "
        + "multivariate function. "
        + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmin_auglag.html",  # noqa: E501
        "Augmented Lagrangian trust region method for minimizing scalar valued "
        + "multivariate function. Uses BFGS to approximate Hessian. "
        + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmin_auglag.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=[True, False],
    GPU=True,
)
def _optimize_desc_aug_lagrangian(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmin_auglag.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : {"fmin-auglag", "fmin-auglag-bfgs"}
        Name of the method to use.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to 'jac', the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.fmin_auglag`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    options["max_nfev"] = stoptol["max_nfev"]
    # local lambdas to handle constants from both objective and constraint
    hess = (lambda x, *c: objective.hess(x, c[0])) if "bfgs" not in method else "bfgs"

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            lambda x, *c: constraint.compute_scaled(x, c[1]),
            lb,
            ub,
            lambda x, *c: constraint.jac_scaled(x, c[1]),
        )
        # TODO (#1394): can't pass constants dict into vjp for now
        constraint_wrapped.vjp = lambda v, x, *args: constraint.vjp_scaled(v, x)
    else:
        constraint_wrapped = None

    result = fmin_auglag(
        lambda x, *c: objective.compute_scalar(x, c[0]),
        x0=x0,
        grad=lambda x, *c: objective.grad(x, c[0]),
        hess=hess,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(objective.constants, constraint.constants if constraint else None),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-auglag",
    description="Least squares augmented Lagrangian for constrained optimization"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsq_auglag.html",  # noqa: E501
    scalar=False,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_aug_lagrangian_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsq_auglag.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : {"lsq-auglag"}
        Name of the method to use.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to 'jac', the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.lsq_auglag`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    options["max_nfev"] = stoptol["max_nfev"]

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            lambda x, *c: constraint.compute_scaled(x, c[1]),
            lb,
            ub,
            lambda x, *c: constraint.jac_scaled(x, c[1]),
        )
    else:
        constraint_wrapped = None

    result = lsq_auglag(
        lambda x, *c: objective.compute_scaled_error(x, c[0]),
        x0=x0,
        jac=lambda x, *c: objective.jac_scaled_error(x, c[0]),
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(objective.constants, constraint.constants if constraint else None),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-exact",
    description="Trust region least squares, similar to the `trf` method in scipy"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsqtr.html",  # noqa: E501
    scalar=False,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsqtr.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : {"lsq-exact"}
        Name of the method to use.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to 'jac', the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.lsqtr`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    elif options.get("initial_trust_radius", "scipy") == "scipy":
        options.setdefault("initial_trust_ratio", 0.1)
    options["max_nfev"] = stoptol["max_nfev"]

    result = lsqtr(
        objective.compute_scaled_error,
        x0=x0,
        jac=objective.jac_scaled_error,
        args=(objective.constants,),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name=[
        "fmintr",
        "fmintr-bfgs",
    ],
    description=[
        "Trust region method for minimizing scalar valued multivariate function. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmintr.html",  # noqa: E501
        "Trust region method for minimizing scalar valued multivariate function. Uses "
        + "BFGS to approximate the Hessian. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmintr.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=[True, False],
    GPU=True,
)
def _optimize_desc_fmin_scalar(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmintr.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to 'jac', the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    hess = objective.hess if "bfgs" not in method else "bfgs"
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    elif options.get("initial_trust_radius", "scipy") == "scipy":
        options.setdefault("initial_trust_ratio", 0.1)
    options["max_nfev"] = stoptol["max_nfev"]

    result = fmintr(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name="sgd",
    description="Stochastic gradient descent with Nesterov momentum"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.sgd.html",  # noqa: E501
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=True,
    hessian=False,
    GPU=True,
)
def _optimize_desc_stochastic(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.sgd.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to 'jac', the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.sgd`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    result = sgd(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        args=(objective.constants,),
        method=method,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


'''@register_optimizer(
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

    #Wrapper for pdfo global optimizer.
    
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

    result = pdfo(fun, x0=x0, constraints=constraint_wrapped, options=options)

    return result'''



@register_optimizer(
    name="Turbo1",
    description="Trust Region Bayesian Optimizer, 1 Trust Region",
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    hessian=False,
    stochastic=True,
    GPU=False
)
def _optimize_Turbo1(objective, constraint, x0, method, x_scale, verbose, stoptol, options=None):
    '''
    Wrapper for TuRBO-1 global optimizer.  Does not accept nonlinear constraints, only bounding boxes.
    '''

    options = {} if options is None else options
    if "batch_size" not in options:
        options["batch_size"] = 10
    if "use_ard" not in options:
        options["use_ard"] = True
    if "cholesky_size" not in options:
        options["cholesky_size"] = 2000
    if "training_steps" not in options:
        options["training_steps"] = 50
    if "box_size" not in options:
        options["box_size"] = 0.2
    
    if "max_evals" not in options:
        options["max_evals"] = -1
    if "objective_tol" not in options:
        options["objective_tol"] = -1
    if "improvement_tol" not in options:
        options["improvement_tol"] = -1
    if "max_time" not in options:
        options["max_time"] = -1

    fun = objective.compute_scalar
    # fun = objective.compute_scaled_error
    #print(inspect.getargspec(fun))
    lenObj = len(objective.x())

    x0array = np.array(x0)
    # Adaptive bounds using Hessian scale
    # try:
    # J = objective.jac_scaled_error(x0)
    H = objective.hess(x0)
    scale, _ = compute_hess_scale(H)
    # scale, _ = compute_jac_scale(J)

    # except Exception:
    #     scale = np.abs(x0array) + 1e-8  # fallback if Hessian fails
    # x0array = x0array / scale
    boxsize = options["box_size"]
    # min_scale = 1e-12
    # lb = np.array(x0array - boxsize * np.maximum(scale, min_scale))
    # ub = np.array(x0array + boxsize * np.maximum(scale, min_scale))
    # ub = np.ones(lenObj) * x0array + min_scale
    # lb = np.ones(lenObj) * x0array - min_scale
    # inds = np.where(np.abs(x0array) > 1e-6)
    # ub[inds] = x0array[inds] + 1e-2
    # lb[inds] = x0array[inds] - 1e-2

    ub = np.maximum((1 + boxsize) * x0array, (1 - boxsize) * x0array) + 1e-12 * np.ones(lenObj)
    lb = np.minimum((1 + boxsize) * x0array, (1 - boxsize) * x0array) - 1e-12 * np.ones(lenObj)
    # If you have Hessian H and domain bounds (ub - lb)
    # hessian_scales = 1.0 / np.sqrt(np.abs(np.diag(H)))  # Your current approach
    # domain_scales = (ub - lb) / 10
    # scale = np.minimum(scale, domain_scales)
    #print("ub - lb", np.max(ub - lb), np.min(ub - lb), np.mean(ub - lb))
    # print("scale", scale)
    #print("lb", lb)
    #print("ub", ub)

    
    n_init = 2*lenObj
    
    turbo1 = Turbo1(f = fun,
                    lb = lb, #constraint.bounds_scaled[0],
                    ub = ub, #constraint.bounds_scaled[1],
                    # scale=scale,
                    n_init = n_init,
                    max_evals = options["max_evals"],
                    objective_tol = options["objective_tol"],
                    improvement_tol = options["improvement_tol"],
                    max_time = options["max_time"],
                    batch_size = options["batch_size"],
                    verbose = verbose > 0,
                    use_ard = options["use_ard"],
                    max_cholesky_size = options["cholesky_size"],
                    n_training_steps = options["training_steps"],
                    min_cuda = 1024,
                    device = "cpu",
                    dtype = "float64",
                    # initial_lengthscales=scale
                   )
    turbo1.optimize()
    X = turbo1.X
    fX = turbo1.fX
    bestX = turbo1.bestX
    bestfX = turbo1.bestfX
    ind_best = np.argmin(bestfX)
    f_best, x_best = bestfX[ind_best], bestX[ind_best, :]
    f_init, x_init = bestfX[0, 0], bestX[0, :]
    
    result = OptimizeResult()
    result.success = True
    result.x = x_best
    result.eqparams = objective.unpack_state(x_best, False)
    result.fun = f_best
    result.allx = X
    result.all_eqparams = [objective.unpack_state(x, False) for x in X]
    result.allfun = fX
    result.nfev = turbo1.n_evals
    result.eqparams_init = objective.unpack_state(x_init, False)
    result.f_init = f_init

    return result

    

@register_optimizer(
    name="DTurbo1",
    description="Trust Region Bayesian Optimizer, 1 Trust Region",
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    hessian=False,
    stochastic=True,
    GPU=False
)
def _optimize_DTurbo1(objective, constraint, x0, method, x_scale, verbose, stoptol, options=None):
    '''
    Wrapper for TuRBO-1 global optimizer.  Does not accept nonlinear constraints, only bounding boxes.
    '''

    options = {} if options is None else options
    if "batch_size" not in options:
        options["batch_size"] = 10
    if "use_ard" not in options:
        options["use_ard"] = True
    if "cholesky_size" not in options:
        options["cholesky_size"] = 2000
    if "training_steps" not in options:
        options["training_steps"] = 50
    if "box_size" not in options:
        options["box_size"] = 0.2
    if "max_iter" not in stoptol:
        stoptol["max_iter"] = 500

    fun = objective.compute_scalar
    # fun = objective.compute_scaled_error
    #print(inspect.getargspec(fun))
    lenObj = len(objective.x())

    x0array = np.array(x0)
    # Adaptive bounds using Hessian scale
    # try:
    J = objective.jac_scaled_error(x0)
    # H = objective.hess(x0)
    # scale, _ = compute_hess_scale(H)
    scale, _ = compute_jac_scale(J)

    # except Exception:
    #     scale = np.abs(x0array) + 1e-8  # fallback if Hessian fails
    # x0array = x0array / scale
    boxsize = options["box_size"]
    min_scale = 1e-12
    # lb = np.array(x0array - boxsize * np.maximum(scale, min_scale))
    # ub = np.array(x0array + boxsize * np.maximum(scale, min_scale))

    ub = np.maximum((1 + boxsize) * x0array, (1 - boxsize) * x0array) + 1e-12 * np.ones(lenObj)
    lb = np.minimum((1 + boxsize) * x0array, (1 - boxsize) * x0array) - 1e-12 * np.ones(lenObj)
    print("ub - lb", np.max(ub - lb), np.min(ub - lb), np.mean(ub - lb))
    # print("scale", scale)
    print("lb", lb)
    print("ub", ub)
    turbo1 = DTurbo1(f = fun,
                    df = objective.grad,
                    lb = lb, #constraint.bounds_scaled[0],
                    ub = ub, #constraint.bounds_scaled[1],
                    num_directions = 1,
                    minibatch_size = 10,
                    num_inducing=10,
                    n_init = 2*lenObj,
                    max_evals = stoptol["maxiter"] + 2*lenObj,
                    batch_size = options["batch_size"],
                    verbose = verbose > 0,
                    use_ard = options["use_ard"],
                    max_cholesky_size = options["cholesky_size"],
                    n_training_steps = options["training_steps"],
                    min_cuda = 1024,
                    device = "cpu",
                    dtype = "float64"
                   )
    turbo1.optimize()
    X = turbo1.X
    fX = turbo1.fX
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]
    print("f_best", f_best)
    print("fun(x_best)", fun(x_best), "x_best.shape", x_best.shape)    
    
    result = OptimizeResult()
    result.success = True
    result.x = x_best
    result.fun = f_best
    result.allx = X
    result.allfun = fX
    result.nfev = turbo1.n_evals

    return result


@register_optimizer(
    name="TurboM",
    description="Trust Region Bayesian Optimizer, M Trust Regions",
    scalar=True,
    equality_constraints=False,
    inequality_constraints=True,
    hessian=False,
    stochastic=True,
    GPU=False
)
def _optimize_TurboM(objective, constraint, x0, method, x_scale, verbose, stoptol, options=None):
    '''
    Wrapper for TuRBO-M global optimizer.  Does not accept nonlinear constraints, only bounding boxes.
    '''

    
    options = {} if options is None else options
    if "batch_size" not in options:
        options["batch_size"] = 10
    if "use_ard" not in options:
        options["use_ard"] = True
    if "cholesky_size" not in options:
        options["cholesky_size"] = 2000
    if "training_steps" not in options:
        options["training_steps"] = 50
    if "trust_regions" not in options:
        options["trust_regions"] = 5
    if "box_size" not in options:
        options["box_size"] = 0.2
    if "max_iter" not in stoptol:
        stoptol["max_iter"] = 500

    fun = objective.compute_scalar
    #print(inspect.getargspec(fun))
    #print(objective.x())
    lenObj = len(objective.x())

    x0array = np.array(x0)
    boxsize = options["box_size"]
    ub = np.maximum((1 + boxsize) * x0array, (1 - boxsize) * x0array) + 1e-12 * np.ones(lenObj)
    lb = np.minimum((1 + boxsize) * x0array, (1 - boxsize) * x0array) - 1e-12 * np.ones(lenObj)
    
    turboM = TurboM(f = fun,
                    lb = lb, #constraint.bounds_scaled[0],
                    ub = ub, #constraint.bounds_scaled[1],
                    n_init = 2*lenObj,
                    max_evals = stoptol["maxiter"] + 2*lenObj*options["trust_regions"],
                    n_trust_regions = options["trust_regions"],
                    batch_size = options["batch_size"],
                    verbose = verbose > 0,
                    use_ard = options["use_ard"],
                    max_cholesky_size = options["cholesky_size"],
                    n_training_steps = options["training_steps"],
                    min_cuda = 1024,
                    device = "cpu",
                    dtype = "float64"
                   )
    turboM.optimize()
    X = turboM.X
    fX = turboM.fX
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]
    
    result = OptimizeResult()
    result.success = True
    result.x = x_best
    result.fun = f_best
    result.allx = X
    result.allfun = fX
    result.nfev = turboM.n_evals

    return result
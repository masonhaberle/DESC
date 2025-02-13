"""Functions for flux surface averages and vector algebra operations."""

import copy
import inspect

import numpy as np

from desc.backend import cond, execute_on_cpu, fori_loop, jnp, put
from desc.grid import ConcentricGrid, Grid, LinearGrid

from ..utils import errorif, warnif
from .data_index import allowed_kwargs, data_index

# map from profile name to equilibrium parameter name
profile_names = {
    "pressure": "p_l",
    "iota": "i_l",
    "current": "c_l",
    "electron_temperature": "Te_l",
    "electron_density": "ne_l",
    "ion_temperature": "Ti_l",
    "atomic_number": "Zeff_l",
}


def _parse_parameterization(p):
    if isinstance(p, str):
        return p
    klass = p.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def compute(parameterization, names, params, transforms, profiles, data=None, **kwargs):
    """Compute the quantity given by name on grid.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    params : dict[str, jnp.ndarray]
        Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc.
        Defaults to attributes of self.
    transforms : dict of Transform
        Transforms for R, Z, lambda, etc. Default is to build from grid
    profiles : dict of Profile
        Profile objects for pressure, iota, current, etc. Defaults to attributes
        of self
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions.
        Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
        v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
        of the cylindrical coordinates R, ϕ, Z.

    Returns
    -------
    data : dict of ndarray
        Computed quantity and intermediate variables.

    """
    basis = kwargs.pop("basis", "rpz").lower()
    errorif(basis not in {"rpz", "xyz"}, NotImplementedError)
    p = _parse_parameterization(parameterization)
    if isinstance(names, str):
        names = [names]
    if basis == "xyz" and "phi" not in names:
        names = names + ["phi"]
    for name in names:
        if name not in data_index[p]:
            raise ValueError(f"Unrecognized value '{name}' for parameterization {p}.")
    bad_kwargs = kwargs.keys() - allowed_kwargs
    if len(bad_kwargs) > 0:
        raise ValueError(f"Unrecognized argument(s): {bad_kwargs}")

    for name in names:
        assert _has_params(name, params, p), f"Don't have params to compute {name}"
        assert _has_profiles(
            name, profiles, p
        ), f"Don't have profiles to compute {name}"
        assert _has_transforms(
            name, transforms, p
        ), f"Don't have transforms to compute {name}"

    if "grid" in transforms:

        def check_fun(name):
            reqs = data_index[p][name]["source_grid_requirement"]
            errorif(
                reqs and not hasattr(transforms["grid"], "source_grid"),
                AttributeError,
                f"Expected grid with attribute 'source_grid' to compute {name}. "
                f"Source grid should have coordinates: {reqs.get('coordinates')}.",
            )
            for req in reqs:
                errorif(
                    not hasattr(transforms["grid"].source_grid, req)
                    or reqs[req] != getattr(transforms["grid"].source_grid, req),
                    AttributeError,
                    f"Expected grid with '{req}:{reqs[req]}' to compute {name}.",
                )

        _ = _get_deps(
            p, names, set(), data, transforms["grid"].axis.size, check_fun=check_fun
        )

    if data is None:
        data = {}

    data = _compute(
        p,
        names,
        params=params,
        transforms=transforms,
        profiles=profiles,
        data=data,
        **kwargs,
    )

    # convert data from default 'rpz' basis to 'xyz' basis, if requested by the user
    if basis == "xyz":
        from .geom_utils import rpz2xyz, rpz2xyz_vec

        for name in data.keys():
            errorif(
                data_index[p][name]["dim"] == (3, 3),
                NotImplementedError,
                "Tensor quantities cannot be converted to Cartesian coordinates.",
            )
            if data_index[p][name]["dim"] == 3:  # only convert vector data
                if name in ["x", "center"]:
                    data[name] = rpz2xyz(data[name])
                else:
                    data[name] = rpz2xyz_vec(data[name], phi=data["phi"])

    return data


def _compute(
    parameterization, names, params, transforms, profiles, data=None, **kwargs
):
    """Same as above but without checking inputs for faster recursion.

    Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
    v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
    of the cylindrical coordinates R, ϕ, Z.

    We need to directly call this function in objectives, since the checks in above
    function are not compatible with JIT. This function computes given names while
    using recursion to compute dependencies. If you want to call this function, you
    cannot give the argument basis='xyz' since that will break the recursion. In that
    case, either call above function or manually convert the output to xyz basis.
    """
    assert kwargs.get("basis", "rpz") == "rpz", "_compute only works in rpz coordinates"
    parameterization = _parse_parameterization(parameterization)
    if isinstance(names, str):
        names = [names]
    if data is None:
        data = {}

    for name in names:
        if name in data:
            # don't compute something that's already been computed
            continue
        if not has_data_dependencies(
            parameterization, name, data, transforms["grid"].axis.size
        ):
            # then compute the missing dependencies
            data = _compute(
                parameterization,
                data_index[parameterization][name]["dependencies"]["data"],
                params=params,
                transforms=transforms,
                profiles=profiles,
                data=data,
                **kwargs,
            )
            if transforms["grid"].axis.size:
                data = _compute(
                    parameterization,
                    data_index[parameterization][name]["dependencies"][
                        "axis_limit_data"
                    ],
                    params=params,
                    transforms=transforms,
                    profiles=profiles,
                    data=data,
                    **kwargs,
                )
        # now compute the quantity
        data = data_index[parameterization][name]["fun"](
            params=params, transforms=transforms, profiles=profiles, data=data, **kwargs
        )
    return data


@execute_on_cpu
def get_data_deps(keys, obj, has_axis=False, basis="rpz", data=None):
    """Get list of keys needed to compute ``keys`` given already computed data.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions

    Returns
    -------
    deps : list of str
        Names of quantities needed to compute key.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    if not data:
        out = []
        for key in keys:
            out += _get_deps_1_key(p, key, has_axis)
        out = set(out)
    else:
        out = _get_deps(p, keys, deps=set(), data=data, has_axis=has_axis)
        out.difference_update(keys)
    if basis.lower() == "xyz":
        out.add("phi")
    return sorted(out)


def _get_deps_1_key(p, key, has_axis):
    """Gather all quantities required to compute ``key``.

    Parameters
    ----------
    p : str
        Type of object to compute for, eg Equilibrium, Curve, etc.
    key : str
        Name of the quantity to compute.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.

    Returns
    -------
    deps_1_key : list of str
        Dependencies required to compute ``key``.


    """
    if has_axis:
        if "full_with_axis_dependencies" in data_index[p][key]:
            return data_index[p][key]["full_with_axis_dependencies"]["data"]
    elif "full_dependencies" in data_index[p][key]:
        return data_index[p][key]["full_dependencies"]["data"]

    deps = data_index[p][key]["dependencies"]["data"]
    if len(deps) == 0:
        return deps
    out = deps.copy()  # to avoid modifying the data_index
    for dep in deps:
        out += _get_deps_1_key(p, dep, has_axis)
    if has_axis:
        axis_limit_deps = data_index[p][key]["dependencies"]["axis_limit_data"]
        out += axis_limit_deps.copy()  # to be safe
        for dep in axis_limit_deps:
            out += _get_deps_1_key(p, dep, has_axis)

    return sorted(set(out))


def _get_deps(parameterization, names, deps, data=None, has_axis=False, check_fun=None):
    """Gather all quantities required to compute ``names`` given already computed data.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    deps : set[str]
        Dependencies gathered so far.
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    check_fun : callable
        If provided, ``check_fun(name)`` is called before adding name to ``deps``.

    Returns
    -------
    deps : set[str]
        All additional quantities required to compute ``names``.

    """
    p = _parse_parameterization(parameterization)
    for name in names:
        if name not in deps and (data is None or name not in data):
            if check_fun is not None:
                check_fun(name)
            deps.add(name)
            deps = _get_deps(
                p,
                data_index[p][name]["dependencies"]["data"],
                deps,
                data,
                has_axis,
                check_fun,
            )
            if has_axis:
                deps = _get_deps(
                    p,
                    data_index[p][name]["dependencies"]["axis_limit_data"],
                    deps,
                    data,
                    has_axis,
                    check_fun,
                )
    return deps


def _grow_seeds(parameterization, seeds, search_space, has_axis=False):
    """Return ``seeds`` plus keys in ``search_space`` with dependency in ``seeds``.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    seeds : set[str]
        Keys to find paths toward.
    search_space : iterable of str
        Additional keys besides ``seeds`` to consider returning.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.

    Returns
    -------
    out : set[str]
        All keys in ``search_space`` that have a dependency in ``seeds``
        plus ``seeds``.

    """
    p = _parse_parameterization(parameterization)
    out = seeds.copy()
    for key in search_space:
        deps = data_index[p][key][
            "full_with_axis_dependencies" if has_axis else "full_dependencies"
        ]["data"]
        if not seeds.isdisjoint(deps):
            out.add(key)
    return out


@execute_on_cpu
def get_derivs(keys, obj, has_axis=False, basis="rpz"):
    """Get dict of derivative orders needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    derivs : dict[list, str]
        Orders of derivatives needed to compute key.
        Keys for R, Z, L, etc

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys

    def _get_derivs_1_key(key):
        if has_axis:
            if "full_with_axis_dependencies" in data_index[p][key]:
                return data_index[p][key]["full_with_axis_dependencies"]["transforms"]
        elif "full_dependencies" in data_index[p][key]:
            return data_index[p][key]["full_dependencies"]["transforms"]
        deps = [key] + get_data_deps(key, p, has_axis=has_axis, basis=basis)
        derivs = {}
        for dep in deps:
            for key, val in data_index[p][dep]["dependencies"]["transforms"].items():
                if key not in derivs:
                    derivs[key] = []
                derivs[key] += val
        return derivs

    derivs = {}
    for key in keys:
        derivs1 = _get_derivs_1_key(key)
        for key1, val in derivs1.items():
            if key1 not in derivs:
                derivs[key1] = []
            derivs[key1] += val
    return {key: np.unique(val, axis=0).tolist() for key, val in derivs.items()}


def get_profiles(keys, obj, grid=None, has_axis=False, basis="rpz"):
    """Get profiles needed to compute a given quantity on a given grid.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index.
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    grid : Grid
        Grid to compute quantity on.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    profiles : list of str or dict of Profile
        Profiles needed to compute key.
        if eq is None, returns a list of the names of profiles needed
        otherwise, returns a dict of Profiles
        Keys for pressure, iota, etc.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    has_axis = has_axis or (grid is not None and grid.axis.size)
    deps = list(keys) + get_data_deps(keys, p, has_axis=has_axis, basis=basis)
    profs = []
    for key in deps:
        profs += data_index[p][key]["dependencies"]["profiles"]
    profs = sorted(set(profs))
    if isinstance(obj, str) or inspect.isclass(obj):
        return profs
    # need to use copy here because profile may be None
    profiles = {name: copy.deepcopy(getattr(obj, name)) for name in profs}
    return profiles


@execute_on_cpu
def get_params(keys, obj, has_axis=False, basis="rpz"):
    """Get parameters needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    params : list[str] or dict[str, jnp.ndarray]
        Parameters needed to compute key.
        If eq is None, returns a list of the names of params needed
        otherwise, returns a dict of ndarray with keys for R_lmn, Z_lmn, etc.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    deps = list(keys) + get_data_deps(keys, p, has_axis=has_axis, basis=basis)
    params = []
    for key in deps:
        params += data_index[p][key]["dependencies"]["params"]
    if isinstance(obj, str) or inspect.isclass(obj):
        return params
    temp_params = {}
    for name in params:
        p = getattr(obj, name)
        if isinstance(p, dict):
            temp_params[name] = p.copy()
        else:
            temp_params[name] = jnp.atleast_1d(p)
    return temp_params


@execute_on_cpu
def get_transforms(
    keys, obj, grid, jitable=False, has_axis=False, basis="rpz", **kwargs
):
    """Get transforms needed to compute a given quantity on a given grid.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    grid : Grid
        Grid to compute quantity on
    jitable: bool
        Whether to skip certain checks so that this operation works under JIT
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    transforms : dict of Transform
        Transforms needed to compute key.
        Keys for R, Z, L, etc

    """
    from desc.basis import DoubleFourierSeries
    from desc.transform import Transform

    method = "jitable" if jitable or kwargs.get("method") == "jitable" else "auto"
    keys = [keys] if isinstance(keys, str) else keys
    has_axis = has_axis or (grid is not None and grid.axis.size)
    derivs = get_derivs(keys, obj, has_axis=has_axis, basis=basis)
    transforms = {"grid": grid}
    for c in derivs.keys():
        if hasattr(obj, c + "_basis"):  # regular stuff like R, Z, lambda etc.
            basis = getattr(obj, c + "_basis")
            # first check if we already have a transform with a compatible basis
            if not jitable:
                for transform in transforms.values():
                    if basis.equiv(getattr(transform, "basis", None)):
                        ders = np.unique(
                            np.vstack([derivs[c], transform.derivatives]), axis=0
                        ).astype(int)
                        # don't build until we know all the derivs we need
                        transform.change_derivatives(ders, build=False)
                        c_transform = transform
                        break
                else:  # if we didn't exit the loop early
                    c_transform = Transform(
                        grid,
                        basis,
                        derivs=derivs[c],
                        build=False,
                        method=method,
                    )
            else:  # don't perform checks if jitable=True as they are not jit-safe
                c_transform = Transform(
                    grid,
                    basis,
                    derivs=derivs[c],
                    build=False,
                    method=method,
                )
            transforms[c] = c_transform
        elif c == "B":  # used for Boozer transform
            transforms["B"] = Transform(
                grid,
                DoubleFourierSeries(
                    M=kwargs.get("M_booz", 2 * obj.M),
                    N=kwargs.get("N_booz", 2 * obj.N),
                    NFP=obj.NFP,
                    sym=obj.R_basis.sym,
                ),
                derivs=derivs["B"],
                build=False,
                build_pinv=True,
                method=method,
            )
        elif c == "w":  # used for Boozer transform
            transforms["w"] = Transform(
                grid,
                DoubleFourierSeries(
                    M=kwargs.get("M_booz", 2 * obj.M),
                    N=kwargs.get("N_booz", 2 * obj.N),
                    NFP=obj.NFP,
                    sym=obj.Z_basis.sym,
                ),
                derivs=derivs["w"],
                build=False,
                build_pinv=True,
                method=method,
            )
        elif c == "h":  # used for omnigenity
            rho = grid.nodes[:, 0]
            eta = (grid.nodes[:, 1] - np.pi) / 2
            alpha = grid.nodes[:, 2] * grid.NFP
            nodes = jnp.array([rho, eta, alpha]).T
            transforms["h"] = Transform(
                Grid(nodes, jitable=jitable),
                obj.x_basis,
                derivs=derivs["h"],
                build=True,
                build_pinv=False,
                method=method,
            )
        elif c not in transforms:  # possible other stuff lumped in with transforms
            transforms[c] = getattr(obj, c)

    # now build them
    for t in transforms.values():
        if hasattr(t, "build"):
            t.build()

    return transforms


def has_data_dependencies(parameterization, qty, data, axis=False):
    """Determine if we have the data needed to compute qty."""
    return _has_data(qty, data, parameterization) and (
        not axis or _has_axis_limit_data(qty, data, parameterization)
    )


def has_dependencies(parameterization, qty, params, transforms, profiles, data):
    """Determine if we have the ingredients needed to compute qty.

    Parameters
    ----------
    parameterization : str or class
        Type of thing we're checking dependencies for. eg desc.equilibrium.Equilibrium
    qty : str
        Name of something from the data index.
    params : dict[str, jnp.ndarray]
        Dictionary of parameters we have.
    transforms : dict[str, Transform]
        Dictionary of transforms we have.
    profiles : dict[str, Profile]
        Dictionary of profiles we have.
    data : dict[str, jnp.ndarray]
        Dictionary of what we've computed so far.

    Returns
    -------
    has_dependencies : bool
        Whether we have what we need.
    """
    return (
        _has_data(qty, data, parameterization)
        and (
            not transforms["grid"].axis.size
            or _has_axis_limit_data(qty, data, parameterization)
        )
        and _has_params(qty, params, parameterization)
        and _has_profiles(qty, profiles, parameterization)
        and _has_transforms(qty, transforms, parameterization)
    )


def _has_data(qty, data, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["data"]
    return all(d in data for d in deps)


def _has_axis_limit_data(qty, data, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["axis_limit_data"]
    return all(d in data for d in deps)


def _has_params(qty, params, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["params"]
    return all(d in params for d in deps)


def _has_profiles(qty, profiles, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["profiles"]
    return all(d in profiles for d in deps)


def _has_transforms(qty, transforms, parameterization):
    p = _parse_parameterization(parameterization)
    flags = {}
    derivs = data_index[p][qty]["dependencies"]["transforms"]
    for key in derivs.keys():
        if key not in transforms:
            return False
        else:
            flags[key] = np.array(
                [d in transforms[key].derivatives.tolist() for d in derivs[key]]
            ).all()
    return all(flags.values())


def dot(a, b, axis=-1):
    """Batched vector dot product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = sum(a*b, axis=axis)

    """
    return jnp.sum(a * b, axis=axis, keepdims=False)


def cross(a, b, axis=-1):
    """Batched vector cross product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = a x b

    """
    return jnp.cross(a, b, axis=axis)


def safenorm(x, ord=None, axis=None, fill=0, threshold=0):
    """Like jnp.linalg.norm, but without nan gradient at x=0.

    Parameters
    ----------
    x : ndarray
        Vector or array to norm.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of norm.
    axis : {None, int, 2-tuple of ints}, optional
        Axis to take norm along.
    fill : float, ndarray, optional
        Value to return where x is zero.
    threshold : float >= 0
        How small is x allowed to be.

    """
    is_zero = (jnp.abs(x) <= threshold).all(axis=axis, keepdims=True)
    y = jnp.where(is_zero, jnp.ones_like(x), x)  # replace x with ones if is_zero
    n = jnp.linalg.norm(y, ord=ord, axis=axis)
    n = jnp.where(is_zero.squeeze(), fill, n)  # replace norm with zero if is_zero
    return n


def safenormalize(x, ord=None, axis=None, fill=0, threshold=0):
    """Normalize a vector to unit length, but without nan gradient at x=0.

    Parameters
    ----------
    x : ndarray
        Vector or array to norm.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of norm.
    axis : {None, int, 2-tuple of ints}, optional
        Axis to take norm along.
    fill : float, ndarray, optional
        Value to return where x is zero.
    threshold : float >= 0
        How small is x allowed to be.

    """
    is_zero = (jnp.abs(x) <= threshold).all(axis=axis, keepdims=True)
    y = jnp.where(is_zero, jnp.ones_like(x), x)  # replace x with ones if is_zero
    n = safenorm(x, ord, axis, fill, threshold) * jnp.ones_like(x)
    # return unit vector with equal components if norm <= threshold
    return jnp.where(n <= threshold, jnp.ones_like(y) / jnp.sqrt(y.size), y / n)


def safediv(a, b, fill=0, threshold=0):
    """Divide a/b with guards for division by zero.

    Parameters
    ----------
    a, b : ndarray
        Numerator and denominator.
    fill : float, ndarray, optional
        Value to return where b is zero.
    threshold : float >= 0
        How small is b allowed to be.
    """
    mask = jnp.abs(b) <= threshold
    num = jnp.where(mask, fill, a)
    den = jnp.where(mask, 1, b)
    return num / den


def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulatively integrate y(x) using the composite trapezoidal rule.

    Taken from SciPy, but changed NumPy references to JAX.NumPy:
        https://github.com/scipy/scipy/blob/v1.10.1/scipy/integrate/_quadrature.py

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically, this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    """
    y = jnp.asarray(y)
    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        if x.ndim == 1:
            d = jnp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        else:
            d = jnp.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )

    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = jnp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not jnp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = jnp.concatenate(
            [jnp.full(shape, initial, dtype=res.dtype), res], axis=axis
        )

    return res


def _get_grid_surface(grid, surface_label):
    """Return grid quantities associated with the given surface label.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta.

    Returns
    -------
    unique_size : int
        The number of the unique values of the surface_label.
    inverse_idx : ndarray
        Indexing array to go from unique values to full grid.
    spacing : ndarray
        The relevant columns of grid.spacing.
    has_endpoint_dupe : bool
        Whether this surface label's nodes have a duplicate at the endpoint
        of a periodic domain. (e.g. a node at 0 and 2π).
    has_idx : bool
        Whether the grid knows the number of unique nodes and inverse idx.

    """
    assert surface_label in {"rho", "poloidal", "zeta"}
    if surface_label == "rho":
        spacing = grid.spacing[:, 1:]
        has_endpoint_dupe = False
    elif surface_label == "poloidal":
        spacing = grid.spacing[:, [0, 2]]
        has_endpoint_dupe = isinstance(grid, LinearGrid) and grid._poloidal_endpoint
    else:
        spacing = grid.spacing[:, :2]
        has_endpoint_dupe = isinstance(grid, LinearGrid) and grid._toroidal_endpoint
    has_idx = hasattr(grid, f"num_{surface_label}") and hasattr(
        grid, f"_inverse_{surface_label}_idx"
    )
    unique_size = getattr(grid, f"num_{surface_label}", -1)
    inverse_idx = getattr(grid, f"_inverse_{surface_label}_idx", jnp.array([]))
    return unique_size, inverse_idx, spacing, has_endpoint_dupe, has_idx


def line_integrals(
    grid,
    q=jnp.array([1.0]),
    line_label="poloidal",
    fix_surface=("rho", 1.0),
    expand_out=True,
    tol=1e-14,
):
    """Compute line integrals over curves covering the given surface.

    As an example, by specifying the combination of ``line_label="poloidal"`` and
    ``fix_surface=("rho", 1.0)``, the intention is to integrate along the
    outermost perimeter of a particular zeta surface (toroidal cross-section),
    for each zeta surface in the grid.

    Notes
    -----
        It is assumed that the integration curve has length 1 when the line
        label is rho and length 2π when the line label is theta or zeta.
        You may want to multiply the input by the line length Jacobian.

        The grid must have nodes on the specified surface in ``fix_surface``.

        Correctness is not guaranteed on grids with duplicate nodes.
        An attempt to print a warning is made if the given grid has duplicate
        nodes and is one of the predefined grid types
        (``Linear``, ``Concentric``, ``Quadrature``).
        If the grid is custom, no attempt is made to warn.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
        The first dimension of the array should have size ``grid.num_nodes``.
        When ``q`` is n-dimensional, the intention is to integrate,
        over the domain parameterized by rho, poloidal, and zeta,
        an n-dimensional function over the previously mentioned domain.
    line_label : str
        The coordinate curve to compute the integration over.
        To clarify, a theta (poloidal) curve is the intersection of a
        rho surface (flux surface) and zeta (toroidal) surface.
    fix_surface : str, float
        A tuple of the form: label, value.
        ``fix_surface`` label should differ from ``line_label``.
        By default, ``fix_surface`` is chosen to be the flux surface at rho=1.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    integrals : ndarray
        Line integrals of the input over curves covering the given surface.
        By default, the returned array has the same shape as the input.

    """
    line_label = grid.get_label(line_label)
    fix_label = grid.get_label(fix_surface[0])
    errorif(
        line_label == fix_label,
        msg="There is no valid use for this combination of inputs.",
    )
    errorif(
        line_label != "poloidal" and isinstance(grid, ConcentricGrid),
        msg="ConcentricGrid should only be used for poloidal line integrals.",
    )
    warnif(
        isinstance(grid, LinearGrid) and grid.endpoint,
        msg="Correctness not guaranteed on grids with duplicate nodes.",
    )
    # Generate a new quantity q_prime which is zero everywhere
    # except on the fixed surface, on which q_prime takes the value of q.
    # Then forward the computation to surface_integrals().
    # The differential element of the line integral, denoted dl,
    # should correspond to the line label's spacing.
    # The differential element of the surface integral is
    # ds = dl * fix_surface_dl, so we scale q_prime by 1 / fix_surface_dl.
    axis = {"rho": 0, "poloidal": 1, "zeta": 2}
    column_id = axis[fix_label]
    mask = grid.nodes[:, column_id] == fix_surface[1]
    q_prime = (mask * jnp.atleast_1d(q).T / grid.spacing[:, column_id]).T
    (surface_label,) = axis.keys() - {line_label, fix_label}
    return surface_integrals(grid, q_prime, surface_label, expand_out, tol)


def surface_integrals(
    grid, q=jnp.array([1.0]), surface_label="rho", expand_out=True, tol=1e-14
):
    """Compute a surface integral for each surface in the grid.

    Notes
    -----
        It is assumed that the integration surface has area 4π² when the
        surface label is rho and area 2π when the surface label is theta or
        zeta. You may want to multiply the input by the surface area Jacobian.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
        The first dimension of the array should have size ``grid.num_nodes``.
        When ``q`` is n-dimensional, the intention is to integrate,
        over the domain parameterized by rho, poloidal, and zeta,
        an n-dimensional function over the previously mentioned domain.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the integration over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    integrals : ndarray
        Surface integral of the input over each surface in the grid.
        By default, the returned array has the same shape as the input.

    """
    return surface_integrals_map(grid, surface_label, expand_out, tol)(q)


def surface_integrals_map(grid, surface_label="rho", expand_out=True, tol=1e-14):
    """Returns a method to compute any surface integral for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the integration over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    function : callable
        Method to compute any surface integral of the input ``q`` over each
        surface in the grid with code: ``function(q)``.

    """
    surface_label = grid.get_label(surface_label)
    warnif(
        surface_label == "poloidal" and isinstance(grid, ConcentricGrid),
        msg="Integrals over constant poloidal surfaces"
        " are poorly defined for ConcentricGrid.",
    )
    unique_size, inverse_idx, spacing, has_endpoint_dupe, has_idx = _get_grid_surface(
        grid, surface_label
    )
    spacing = jnp.prod(spacing, axis=1)

    # Todo: Define mask as a sparse matrix once sparse matrices are no longer
    #       experimental in jax.
    if has_idx:
        # The ith row of masks is True only at the indices which correspond to the
        # ith surface. The integral over the ith surface is the dot product of the
        # ith row vector and the integrand defined over all the surfaces.
        mask = inverse_idx == jnp.arange(unique_size)[:, jnp.newaxis]
        # Imagine a torus cross-section at zeta=π.
        # A grid with a duplicate zeta=π node has 2 of those cross-sections.
        #     In grid.py, we multiply by 1/n the areas of surfaces with
        # duplicity n. This prevents the area of that surface from being
        # double-counted, as surfaces with the same node value are combined
        # into 1 integral, which sums their areas. Thus, if the zeta=π
        # cross-section has duplicity 2, we ensure that the area on the zeta=π
        # surface will have the correct total area of π+π = 2π.
        #     An edge case exists if the duplicate surface has nodes with
        # different values for the surface label, which only occurs when
        # has_endpoint_dupe is true. If ``has_endpoint_dupe`` is true, this grid
        # has a duplicate surface at surface_label=0 and
        # surface_label=max surface value. Although the modulo of these values
        # are equal, their numeric values are not, so the integration
        # would treat them as different surfaces. We solve this issue by
        # combining the indices corresponding to the integrands of the duplicated
        # surface, so that the duplicate surface is treated as one, like in the
        # previous paragraph.
        mask = cond(
            has_endpoint_dupe,
            lambda _: put(mask, jnp.array([0, -1]), mask[0] | mask[-1]),
            lambda _: mask,
            operand=None,
        )
    else:
        # If we don't have the idx attributes, we are forced to expand out.
        errorif(
            not has_idx and not expand_out,
            msg=f"Grid lacks attributes 'num_{surface_label}' and "
            f"'inverse_{surface_label}_idx', so this method "
            f"can't satisfy the request expand_out={expand_out}.",
        )
        # don't try to expand if already expanded
        expand_out = expand_out and has_idx
        axis = {"rho": 0, "poloidal": 1, "zeta": 2}[surface_label]
        # Converting nodes from numpy.ndarray to jaxlib.xla_extension.ArrayImpl
        # reduces memory usage by > 400% for the forward computation and Jacobian.
        nodes = jnp.asarray(grid.nodes[:, axis])
        # This branch will execute for custom grids, which don't have a use
        # case for having duplicate nodes, so we don't bother to modulo nodes
        # by 2pi or 2pi/NFP.
        mask = jnp.abs(nodes - nodes[:, jnp.newaxis]) <= tol
        # The above implementation was benchmarked to be more efficient than
        # alternatives with explicit loops in GitHub pull request #934.

    def integrate(q=jnp.array([1.0])):
        """Compute a surface integral for each surface in the grid.

        Notes
        -----
            It is assumed that the integration surface has area 4π² when the
            surface label is rho and area 2π when the surface label is theta or
            zeta. You may want to multiply the input by the surface area Jacobian.

        Parameters
        ----------
        q : ndarray
            Quantity to integrate.
            The first dimension of the array should have size ``grid.num_nodes``.
            When ``q`` is n-dimensional, the intention is to integrate,
            over the domain parameterized by rho, poloidal, and zeta,
            an n-dimensional function over the previously mentioned domain.

        Returns
        -------
        integrals : ndarray
            Surface integral of the input over each surface in the grid.

        """
        integrands = (spacing * jnp.nan_to_num(q).T).T
        integrals = jnp.tensordot(mask, integrands, axes=1)
        return grid.expand(integrals, surface_label) if expand_out else integrals

    return integrate


def surface_averages(
    grid,
    q,
    sqrt_g=jnp.array([1.0]),
    surface_label="rho",
    denominator=None,
    expand_out=True,
    tol=1e-14,
):
    """Compute a surface average for each surface in the grid.

    Notes
    -----
        Implements the flux-surface average formula given by equation 4.9.11 in
        W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to average.
        The first dimension of the array should have size ``grid.num_nodes``.
        When ``q`` is n-dimensional, the intention is to average,
        over the domain parameterized by rho, poloidal, and zeta,
        an n-dimensional function over the previously mentioned domain.
    sqrt_g : ndarray
        Coordinate system Jacobian determinant; see ``data_index["sqrt(g)"]``.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the average over.
    denominator : ndarray
        By default, the denominator is computed as the surface integral of
        ``sqrt_g``. This parameter can optionally be supplied to avoid
        redundant computations or to use a different denominator to compute
        the average. This array should broadcast with arrays of size
        ``grid.num_nodes`` (``grid.num_surface_label``) if ``expand_out``
        is true (false).
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    averages : ndarray
        Surface average of the input over each surface in the grid.
        By default, the returned array has the same shape as the input.

    """
    return surface_averages_map(grid, surface_label, expand_out, tol)(
        q, sqrt_g, denominator
    )


def surface_averages_map(grid, surface_label="rho", expand_out=True, tol=1e-14):
    """Returns a method to compute any surface average for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the average over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    function : callable
        Method to compute any surface average of the input ``q`` and optionally
        the volume Jacobian ``sqrt_g`` over each surface in the grid with code:
        ``function(q, sqrt_g)``.

    """
    surface_label = grid.get_label(surface_label)
    has_idx = hasattr(grid, f"num_{surface_label}") and hasattr(
        grid, f"_inverse_{surface_label}_idx"
    )
    # If we don't have the idx attributes, we are forced to expand out.
    errorif(
        not has_idx and not expand_out,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', so this method "
        f"can't satisfy the request expand_out={expand_out}.",
    )
    integrate = surface_integrals_map(
        grid, surface_label, expand_out=not has_idx, tol=tol
    )
    # don't try to expand if already expanded
    expand_out = expand_out and has_idx

    def _surface_averages(q, sqrt_g=jnp.array([1.0]), denominator=None):
        """Compute a surface average for each surface in the grid.

        Notes
        -----
            Implements the flux-surface average formula given by equation 4.9.11 in
            W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.

        Parameters
        ----------
        q : ndarray
            Quantity to average.
            The first dimension of the array should have size ``grid.num_nodes``.
            When ``q`` is n-dimensional, the intention is to average,
            over the domain parameterized by rho, poloidal, and zeta,
            an n-dimensional function over the previously mentioned domain.
        sqrt_g : ndarray
            Coordinate system Jacobian determinant; see ``data_index["sqrt(g)"]``.
        denominator : ndarray
            By default, the denominator is computed as the surface integral of
            ``sqrt_g``. This parameter can optionally be supplied to avoid
            redundant computations or to use a different denominator to compute
            the average. This array should broadcast with arrays of size
            ``grid.num_nodes`` (``grid.num_surface_label``) if ``expand_out``
            is true (false).

        Returns
        -------
        averages : ndarray
            Surface average of the input over each surface in the grid.

        """
        q, sqrt_g = jnp.atleast_1d(q, sqrt_g)
        numerator = integrate((sqrt_g * q.T).T)
        # memory optimization to call expand() at most once
        if denominator is None:
            # skip integration if constant
            denominator = (
                (4 * jnp.pi**2 if surface_label == "rho" else 2 * jnp.pi) * sqrt_g
                if sqrt_g.size == 1
                else integrate(sqrt_g)
            )
            averages = (numerator.T / denominator).T
            if expand_out:
                averages = grid.expand(averages, surface_label)
        else:
            if expand_out:
                # implies denominator given with size grid.num_nodes
                numerator = grid.expand(numerator, surface_label)
            averages = (numerator.T / denominator).T
        return averages

    return _surface_averages


def surface_integrals_transform(grid, surface_label="rho"):
    """Returns a method to compute any integral transform over each surface in grid.

    The returned method takes an array input ``q`` and returns an array output.

    Given a set of kernel functions in ``q``, each parameterized by at most
    five variables, the returned method computes an integral transform,
    reducing ``q`` to a set of functions of at most three variables.

    Define the domain D = u₁ × u₂ × u₃ and the codomain C = u₄ × u₅ × u₆.
    For every surface of constant u₁ in the domain, the returned method
    evaluates the transform Tᵤ₁ : u₂ × u₃ × C → C, where Tᵤ₁ projects
    away the parameters u₂ and u₃ via an integration of the given kernel
    function Kᵤ₁ over the corresponding surface of constant u₁.

    Notes
    -----
        It is assumed that the integration surface has area 4π² when the
        surface label is rho and area 2π when the surface label is theta or
        zeta. You may want to multiply the input ``q`` by the surface area
        Jacobian.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the integration over.
        These correspond to the domain parameters discussed in this method's
        description. In particular, ``surface_label`` names u₁.

    Returns
    -------
    function : callable
        Method to compute any surface integral transform of the input ``q`` over
        each surface in the grid with code: ``function(q)``.

        The first dimension of ``q`` should always discretize some function, g,
        over the domain, and therefore, have size ``grid.num_nodes``.
        The second dimension may discretize some function, f, over the
        codomain, and therefore, have size that matches the desired number of
        points at which the output is evaluated.

        This method can also be used to compute the output one point at a time,
        in which case ``q`` can have shape (``grid.num_nodes``, ).

        Input
        -----
        If ``q`` has one dimension, then it should have shape
        (``grid.num_nodes``, ).
        If ``q`` has multiple dimensions, then it should have shape
        (``grid.num_nodes``, *f.shape).

        Output
        ------
        Each element along the first dimension of the returned array, stores
        Tᵤ₁ for a particular surface of constant u₁ in the given grid.
        The order is sorted in increasing order of the values which specify u₁.

        If ``q`` has one dimension, the returned array has shape
        (grid.num_surface_label, ).
        If ``q`` has multiple dimensions, the returned array has shape
        (grid.num_surface_label, *f.shape).

    """
    # Expansion should not occur here. The typical use case of this method is to
    # transform into the computational domain, so the second dimension that
    # discretizes f over the codomain will typically have size grid.num_nodes
    # to broadcast with quantities in data_index.
    surface_label = grid.get_label(surface_label)
    has_idx = hasattr(grid, f"num_{surface_label}") and hasattr(
        grid, f"_inverse_{surface_label}_idx"
    )
    errorif(
        not has_idx,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', which are required for this function.",
    )
    return surface_integrals_map(grid, surface_label, expand_out=False)


def surface_variance(
    grid,
    q,
    weights=jnp.array([1.0]),
    bias=False,
    surface_label="rho",
    expand_out=True,
    tol=1e-14,
):
    """Compute the weighted sample variance of ``q`` on each surface of the grid.

    Computes nₑ / (nₑ − b) * (∑ᵢ₌₁ⁿ (qᵢ − q̅)² wᵢ) / (∑ᵢ₌₁ⁿ wᵢ).
    wᵢ is the weight assigned to qᵢ given by the product of ``weights[i]`` and
       the differential surface area element (not already weighted by the area
       Jacobian) at the node where qᵢ is evaluated,
    q̅ is the weighted mean of q,
    b is 0 if the biased sample variance is to be returned and 1 otherwise,
    n is the number of samples on a surface, and
    nₑ ≝ (∑ᵢ₌₁ⁿ wᵢ)² / ∑ᵢ₌₁ⁿ wᵢ² is the effective number of samples.

    As the weights wᵢ approach each other, nₑ approaches n, and the output
    converges to ∑ᵢ₌₁ⁿ (qᵢ − q̅)² / (n − b).

    Notes
    -----
        There are three different methods to unbias the variance of a weighted
        sample so that the computed variance better estimates the true variance.
        Whether the method is correct for a particular use case depends on what
        the weights assigned to each sample represent.

        This function implements the first case, where the weights are not random
        and are intended to assign more weight to some samples for reasons
        unrelated to differences in uncertainty between samples. See
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights.

        The second case is when the weights are intended to assign more weight
        to samples with less uncertainty. See
        https://en.wikipedia.org/wiki/Inverse-variance_weighting.
        The unbiased sample variance for this case is obtained by replacing the
        effective number of samples in the formula this function implements,
        nₑ, with the actual number of samples n.

        The third case is when the weights denote the integer frequency of each
        sample. See
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights.
        This is indeed a distinct case from the above two because here the
        weights encode additional information about the distribution.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to compute the sample variance.
    weights : ndarray
        Weight assigned to each sample of ``q``.
        A good candidate for this parameter is the surface area Jacobian.
    bias : bool
        If this condition is true, then the biased estimator of the sample
        variance is returned. This is desirable if you are only concerned with
        computing the variance of the given set of numbers and not the
        distribution the numbers are (potentially) sampled from.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the variance over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    variance : ndarray
        Variance of the given weighted sample over each surface in the grid.
        By default, the returned array has the same shape as the input.

    """
    surface_label = grid.get_label(surface_label)
    _, _, spacing, _, has_idx = _get_grid_surface(grid, surface_label)
    # If we don't have the idx attributes, we are forced to expand out.
    errorif(
        not has_idx and not expand_out,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', so this method "
        f"can't satisfy the request expand_out={expand_out}.",
    )
    integrate = surface_integrals_map(
        grid, surface_label, expand_out=not has_idx, tol=tol
    )

    v1 = integrate(weights)
    v2 = integrate(weights**2 * jnp.prod(spacing, axis=-1))
    # effective number of samples per surface
    n_e = v1**2 / v2
    # analogous to Bessel's bias correction
    correction = n_e / (n_e - (not bias))

    q = jnp.atleast_1d(q)
    # compute variance in two passes to avoid catastrophic round off error
    mean = (integrate((weights * q.T).T).T / v1).T
    if has_idx:  # guard so that we don't try to expand when already expanded
        mean = grid.expand(mean, surface_label)
    variance = (correction * integrate((weights * ((q - mean) ** 2).T).T).T / v1).T
    if expand_out and has_idx:
        return grid.expand(variance, surface_label)
    else:
        return variance


def surface_max(grid, x, surface_label="rho"):
    """Get the max of x for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Quantity to find max.
        The array should have size grid.num_nodes.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute max over.

    Returns
    -------
    maxs : ndarray
        Maximum of x over each surface in grid.
        The returned array has the same shape as the input.

    """
    return -surface_min(grid, -x, surface_label)


def surface_min(grid, x, surface_label="rho"):
    """Get the min of x for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Quantity to find min.
        The array should have size grid.num_nodes.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute min over.

    Returns
    -------
    mins : ndarray
        Minimum of x over each surface in grid.
        The returned array has the same shape as the input.

    """
    surface_label = grid.get_label(surface_label)
    unique_size, inverse_idx, _, _, has_idx = _get_grid_surface(grid, surface_label)
    errorif(
        not has_idx,
        NotImplementedError,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', which are required for this function.",
    )
    inverse_idx = jnp.asarray(inverse_idx)
    x = jnp.asarray(x)
    mins = jnp.full(unique_size, jnp.inf)

    def body(i, mins):
        mins = put(mins, inverse_idx[i], jnp.minimum(x[i], mins[inverse_idx[i]]))
        return mins

    mins = fori_loop(0, inverse_idx.size, body, mins)
    # The above implementation was benchmarked to be more efficient than
    # alternatives without explicit loops in GitHub pull request #501.
    return grid.expand(mins, surface_label)

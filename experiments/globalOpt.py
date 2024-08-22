import numpy as np
import matplotlib.pyplot as plt
import desc.io
from desc.grid import LinearGrid, ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
    AspectRatio,
    ForceBalance,
    QuasisymmetryTwoTerm,
    RotationalTransform
)
from desc.optimize import Optimizer
from desc.plotting import (
    plot_grid,
    plot_boozer_modes,
    plot_boozer_surface,
    plot_qs_error,
    plot_boundaries,
    plot_boundary
)

eq_fam = desc.io.load("inputs/nfp2_QA.h5")
eq_init = eq_fam[-1]



idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

eq_qs = eq_init.copy()

constraints = (
    ForceBalance(eq=eq_qs),
    FixBoundaryR(eq=eq_qs, modes=R_modes),
    FixBoundaryZ(eq=eq_qs, modes=Z_modes),
    FixPressure(eq=eq_qs)
)


optimizer = Optimizer("proximal-lsq-exact")

grid_vol = ConcentricGrid(
    L=eq_init.L_grid,
    M=eq_init.M_grid,
    N=eq_init.N_grid,
    NFP=eq_init.NFP,
    sym=eq_init.sym
)

plot_boundary(eq_qs, figsize=(8,8))

'''objective_f = ObjectiveFunction((
    QuasisymmetryTwoTerm(eq=eq_qs, grid=grid_vol, helicity=(1, eq_init.NFP), weight=1),
    AspectRatio(eq=eq_qs, target=6, weight=1),
    RotationalTransform(eq=eq_qs, target=0.42, weight=1))
)

eq_qs, result = eq_qs.optimize(
    objective=objective_f,
    constraints=constraints,
    optimizer=optimizer,
    ftol = 0.01,
    copy=False,
    verbose=1
)'''



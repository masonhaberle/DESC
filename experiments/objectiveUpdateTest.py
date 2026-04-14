import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import desc.io
from desc.grid import LinearGrid, ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixModeR,
    FixModeZ,
    FixPressure,
    FixCurrent,
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
    plot_boundary,
    plot_3d
)

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile


eq_fam = desc.io.load("inputs/lpQH.h5")
eq_init = eq_fam[-1]
eq_qs = eq_init.copy()

eq_qs.change_resolution(L=3, M=3, N=3)
eq_qs.surface = eq_qs.get_surface_at(rho=1.0)

grid_vol = LinearGrid(
    M=eq_qs.M_grid,
    N=eq_qs.N_grid,
    NFP=eq_qs.NFP,
    rho=np.arange(0.1,1.1,0.1),
    sym=eq_qs.sym
)

optimizer = Optimizer("Turbo1")



objective_f = ObjectiveFunction((
    AspectRatio(eq=eq_qs, target=8, weight=20),
    ForceBalance(eq=eq_qs, weight=20),
    QuasisymmetryTwoTerm(eq=eq_qs, grid=grid_vol, helicity=(1, eq_qs.NFP), weight=1),
    RotationalTransform(eq=eq_qs, grid=grid_vol, target=1.24, weight=20)
))
    
objective_f.build()
print("Debug: ", objective_f.compute_scalar(objective_f.x()))

modarray = objective_f.x()
modarray = modarray.at[0].set(-modarray[0])
print("Debug: ", objective_f.compute_scalar(modarray))

print("Debug: ", objective_f.compute_scalar(objective_f.x()))

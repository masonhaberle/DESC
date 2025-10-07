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

max_time = int(sys.argv[1])
box_size = float(sys.argv[2])
maxMode = int(sys.argv[3])


eq_fam = desc.io.load("inputs/lpQH.h5")
eq_init = eq_fam[-1]
eq_qs = eq_init.copy()

eq_qs.change_resolution(L=5, M=5, N=5)
eq_qs.surface = eq_qs.get_surface_at(rho=1.0)

grid_vol = LinearGrid(
    M=eq_qs.M_grid,
    N=eq_qs.N_grid,
    NFP=eq_qs.NFP,
    rho=np.arange(0.1,1.1,0.1),
    sym=eq_qs.sym
)

eq_qs = eq_init.copy()
optimizer = Optimizer("Turbo1")

fixed_R_modes = np.vstack(([0,0,0], eq_qs.surface.R_basis.modes[np.max(np.abs(eq_qs.surface.R_basis.modes), 1) > maxMode, :]))
fixed_Z_modes = eq_qs.surface.Z_basis.modes[np.max(np.abs(eq_qs.surface.Z_basis.modes), 1) > maxMode, :]
maxModeInt = maxMode

#Attempt to fix interior
fixed_R_modes_int = eq_qs.R_basis.modes[np.max(np.abs(eq_qs.R_basis.modes), 1) > maxModeInt, :]
fixed_Z_modes_int = eq_qs.Z_basis.modes[np.max(np.abs(eq_qs.Z_basis.modes), 1) > maxModeInt, :]

objective_f = ObjectiveFunction((
    AspectRatio(eq=eq_qs, target=8, weight=10),
    ForceBalance(eq=eq_qs, weight=1),
    QuasisymmetryTwoTerm(eq=eq_qs, grid=grid_vol, helicity=(1, eq_qs.NFP), weight=1),
    RotationalTransform(eq=eq_qs, grid=grid_vol, target=1.24, weight=10)
))

constraints = (
    FixBoundaryR(eq=eq_qs, modes=fixed_R_modes),
    FixBoundaryZ(eq=eq_qs, modes=fixed_Z_modes),
    FixModeR(eq=eq_qs, modes=fixed_R_modes_int),
    FixModeZ(eq=eq_qs, modes=fixed_Z_modes_int),
    FixPressure(eq=eq_qs),
    FixPsi(eq=eq_qs),
    FixCurrent(eq=eq_qs),
)

global_eq_qs, result = eq_qs.optimize(
    objective=objective_f,
    constraints=constraints,
    optimizer=optimizer,
    copy=True,
    verbose=1,
    options={"max_time":max_time, "trust_regions":1, "box_size":box_size, "training_steps": 20, "batch_size":30}
)

outputname = "Output_T" + sys.argv[1] + "_B" + sys.argv[2] + "_M" + sys.argv[3]
try:
    os.mkdir(outputname)
except(OSError):
    print("Output dir already exists, overwriting")


global_eq_qs.save(outputname+"/eq_result.h5")

eq_qs.params_dict = result.eqparams_init[0]
eq_qs.save(outputname+"/eq_init.h5")

with open(outputname+"/obj_history.txt", "w") as obj_history:
    for item in result.allfun:
        obj_history.write(f"{item[0]:.5f}\n")

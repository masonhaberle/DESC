import numpy as np
import matplotlib.pyplot as plt
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

grid_vol = LinearGrid(
    M=eq_qs.M_grid,
    N=eq_qs.N_grid,
    NFP=eq_qs.NFP,
    rho=np.arange(0.1,1.1,0.1),
    sym=eq_qs.sym
)

eq_qs = eq_init.copy()
optimizer = Optimizer("Turbo1")

for maxMode in range(1,2):
    fixed_R_modes = np.vstack(([0,0,0], eq_qs.surface.R_basis.modes[np.max(np.abs(eq_qs.surface.R_basis.modes), 1) > maxMode, :]))
    
    fixed_Z_modes = eq_qs.surface.Z_basis.modes[np.max(np.abs(eq_qs.surface.Z_basis.modes), 1) > maxMode, :]
    maxModeInt = maxMode

    #Attempt to fix interior
    fixed_R_modes_int = eq_qs.R_basis.modes[np.max(np.abs(eq_qs.R_basis.modes), 1) > maxModeInt, :]
    fixed_Z_modes_int = eq_qs.Z_basis.modes[np.max(np.abs(eq_qs.Z_basis.modes), 1) > maxModeInt, :]

    
    constWeight = 10
    
    objective_f = ObjectiveFunction((
        AspectRatio(eq=eq_qs, target=6, weight=1),
        #ForceBalance(eq=eq_qs, weight=10),
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
        verbose=2,
        maxiter=500,
        options={"trust_regions":1, "box_size":1e-1, "training_steps": 50, "batch_size":100}
    )
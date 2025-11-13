import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import plotly.io as pio

import desc.io
from desc.grid import LinearGrid
from desc.plotting import (
    plot_grid,
    plot_boozer_modes,
    plot_boozer_surface,
    plot_qs_error,
    plot_boundaries,
    plot_boundary,
    plot_3d
)

folder = sys.argv[1]
path = os.path.join(folder, "eq_result.h5")
eq = desc.io.load(path)

grid = LinearGrid(
    rho=1,
    theta=np.linspace(0, 2 * np.pi, 300),
    zeta=np.linspace(0, 2 * np.pi, 300),
    sym=False,
)

plt.figure("Result Boozer")
#fig = plot_3d(eq, "|B|", grid=grid, figsize=(5,5))
plot_boozer_surface(eq)
plt.savefig(folder + "/result_boozer.png")


plt.figure("Result Boundary")
plot_boundary(eq)
plt.savefig(folder + "/result_boundary.png")

plt.figure("Result 3D")
plot_3d(eq, "|B|")
plt.savefig(folder + "/result_3d.png")


path = os.path.join(folder, "obj_history.txt")
objectives = []
maxval = 1000000000
with open(path) as obj_history:
    for line in obj_history:
        val = float(line)
        val = min(val, maxval)
        objectives.append(val)

plt.figure("Result Objective")
fig = plt.plot(objectives)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(folder + "/result_obj.png")



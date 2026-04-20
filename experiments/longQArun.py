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
orig_max_mode = int(sys.argv[3])
num_cont = int(sys.argv[4])

print(max_time)

eq_fam = desc.io.load("DESC/experiments/inputs/precise_QA.h5")
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

optimizer = Optimizer("Turbo1")

results_array = []

for i in range(num_cont):
    maxMode = orig_max_mode + i
    fixed_R_modes = np.vstack(([0,0,0], eq_qs.surface.R_basis.modes[np.max(np.abs(eq_qs.surface.R_basis.modes), 1) > maxMode, :]))
    fixed_Z_modes = eq_qs.surface.Z_basis.modes[np.max(np.abs(eq_qs.surface.Z_basis.modes), 1) > maxMode, :]
    maxModeInt = maxMode

    #Attempt to fix interior
    fixed_R_modes_int = eq_qs.R_basis.modes[np.max(np.abs(eq_qs.R_basis.modes), 1) > maxModeInt, :]
    fixed_Z_modes_int = eq_qs.Z_basis.modes[np.max(np.abs(eq_qs.Z_basis.modes), 1) > maxModeInt, :]

    objective_f = ObjectiveFunction((
        AspectRatio(eq=eq_qs, target=6, weight=20),
        ForceBalance(eq=eq_qs, grid=grid_vol, weight=20),
        QuasisymmetryTwoTerm(eq=eq_qs, grid=grid_vol, helicity=(1, 0), weight=1),
        RotationalTransform(eq=eq_qs, grid=grid_vol, target=0.42, weight=20)
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

    eq_qs, result = eq_qs.optimize(
        objective=objective_f,
        constraints=constraints,
        optimizer=optimizer,
        copy=True,
        verbose=1,
        options={"max_time":max_time/num_cont, "trust_regions":1, "box_size":box_size/(4**i), "training_steps": 20, "batch_size":30}
    )

    results_array.append(result)

eq_outputs = []
for result in results_array:
    for output in result.progress:
        eq_output = eq_qs.copy()
        eq_output.params_dict = output[1]
        eq_outputs.append(eq_output)


orig_outputname = "QA_T" + sys.argv[1] + "_B" + sys.argv[2] + "_M" + sys.argv[3] + "_C" + sys.argv[4]
try:
    os.mkdir(orig_outputname)
    outputname = orig_outputname
except(OSError):
    writing = True
    write_attempt = 2
    while writing:
        try:
            os.mkdir(orig_outputname + "_" + str(write_attempt))
            outputname = orig_outputname + "_" + str(write_attempt)
            writing = False
        except(OSError):
            write_attempt += 1

eq_qs.save(outputname+"/eq_result.h5")

for i in range(len(eq_outputs)):
    eq_outputs[i].save(outputname+"/eq_output_" + str(i) + ".h5")

objs = []
subobjs = [type(subobj).__name__() for subobj in objective_f.objectives]
obj_dict = {subobj : [] for subobj in subobjs}

with open(outputname+"/obj_history.txt", "w") as obj_history:
    for result in results_array:
        for x in result.allx:
            fx = objective_f.compute_scalar(x)
            objs.append(fx)
            obj_history.write(f"{fx:.5f}\n")
            for subobjname, subobj in zip(subobjs, objective_f.objectives):
                obj_dict[subobjname].append(abs(subobj.compute_scalar(x)))


'''with open(outputname+"/split_objs.txt", "w") as split_obj:
    for elt in results_array[0].objectiveRef:
        split_obj.write(type(elt).__name__ + " ")
    for result in results_array:
        for item in result.bestObjVector:
            split_obj.write("\n")
            for elt in item:
                split_obj.write(f"{elt:.5f} ")'''


plt.figure("Result Boozer")
#fig = plot_3d(eq, "|B|", grid=grid, figsize=(5,5))
plot_boozer_surface(eq_qs)
plt.savefig(outputname + "/result_boozer.png")

plt.figure("Result Boundary")
plot_boundary(eq_qs)
plt.savefig(outputname + "/result_boundary.png")

fig = plot_3d(eq_qs, "|B|")
fig.write_html(outputname + "/result_3d.html")
pio.write_image(fig, outputname + "/result_3d.png")

plt.figure("Result Objective")
fig = plt.plot(objs)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(outputname + "/result_obj.png")

plt.figure("Result Log Objective")
fig = plt.semilogy(objs)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(outputname + "/result_log_obj.png")


'''path = os.path.join(folder, "split_objs.txt")
maxval = 1000
with open(path) as obj_history:
    terms = obj_history.readline().split()
    obj_dict = {term : [] for term in terms}
    for line in obj_history:
        vals = [min(maxval, float(val)) for val in line.split()]
        for i in range(len(terms)):
            obj_dict[terms[i]].append(vals[i])'''


plt.figure("Split Objective")
for subobj in subobjs:
    line, = plt.plot(obj_dict[subobj])
    line.set_label(subobj)
line, = plt.plot(objs)
line.set_label("Objective")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Split Optimization Progress")
plt.savefig(outputname + "/split_obj.png")

plt.figure("Log Split Objective")
for subobj in subobjs:
    line, = plt.semilogy(obj_dict[subobj])
    line.set_label(subobj)
line, = plt.semilogy(objs)
line.set_label("Objective")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Split Optimization Progress")
plt.savefig(outputname + "/log_split_obj.png")
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import desc.io
from desc.grid import LinearGrid, ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixCurrent,
    FixPsi,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    QuasisymmetryTripleProduct,
    RotationalTransform,
)
from desc.optimize import Optimizer

# Load initial equilibrium
eq_fam = desc.io.load("inputs/nfp2_QA.h5")
eq_init = eq_fam[-1]
eq = eq_init.copy()

# Print available parameters
print("Available parameters:")
for key in eq.params_dict.keys():
    print(f"  {key}")

# Debug: print available attributes and methods of eq
print(f"dir(eq): {dir(eq)}")

# Debug: print eq.x_idx to inspect mapping
print(f"eq.x_idx: {eq.x_idx}")

# Create grid for force balance evaluation
grid = ConcentricGrid(
    L=eq.L_grid,
    M=eq.M_grid,
    N=eq.N_grid,
    NFP=eq.NFP,
    sym=eq.sym
)

# Create objectives
force_objective = ObjectiveFunction((
    ForceBalance(eq=eq, weight=1.0),
))

aspect_objective = ObjectiveFunction((
    AspectRatio(eq=eq, weight=1.0, target=6.0),
))

quasisym_objective = ObjectiveFunction((
    QuasisymmetryTripleProduct(eq=eq),
))

iota_objective = ObjectiveFunction((
    RotationalTransform(eq=eq, weight=1.0, target=0.5),
))

# Build all objectives
force_objective.build()
aspect_objective.build()
quasisym_objective.build()
iota_objective.build()

# Get the flat parameter vector
x0 = np.array(eq.pack_params(eq.params_dict))
n_dof = len(x0)

# First plot: Force Balance
plt.figure(figsize=(12, 8))
for flat_idx in range(n_dof):
    base_value = x0[flat_idx]
    values = np.linspace(base_value - 1, base_value + 1, 5000)
    force_errors = []
    for val in values:
        x_temp = np.array(x0)
        x_temp[flat_idx] = val
        error = force_objective.compute_scalar(x_temp)
        force_errors.append(error)
    plt.semilogy(values - base_value, force_errors, label=f'param {flat_idx}')

plt.xlabel('Parameter Value')
plt.ylabel('Force Balance Error')
plt.title('Force Balance vs Each Degree of Freedom')
plt.grid(True)
# Optionally, do not show all labels if too many curves
# plt.legend(loc='best', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig('force_balance_all_dofs.png')
# plt.show()

print(f"Plot saved as force_balance_all_dofs.png")

# Second plot: Aspect Ratio
plt.figure(figsize=(12, 8))
for flat_idx in range(n_dof):
    base_value = x0[flat_idx]
    values = np.linspace(base_value - 1, base_value + 1, 5000)
    aspect_errors = []
    for val in values:
        x_temp = np.array(x0)
        x_temp[flat_idx] = val
        error = aspect_objective.compute_scalar(x_temp)
        aspect_errors.append(error)
    plt.semilogy(values - base_value, aspect_errors, label=f'param {flat_idx}')

plt.xlabel('Parameter Value')
plt.ylabel('Aspect Ratio Error')
plt.title('Aspect Ratio vs Each Degree of Freedom')
plt.grid(True)
# Optionally, do not show all labels if too many curves
# plt.legend(loc='best', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig('aspect_ratio_all_dofs.png')
# plt.show()

print(f"Plot saved as aspect_ratio_all_dofs.png")

# Third plot: Quasisymmetry
plt.figure(figsize=(12, 8))
for flat_idx in range(n_dof):
    base_value = x0[flat_idx]
    values = np.linspace(base_value - 1, base_value + 1, 5000)
    quasisym_errors = []
    for val in values:
        x_temp = np.array(x0)
        x_temp[flat_idx] = val
        error = quasisym_objective.compute_scalar(x_temp)
        quasisym_errors.append(error)
    plt.semilogy(values - base_value, quasisym_errors, label=f'param {flat_idx}')

plt.xlabel('Parameter Value')
plt.ylabel('Quasisymmetry Error')
plt.title('Quasisymmetry vs Each Degree of Freedom')
plt.grid(True)
plt.tight_layout()
plt.savefig('quasisymmetry_all_dofs.png')
# plt.show()

print(f"Plot saved as quasisymmetry_all_dofs.png")

# Fourth plot: Iota Profile
plt.figure(figsize=(12, 8))
for flat_idx in range(n_dof):
    base_value = x0[flat_idx]
    values = np.linspace(base_value - 1, base_value + 1, 5000)
    iota_errors = []
    for val in values:
        x_temp = np.array(x0)
        x_temp[flat_idx] = val
        error = iota_objective.compute_scalar(x_temp)
        iota_errors.append(error)
    plt.semilogy(values - base_value, iota_errors, label=f'param {flat_idx}')

plt.xlabel('Parameter Value')
plt.ylabel('Iota Profile Error')
plt.title('Iota Profile vs Each Degree of Freedom')
plt.grid(True)
plt.tight_layout()
plt.savefig('iota_profile_all_dofs.png')
plt.show()

print(f"Plot saved as iota_profile_all_dofs.png") 
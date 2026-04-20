import sys
import os
import matplotlib.pyplot as plt
import numpy as np

file = sys.argv[1]

iters = []
objectives = []
maxval = 1000000000
itershift = 0
with open(file) as errorlog:
    for line in errorlog:
        if "Starting from fbest" in line:
            itershift = 0 if iters == [] else iters[-1]
            iters.append(itershift + 1)
            objectives.append(float(line.split()[-1]))
        if "New best:" in line: 
            iters.append(int(line.split()[0][:-1]) + itershift)
            objectives.append(float(line.split()[-1]))


plt.figure("Result Objective")
fig = plt.plot(iters, objectives)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(file.split(".")[0] + "_obj.png")

plt.figure("Result Log Objective")
fig = plt.semilogy(iters, objectives)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(file.split(".")[0] + "_log_obj.png")


objectives = [elt for elt in objectives if elt < 20*objectives[-1]]
iters = iters[len(iters) - len(objectives):]
plt.figure("Tail Long Result Objective")
fig = plt.semilogy(iters, objectives)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(file.split(".")[0] + "tail_log_obj.png")


objectives = [elt for elt in objectives if elt < 2*objectives[-1]]
iters = iters[len(iters) - len(objectives):]
plt.figure("Tail Long Log Result Objective")
fig = plt.semilogy(iters, objectives)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(file.split(".")[0] + "taillong_log_obj.png")

import sys
import os
import matplotlib.pyplot as plt
import numpy as np


folder = sys.argv[1]


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

plt.figure("Result Log Objective")
fig = plt.semilogy(objectives)
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimization Progress")
plt.savefig(folder + "/result_log_obj.png")


path = os.path.join(folder, "split_objs.txt")
maxval = 1000
with open(path) as obj_history:
    terms = obj_history.readline().split()
    obj_dict = {term : [] for term in terms}
    for line in obj_history:
        vals = [min(maxval, float(val)) for val in line.split()]
        for i in range(len(terms)):
            obj_dict[terms[i]].append(vals[i])


plt.figure("Split Objective")
for term in terms:
    line, = plt.plot(obj_dict[term])
    line.set_label(term)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Split Optimization Progress")
plt.savefig(folder + "/split_obj.png")

plt.figure("Log Split Objective")
for term in terms:
    line, = plt.semilogy(obj_dict[term])
    line.set_label(term)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Split Optimization Progress")
plt.savefig(folder + "/log_split_obj.png")


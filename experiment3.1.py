from WsQaoaRunner import execute_circuit
from graphs.graphStorage import GraphStorage, ProblemInstance
from objectiveFunctions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# load problem instance
#Graph = GraphStorage.load("graphs/set-test/fc-6-graph.txt")
#cuts = GraphStorage.loadGWcuts("graphs/set-test/fc-6-cuts.txt")
#cut_index = 2
Graph = GraphStorage.load("graphs/set-1/fc-12-graph.txt")
cuts = GraphStorage.loadGWcuts("graphs/set-1/fc-12-cuts.txt")
cut_index = 17
problem_instance = ProblemInstance(Graph, cuts[cut_index][0], cuts[cut_index][1])
max_cut = cuts[-1][1]

# setup configuration
obj_func = F_EE()
max_cut_obj_func = F_MaxCut_probability(max_cut)

eps=0.075
step_size = np.pi/20

# prepare grid
gammaStart, gammaEnd = 0, 2*np.pi + step_size
betaStart, betaEnd = 0, np.pi +  step_size
a_gamma = np.arange(gammaStart, gammaEnd, step_size)
a_beta = np.arange(betaStart, betaEnd, step_size)

a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
shape = a_gamma.shape
a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

# evaluate grid
objective_values = []
max_cut_probs = []
num_operations = len(a_gamma)
print(f"Total Operations: {num_operations}")
for i in range(len(a_gamma)):
    if (i + 1) % 10 == 0: print(f"{i+1}/{num_operations}")
    counts = execute_circuit(problem_instance, p=1, parameters=[a_gamma[i], a_beta[i]], eps=eps)
    objective_value = obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
    objective_values.append(-objective_value)
    max_cut_prob = max_cut_obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
    max_cut_probs.append(max_cut_prob)

objective_values = np.array(objective_values)

# find minimum
result = np.where(objective_values == np.amin(objective_values))
gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

# draw contours of objective values
a_gamma, a_beta, objective_values = a_gamma.reshape(shape), a_beta.reshape(shape), objective_values.reshape(shape)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
img = ax.contourf(a_gamma, a_beta, objective_values, cmap=cm.get_cmap('viridis', 256), antialiased=True)

# draw minimum objective value
ax.scatter(gamma, beta, s=100, edgecolor="r", facecolor="none", linewidth=3.5)

plt.show()

# repeat for max cut probabilities
max_cut_probs = np.array(max_cut_probs)
a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

# find minimum
result = np.where(max_cut_probs == np.amin(max_cut_probs))
gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

# draw contours of objective values
a_gamma, a_beta, max_cut_probs = a_gamma.reshape(shape), a_beta.reshape(shape), max_cut_probs.reshape(shape)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
img = ax.contourf(a_gamma, a_beta, max_cut_probs, cmap=cm.get_cmap('viridis', 256), antialiased=True)

# draw minimum objective value
ax.scatter(gamma, beta, s=100, edgecolor="r", facecolor="none", linewidth=3.5)

plt.show()

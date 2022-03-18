from WsQaoaRunner import run_ws_qaoa, execute_circuit
from optimization.OptimizationStrategies import (
    optimizer_settings, StandardOptimization, IncrementalFullOptimization, IncrementalPartialOptimization)
from graphs.graphStorage import GraphStorage, ProblemInstance
from objectiveFunctions import *
import matplotlib.pyplot as plt

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
optimizer_set = optimizer_settings(method="COBYLA")

opt_strat = StandardOptimization(optimizer_set, True)
#opt_strat = IncrementalFullOptimization(optimizer_set, True)
#opt_strat = IncrementalPartialOptimization(optimizer_set, True)

obj_func = F_EE()
max_cut_obj_func = F_MaxCut_probability(max_cut)

#eps = 0.15
p = 1
steps = 0.125
replications = 1

# run experiment, compute median values over all replications
median_objective = {}
median_maxcut = {}
for eps in np.arange(0, 0.5001, steps):
    print(f"Epsilon: {eps}")
    objective_values_list, max_cut_prob_list = [], []
    for i in range(replications):
        print(f"Replication {i+1}/{replications}")
        res = run_ws_qaoa(problem_instance, opt_strat, obj_func, eps, p, optimize_epsilon=False)
        res = res[p] # extract results for p=1
        params = res.bestValue[0]
        epsilon = res.epsilon
        counts = execute_circuit(problem_instance, p, params, epsilon)
        objective_value = obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
        objective_values_list.append(objective_value)
        max_cut_prob = max_cut_obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
        max_cut_prob_list.append(max_cut_prob)
    median_maxcut[eps] = np.median(max_cut_prob_list)
    median_objective[eps] = np.median(objective_values_list)

# present results
print("\nmedian objective")
for k,v in median_objective.items():
    print(f"{k}: {v}")
print(median_objective)

print("\nmedian maxcut")
for k,v in median_maxcut.items():
    print(f"{k}: {v}")
print(median_maxcut)

plt.plot(median_objective.keys(), median_objective.values(), label=type(obj_func).__name__)
plt.legend()
plt.ylabel('objective values')
plt.show()

plt.plot(median_maxcut.keys(), median_maxcut.values())
plt.ylabel('maxcut prob')
plt.show()

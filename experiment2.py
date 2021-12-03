from WsQaoaRunner import run_ws_qaoa, execute_circuit
from optimization.OptimizationStrategies import (
    optimizer_settings, StandardOptimization, IncrementalFullOptimization, IncrementalPartialOptimization)
from graphs.graphStorage import GraphStorage, ProblemInstance
from objectiveFunctions import *
import matplotlib.pyplot as plt

# load problem instance
# Graph = GraphStorage.load("graphs/set-test/fc-6-graph.txt")
# cuts = GraphStorage.loadGWcuts("graphs/set-test/fc-6-cuts.txt")
# cut_index = 2
Graph = GraphStorage.load("graphs/set-1/fc-12-graph.txt")
cuts = GraphStorage.loadGWcuts("graphs/set-1/fc-12-cuts.txt")
cut_index = 17
problem_instance = ProblemInstance(Graph, cuts[cut_index][0], cuts[cut_index][1])
max_cut = cuts[-1][1]

# setup configuration
optimizer_set = optimizer_settings(method="COBYLA")

opt_strats = [StandardOptimization(optimizer_set, True),
              IncrementalFullOptimization(optimizer_set, True),
              IncrementalPartialOptimization(optimizer_set, True)]

obj_func = F_Greedy()
max_cut_obj_func = F_MaxCut_probability(max_cut)
better_cut_obj_func = F_BetterCut_probability()

eps = 0.075
p = 3
replications = 3
cold = False # cold-started QAOA

if cold: problem_instance.initial_cut = None

# run experiment, compute median values over all replications
median_objective = {}
median_maxcut = {}
median_bettercut = {}
median_epochs = {}
for opt_strat in opt_strats:
    opt_strat_name = type(opt_strat).__name__
    print(f"Strategy: {opt_strat_name}")
    
    objective_values_list = {}
    max_cut_prob_list = {}
    better_cut_prob_list = {}
    epochs_list = {}
    
    for p_i in range(0, p+1):
        objective_values_list[p_i] = []
        max_cut_prob_list[p_i] = []
        better_cut_prob_list[p_i] = []
        epochs_list[p_i] = []
    
    for i in range(replications):
        print(f"Replication: {i+1}/{replications}")
        res = run_ws_qaoa(problem_instance, opt_strat, obj_func, eps, p, optimize_epsilon=False)
        for p_i in range(0, p+1):
            if p_i > 0:
                res_i = res[p_i] # extract results for each iteration
                params = res_i.bestValue[0]
                epsilon = res_i.epsilon
            else:
                params = []
                epsilon = eps
            
            counts = execute_circuit(problem_instance, p_i, params, epsilon)
            objective_value = obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
            objective_values_list[p_i].append(objective_value)
            max_cut_prob = max_cut_obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
            max_cut_prob_list[p_i].append(max_cut_prob)
            better_cut_prob = better_cut_obj_func.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
            better_cut_prob_list[p_i].append(better_cut_prob)
            epochs_list[p_i].append(len(res_i.optimizationPath) if p_i > 0 else 0)


    median_objective[opt_strat_name] = {
        p_i: np.median(objective_values_list[p_i]) for p_i in range(0, p+1)}
    median_maxcut[opt_strat_name] = {
        p_i: np.median(max_cut_prob_list[p_i]) for p_i in range(0,p+1)}
    median_bettercut[opt_strat_name] = {
        p_i: np.median(better_cut_prob_list[p_i]) for p_i in range(0,p+1)}
    median_epochs[opt_strat_name] = {
        p_i: np.median(epochs_list[p_i]) for p_i in range(0,p+1)}


# present results
print(median_objective)
print(median_maxcut)
print(median_bettercut)
print(median_epochs)

for k in median_objective.keys():
    v = median_objective[k]
    plt.plot(v.keys(), v.values(), label=k)
plt.ylabel("objective values")
plt.legend()
plt.show()

for k in median_maxcut.keys():
    v = median_maxcut[k]
    plt.plot(v.keys(), v.values(), label=k)
plt.ylabel("maxcut prob")
plt.legend()
plt.show()

for k in median_bettercut.keys():
    v = median_bettercut[k]
    plt.plot(v.keys(), v.values(), label=k)
plt.ylabel("bettercut prob")
plt.legend()
plt.show()

for k in median_epochs.keys():
    v = median_epochs[k]
    plt.plot(v.keys(), v.values(), label=k)
plt.ylabel("epochs")
plt.legend()
plt.show()

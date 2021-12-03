from WsQaoaRunner import run_ws_qaoa, execute_circuit
from optimization.OptimizationStrategies import (
    optimizer_settings, StandardOptimization, IncrementalFullOptimization, IncrementalPartialOptimization)
from graphs.graphStorage import GraphStorage, ProblemInstance
from objectiveFunctions import *
import matplotlib.pyplot as plt
import json

# load problem instances
path = "graphs/set-1"
fname = f"{path}/problem_instances.txt"
a_file = open(fname, "r")
problem_instances_raw = json.loads(a_file.read())
problem_instances = []

for pir in problem_instances_raw:
    graph = GraphStorage.load(f"{path}/{pir[0]}")
    cuts = GraphStorage.loadGWcuts(f"{path}/{pir[1]}")
    cut = cuts[pir[2]]
    max_cut = cuts[-1]
    problem_instance = ProblemInstance(graph, cut[0], cut[1], max_cut[1], pir[0])
    problem_instances.append(problem_instance)

# split problem instances in 12 and 24 nodes
problem_instances_12nodes = [
    problem_instance for problem_instance in problem_instances if problem_instance.graph.shape[0]==12]
problem_instances_24nodes = [
    problem_instance for problem_instance in problem_instances if problem_instance.graph.shape[0]==24]
#print(problem_instances_12nodes)
#print(problem_instances_24nodes)

# initialize objective functions
objective_functions_12nodes = [F_EE(), F_CVaR(0.05), F_Gibbs(5), F_Greedy(), F_EEminusI()]
objective_functions_24nodes = [F_EE(), F_CVaR(0.05), F_Gibbs(2), F_Greedy(), F_EEminusI()]

# initialize optimization strategy
optimizer_set = optimizer_settings(method="COBYLA")
opt_strat = IncrementalPartialOptimization(optimizer_set, False) # this time don't care about intermediate results


# run optimization and obtain max cut probability
problem_instances = problem_instances_12nodes
objective_functions = objective_functions_12nodes
p = 3
eps = 0.15
optimize_epsilon = True
replications = 1

median_final_probs = []
for obj_func in objective_functions:
    obj_func_name = type(obj_func).__name__
    print(obj_func_name)
    for prob_inst in problem_instances:
        print(prob_inst.label)
        max_cut_probs, better_cut_probs, epsilons = [], [], []
        for rep in range(replications):
            print(f"Replication {rep+1}/{replications}")
            optimized = run_ws_qaoa(prob_inst, opt_strat, obj_func, eps, p, optimize_epsilon=optimize_epsilon)
            # get final optimized parameters and evaluate
            params = optimized[p].bestValue[0]
            epsilon = optimized[p].epsilon
            counts = execute_circuit(prob_inst, p, params, epsilon)
            max_cut_prob = F_MaxCut_probability(
                prob_inst.max_cut_size).evaluate(
                    counts, prob_inst.graph, prob_inst.initial_size)
            better_cut_prob = F_BetterCut_probability().evaluate(
                    counts, prob_inst.graph, prob_inst.initial_size)
            max_cut_probs.append(max_cut_prob)
            better_cut_probs.append(better_cut_prob)
            epsilons.append(epsilon)
        # compute medians for all replications
        median_final_probs.append({
            "name": prob_inst.label, 
            "objective_fun": obj_func_name,
            "final_maxcutprob": np.median(max_cut_probs),
            "final_bettercutprob": np.median(better_cut_probs),
            "final_epsilon": np.median(epsilons),
            "epsilon_optimized": optimize_epsilon})

print(median_final_probs)

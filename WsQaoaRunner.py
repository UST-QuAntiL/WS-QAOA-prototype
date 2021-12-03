from helperFunctions import *
from circuit.QAOACircuitGenerator import QAOACircuitGenerator
from qiskit import Aer, execute
import numpy as np

def run_ws_qaoa(problem_instance, optimization_strategy, objective_function, epsilon, p, optimize_epsilon=False):
    
    def to_be_optimized(parameters, p_i, previous_parameters=[], epsilon=0.15, optimize_epsilon=False):
        # parameters will be optimized
        # previous_parameters will just be prepended --> allows partial optimization
        parameters = np.append(previous_parameters, parameters)
        if optimize_epsilon:
            eps = parameters[-1]
            parameters = parameters[:-1]
        else:
            eps = epsilon
        counts = execute_circuit(problem_instance, p_i, parameters, eps)
        objective_value = objective_function.evaluate(counts, problem_instance.graph, problem_instance.initial_size)
        return -objective_value
    
    results = optimization_strategy.optimize(p, to_be_optimized, epsilon, optimize_epsilon)
    
    return results

def execute_circuit(problem_instance, p, parameters, eps):
    # setup simulator
    backend = Aer.get_backend("qasm_simulator")
    shots = 5000
    
    # apply epsilon to the initial cut
    if problem_instance.initial_cut is not None:
        cut_list = cut2list(problem_instance.initial_cut)
        regularized_cut = epsilon_function(cut_list, eps)
    else:
        regularized_cut = None

    # generate circuit
    circuit = QAOACircuitGenerator.genQaoaMaxcutCircuit(problem_instance.graph, parameters, regularized_cut, p)

    counts = execute(circuit, backend, shots=shots).result().get_counts()
    return counts

from abc import ABC, abstractmethod
from helperFunctions import *
import numpy as np

class objectiveFunction(ABC):
    @abstractmethod
    def evaluate(self, counts, G, initial_cut_size=None):
        pass

    cached_graph = None
    cached_cut_size = {}

    # Compute size of cut x on Graph G. x is a bitstring.
    def cut_size(self, x, G):
        n_vertices = G.shape[0]
        cut_string = ''.join(str(x))
        if cut_string in objectiveFunction.cached_cut_size.keys() and hash(str(G)) == objectiveFunction.cached_graph:
            return objectiveFunction.cached_cut_size.get(cut_string)
        else:
            objectiveFunction.cached_graph = hash(str(G))
        C = 0
        for i in range(n_vertices):
            for j in range(i):
                C += G[i,j] * (not x[i] == x[j])
        objectiveFunction.cached_cut_size[cut_string] = C
        return C

    def computeCosts(self, counts, G):
        allCosts = np.array([self.cut_size(parseSolution(x), G) for x in list(counts.keys())])
        z = zip(list(counts.keys()), list(counts.values()), list(allCosts))
        z = list(z)
        return z

class F_EE(objectiveFunction):
    def __init__(self):
        pass

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value = (np.sum(np.array([z[i][2] * z[i][1] for i in range(len(z))])) / n_samples)
        else:
            return 0
        return total_objective_value

class F_CVaR(objectiveFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        z.sort(key=takeThird, reverse=True)
        total_objective_value = 0
        alphaRemaining = np.ceil(self.alpha * np.sum(list(counts.values())))
        n_considered = alphaRemaining
        for i in range(len(z)):
            if z[i][1] < alphaRemaining:
                total_objective_value += z[i][1] * z[i][2]
                alphaRemaining -= z[i][1]
            else:
                total_objective_value += alphaRemaining * z[i][2]
                break
        if n_considered > 0:
            total_objective_value /= n_considered
        else:
            return 0
        return total_objective_value
    
class F_Gibbs(objectiveFunction):
    def __init__(self, eta):
        self.eta = eta

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        n_samples = np.sum(list(counts.values()))
        z = np.array(z, dtype=object)
        if n_samples > 0:
            total_objective_value = np.log(np.sum((np.e ** (self.eta * z[:,2]))* z[:,1])/n_samples)
        else:
            return 0
        return total_objective_value
    
class F_Greedy(objectiveFunction):
    def __init__(self):
        pass

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value =  np.sum(np.array([z[i][2] * z[i][1] if z[i][2] > initial_cut_size else 0 for i in range(len(z))])) / n_samples
        else:
            return 0
        return total_objective_value
    
class F_EEminusI(objectiveFunction):
    def __init__(self):
        pass

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        total_objective_value =  np.sum(np.array([z[i][2] * z[i][1] if z[i][2] != initial_cut_size else 0 for i in range(len(z))]))
        n_samples = np.sum(np.array([z[i][1] if z[i][2] != initial_cut_size else 0 for i in range(len(z))]))
        if n_samples > 0:
            total_objective_value = total_objective_value/n_samples
        else:
            return 0
        return total_objective_value

class F_MaxCut_probability(objectiveFunction):
    def __init__(self, max_cut):
        self.max_cut = max_cut

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value =  np.sum(np.array([z[i][1] if z[i][2] == self.max_cut else 0 for i in range(len(z))])) / n_samples
        else:
            return 0
        return total_objective_value

class F_BetterCut_probability(objectiveFunction):
    def __init__(self):
        pass

    def evaluate(self, counts, G, initial_cut_size):
        z = self.computeCosts(counts, G)
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value =  np.sum(np.array([z[i][1] if z[i][2] > initial_cut_size else 0 for i in range(len(z))])) / n_samples
        else:
            return 0
        return total_objective_value

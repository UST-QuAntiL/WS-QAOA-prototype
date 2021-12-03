from scipy.optimize import minimize
import numpy as np
from copy import copy

class MinimizeWrapper():
    def __init__(self):
        self.optimizationPath = []
        self.objectiveFunction = None


    def minimize(self, fun, x0, args=(), method=None, jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None,
                 callback=None, options=None):
        self.optimizationPath = []
        self.objectiveFunction = fun
        import time
        t1 = time.time()

        minimizationResult = minimize(self.wrapObjectiveFunction, x0, args=args, method=method, jac=jac, hess=hess,
                 hessp=hessp, bounds=bounds, constraints=constraints, tol=tol,
                 callback=callback, options=options)

        t2 = time.time()
        minimizationResult.optimizationPath = self.optimizationPath
        index = np.argmin(np.array(self.optimizationPath, dtype=object)[:,1])
        minimizationResult.bestValue = copy(self.optimizationPath[index])
        minimizationResult.bestIsIntermediate = True if index != len(self.optimizationPath)-1 else False
        minimizationResult.optimizationTime = t2-t1
        return minimizationResult


    def wrapObjectiveFunction(self, x0, *args):
        result = self.objectiveFunction(x0, *args)
        self.optimizationPath.append([list(x0), result])
        return result
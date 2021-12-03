from abc import ABC, abstractmethod
from enum import Enum
from .MinimizeWrapper import MinimizeWrapper
import numpy as np

class optimizationStrategy(ABC):
    def __init__(self, optimizer_settings, save_intermediate=False):
        # set True to record intermediate optimized parameters
        self.save_intermediate = save_intermediate
        self.optimizer_settings = optimizer_settings

        # contraints for optimization of epsilon
        self.eps_cons = []
        self.eps_cons.append({'type': 'ineq', 'fun': lambda x: x[-1] - 0})
        self.eps_cons.append({'type': 'ineq', 'fun': lambda x: 0.5 - x[-1]})
    
    @abstractmethod
    def optimize(self, p, function, epsilon, optimize_epsilon=False):
        pass
    
    def appendEpsilon(self, opt, epsilon, optimize_epsilon):
        if optimize_epsilon:
            # extract epsilon from optimized parameters
            opt.epsilon = opt.bestValue[0][-1]
            
            # remove from optimized parameters everywhere else
            opt.bestValue[0] = opt.bestValue[0][:-1]
            opt.x = opt.x[:-1]
            for i in range(len(opt.optimizationPath)):
                opt.optimizationPath[i][0] = opt.optimizationPath[i][0][:-1]
        else:
            # just add epsilon that was used for the run
            opt.epsilon = epsilon
        
        return opt

class StandardOptimization(optimizationStrategy):
    def optimize(self, p, function, epsilon, optimize_epsilon=False):
        optimized = {}
        
        if self.save_intermediate:
            for p_i in range(1,p+1):
                # initialize parameters
                parameters = np.random.default_rng().uniform(0, np.pi, size=2*p_i).tolist()
                if optimize_epsilon and p_i == 1:
                    parameters.append(epsilon)
                
                opt = MinimizeWrapper().minimize(function, parameters, 
                                                 (p_i, [], epsilon, optimize_epsilon), #args
                                                 *self.optimizer_settings.as_tuple(add_cons=self.eps_cons if optimize_epsilon else None))
                opt = self.appendEpsilon(opt, epsilon, optimize_epsilon)
                epsilon = opt.epsilon
                optimize_epsilon = False # optimize only for p=1
                optimized[p_i] = opt
        else:
            parameters = np.random.default_rng().uniform(0, np.pi, size=2*p).tolist()
            if optimize_epsilon:
                parameters.append(epsilon)
            opt = MinimizeWrapper().minimize(function, parameters, 
                                             (p, [], epsilon, optimize_epsilon), # args
                                             *self.optimizer_settings.as_tuple(add_cons=self.eps_cons if optimize_epsilon else None))
            opt = self.appendEpsilon(opt, epsilon, optimize_epsilon)
            epsilon = opt.epsilon
            optimized[p] = opt
        
        return optimized

class IncrementalFullOptimization(optimizationStrategy):
    def optimize(self, p, function, epsilon, optimize_epsilon=False):
        optimized = {}
        
        # execute p=1
        parameters = np.random.default_rng().uniform(0, np.pi, size=2*1).tolist()
        if optimize_epsilon: 
            parameters.append(epsilon)
        opt = MinimizeWrapper().minimize(function, parameters, 
                                         (1, [], epsilon, optimize_epsilon), #args
                                         *self.optimizer_settings.as_tuple(add_cons=self.eps_cons if optimize_epsilon else None))
        opt = self.appendEpsilon(opt, epsilon, optimize_epsilon)
        epsilon = opt.epsilon
        optimize_epsilon = False # epsilon only optimized for p=1

        if self.save_intermediate:
            optimized[1] = opt

        for p_i in range(2,p+1):
            # initialize parameters for next repetition
            parameters = opt.bestValue[0]+[0,0]
            opt = MinimizeWrapper().minimize(function, parameters, 
                                             (p_i, [], epsilon, optimize_epsilon),
                                             *self.optimizer_settings.as_tuple(add_cons=self.eps_cons if optimize_epsilon else None))
            opt = self.appendEpsilon(opt, epsilon, optimize_epsilon)
            epsilon = opt.epsilon
            if self.save_intermediate:
                optimized[p_i] = opt
        
        return optimized if self.save_intermediate else {p: opt}

class IncrementalPartialOptimization(optimizationStrategy):
    def optimize(self, p, function, epsilon, optimize_epsilon=False):
        optimized = {}
        
        # execute p=1
        parameters = np.random.default_rng().uniform(0, np.pi, size=2*1).tolist()
        if optimize_epsilon: 
            parameters.append(epsilon)
        opt = MinimizeWrapper().minimize(function, parameters, 
                                         (1, [], epsilon, optimize_epsilon), #args
                                         *self.optimizer_settings.as_tuple(add_cons=self.eps_cons if optimize_epsilon else None))
        opt = self.appendEpsilon(opt, epsilon, optimize_epsilon)
        epsilon = opt.epsilon
        optimize_epsilon = False # epsilon only optimized for p=1

        if self.save_intermediate:
            optimized[1] = opt

        for p_i in range(2,p+1):
            # initialize parameters for next repetition, optimize only these
            previous_parameters = opt.bestValue[0]
            parameters = [0,0]
            if optimize_epsilon: 
                parameters.append(epsilon)
            opt = MinimizeWrapper().minimize(function, parameters, 
                                             (p_i, previous_parameters, epsilon, optimize_epsilon), #args
                                             *self.optimizer_settings.as_tuple(add_cons=self.eps_cons if optimize_epsilon else None))
            opt = self.appendEpsilon(opt, epsilon, optimize_epsilon) 
            opt = self.merge(previous_parameters, opt)
            epsilon = opt.epsilon
            if self.save_intermediate:
                optimized[p_i] = opt
        
        return optimized if self.save_intermediate else {p: opt}
    
    # merge previous parameters with optimized parameters in the optimization result
    # to have a uniform format of optimization results
    def merge(self, previous_parameters, opt):
        previous_parameters = list(previous_parameters)
        opt.bestValue[0] = previous_parameters + opt.bestValue[0]
        opt.x = np.append(previous_parameters, opt.x)
        for i in range(len(opt.optimizationPath)):
            opt.optimizationPath[i][0] = previous_parameters + opt.optimizationPath[i][0]
        return opt

class optimizer_settings():
    def __init__(self, method=None, jac=None, hess=None, hessp=None, bounds=None, 
                 constraints=[], tol=None, callback=None, options=None):
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
    
    def as_tuple(self, add_cons=None):
        if add_cons is not None: 
            return (self.method, self.jac, self.hess, self.hessp, self.bounds, 
                self.constraints+add_cons, self.tol, self.callback, self.options)
        else:
            return (self.method, self.jac, self.hess, self.hessp, self.bounds, 
                self.constraints, self.tol, self.callback, self.options)
                
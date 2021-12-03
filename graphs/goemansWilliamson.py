import networkx as nx
import cvxgraphalgs as cvxgr
from cvxgraphalgs.algorithms.max_cut import _solve_cut_vector_program
from copy import deepcopy
import numpy as np

def bestGWcuts(graph, n_GW_cuts, n_best, cost_fun=None, allow_duplicates=False):
    # returns n_best best cuts out of n_GW_cuts to be computed
    if n_best > n_GW_cuts:
        raise Exception("n_best has to be less or equal to n_GW_cuts")

    GW_cuts = []
    for i in range(n_GW_cuts):
        approximation = cvxgr.algorithms.goemans_williamson_weighted(nx.Graph(graph))
        # compute binary representation of cut for discrete solution
        approximation_list = []
        for n in range(len(approximation.vertices)):
            if (n in approximation.left):
                approximation_list.append(0)
            else:
                approximation_list.append(1)

        if allow_duplicates or not GW_cuts or not (approximation_list in list(np.array(GW_cuts, dtype=object)[:,0])):
            GW_cuts.append([approximation_list, cost_fun(approximation_list, graph) if cost_fun else 0])

    # return n_best best cuts
    GW_cuts = np.array(GW_cuts, dtype=object)
    GW_cuts = GW_cuts[GW_cuts[:, 1].argsort()]
    GW_cuts = GW_cuts[-n_best:]
    return GW_cuts

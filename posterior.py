""" 
posterior.py

Algorithms for posterior analysis of results
"""
import varbayes as vb
import pandas as pd
import numpy as np
import trees
from projgamma import pt_logd_projgamma_my_mt_inplace_unstable
from collections import defaultdict
from scipy.special import softmax

EPS = 1e-16

def similarity_matrix(dmat):
    """
    Similarity Matrix
    ---
    Posterior Similarity Matrix given stochastic cluster assignments in Bayesian 
    Non-parametric mixture model.
    ---
    inputs:
        dmat (s x n)
    outputs:
        smat (n x n)    
    """
    dmatI = dmat.T
    smat = np.zeros((dmat.shape[0], dmat.shape[0]))
    for s in range(dmat.shape[1]):
        smat[:] += (dmatI[s][None] == dmatI[s][:,None])
    return (smat / dmat.shape[1])

def minimum_spanning_trees(smat):
    graph = trees.Graph(smat.shape[0])
    edges = []
    for i in range(smat.shape[0]):
        for j in range(i + 1, smat.shape[0]):
            # edge weight is reciprocal of posterior co-clustering prob.
            edges.append((i,j, 1 / (smat[i,j] + EPS)))
    for edge in edges:
        graph.addEdge(*edge)
    return graph.KruskalMST()

def emergent_clusters_nl(graph, k):
    components = []
    seen = set()

    def neighbor_list(graph: pd.DataFrame):
        neighbors = defaultdict(set)
        # an edge is a 3-element tuple
        for node1, node2 in graph.iloc[:-k][['node1','node2']].values.tolist(): 
            neighbors[node1].add(node2)
            neighbors[node2].add(node1)
        return neighbors # dictionary of sets listing neighbors for each element
    
    neighbors = neighbor_list(graph)

    def find_component(start):
        stack = [start]
        component = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            component.append(cur)
            for neigh in neighbors[cur]:
                stack.append(neigh)
        return component    

    for v in neighbors:
        if v in seen:
            continue
        components.append(find_component(v))
    
    return components

def emergent_clusters(graph, k):
    """
    Emergent Clusters
    ---
    Given similarity matrix between observations, creates labels for (k-1) 
    emergent clusters.
    ---
    inputs:
        smat (n x n array)
        k    (integer)
    outputs:
        labels (n)
    """
    graph_ = graph.iloc[:-(k-1)] # delete k-1 largest edges; tree in k clusters
    N = graph[['node1','node2']].max().max() + 1 # total # of nodes 

    sets = [set() for _ in range(k)] # output sets

    todo = graph_[['node1','node2']].values.tolist() # list of edges
    done = []
    for i in range(k):
        current = todo.pop() # start the set with the first available edge.
        sets[i].add(current[0])
        sets[i].add(current[1])
        addlfound = False  # if any additional nodes were found during this sweep
        while len(todo) > 0:
            # keep running until every node in set has been found.
            while len(todo) > 0:
                current = todo.pop() # take available edge
                if current[0] in sets[i]:  # if either node in set, add other node
                    sets[i].add(current[1])
                    addlfound = True
                elif current[1] in sets[i]:
                    sets[i].add(current[0])
                    addlfound = True
                else:                # otherwise, append edge to outstack
                    done.append(current)
            if addlfound:  # if any additional nodes were found this sweep
                todo = done  # cycle out-stack to in-stack
                done = []   
                addlfound = False
            raise
        
        todo = done # cycle out-stack to in-stack, proceed to next set.
        done = []
       
    labels = np.zeros(N, dtype = int)
    for i, s in enumerate(sets):
        labels[s] = i
    return labels

def emergent_clusters_pre(model : vb.VarPYPG):
    """
    Assumes that label-switching has been corrected.
    Computes posterior cluster assignment
        1:  alpha_{jl}^* = mean(alpha_{jl}^{(s)} for s = 1,...,S)
        2:  nu_{j} = mean(nu_{js} for s = 1,...,S)
        3:  delta_i^* = argmax(cluster prob given alpha^*, nu^*)
    """
    albar = np.zeros((model.J, model.D))
    nubar = np.zeros((model.J - 1))
    for _ in range(100):
        samples = model.surrogate.sample(100)
        alpha   = samples['alpha'].numpy().mean(axis = 0)
        nu      = samples['nu'].numpy().mean(axis = 0)
        albar += alpha
        nubar += nu
    albar /= 100
    nubar /= 100
    # pi = vb.stickbreak(nu)
    ll = np.zeros((model.N, 1, model.J))
    pt_logd_projgamma_my_mt_inplace_unstable(
        ll, model.Yp, albar[None], np.ones(albar[None].shape),
        )
    logpi = np.zeros((1, model.J))
    logpi[:,:-1] += np.log(nubar)
    logpi[:,1:]  += np.cumsum(np.log(1 - nubar))
    lp = ll.sum(axis = 1) + logpi # log-posterior of delta (unnormalized)
    po = softmax(lp, axis = -1)   # posterior of delta (normalized)
    return np.argmax(po, axis = 1)

def emergent_clusters_post(model : vb.VarPYPG):
    """
    Assumes that label-switching has been corrected.
    Computes posterior cluster assignment
        1: delta_{is} ~ P(cluster_prob | alpha_s, nu_s)
        2: delta_i^* = \argmax delta_i / S
    """
    deltas = model.generate_conditional_posterior_deltas()
    assert deltas.shape[1] == 1000
    d_arr = np.zeros((*deltas.shape, model.J), dtype = int)
    for i in range(model.J):
        d_arr[:,:,i] = (deltas == i) * 1
    return(np.argmax(d_arr.mean(axis = 1), axis = 1))
   
if __name__ == '__main__':
    import pandas as pd
    # knock off 5 edges    
    graph = pd.read_csv('./datasets/slosh/sloshltd_mst.csv')
    # labels = emergent_clusters(graph, 7)
    labels = emergent_clusters_nl(graph, 100)
    raise
    
    
# EOF

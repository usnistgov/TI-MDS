# Topological Initialization for Multidimensional Scaling
# Melinda Kleczynski, Anthony J. Kearsley
# Finalized April 17, 2025 

# Function files 

# numpy 2.2.4
# oat_python 0.1.1
# scipy 1.15.2

import numpy as np
import oat_python as oat
from scipy.sparse import csr_matrix 

def get_mpc_edges(input_pdist): 

    n_data_pts = np.shape(input_pdist)[0] 

    # format pairwise distance matrix 
    formatted_dist_mat = csr_matrix(input_pdist)
    for i in range(n_data_pts):
        for j in range(n_data_pts):
            if input_pdist[i, j] == 0:
                formatted_dist_mat[i, j] = 0  # need explicit zeros

    # perform TDA  
    boundary = oat.rust.FactoredBoundaryMatrixVr(dissimilarity_matrix = formatted_dist_mat, homology_dimension_max = 1)   
    ph = boundary.homology(return_cycle_representatives = True, return_bounding_chains = False)   
    ph1 = ph[ph.dimension == 1].reset_index() 
    ph1.insert(4, 'persistence', ph1.death-ph1.birth) 

    # most persistent cycle 
    mpc_index = np.argmax(ph1.persistence) 
    mpc_edges = np.array([edge for edge in ph1['cycle representative'][mpc_index].simplex]) 

    return mpc_edges, ph1 

def order_cycle_vertices(edges_in_cycle):  

    n_edges = edges_in_cycle.shape[0]  

    ordered_edges = [[edges_in_cycle[0, 0], edges_in_cycle[0, 1]]]  # start with arbitrary edge  
    unused_edges = [[edges_in_cycle[k, 0], edges_in_cycle[k, 1]] for k in range(1, n_edges)] 

    for k in range(len(unused_edges)):

        connecting_vertex = ordered_edges[-1][1]

        next_edge_index = np.argmax([connecting_vertex in edge for edge in unused_edges]) 
        next_edge = unused_edges[next_edge_index]  
        del unused_edges[next_edge_index]

        if next_edge[1] == connecting_vertex:  # reorder so connecting vertices line up  
            next_edge = [next_edge[1], next_edge[0]]

        ordered_edges = ordered_edges + [next_edge]

    return np.array([edge[0] for edge in ordered_edges])

def get_top(input_pdist):

    n_data_pts = np.shape(input_pdist)[0]

    mpc_edges, ph1 = get_mpc_edges(input_pdist) 
    n_edges = mpc_edges.shape[0]

    ordered_cycle_vertices = order_cycle_vertices(mpc_edges)

    cycle_diameter = np.max(input_pdist[np.ix_(ordered_cycle_vertices, ordered_cycle_vertices)])  
    angles = [2*np.pi*i/n_edges for i in range(n_edges)]  

    crep_initializations = np.array([[0.5*cycle_diameter*np.cos(a), 0.5*cycle_diameter*np.sin(a)] for a in angles])

    sub_AI_pdist = input_pdist[:, ordered_cycle_vertices]

    top_initialization = np.zeros((n_data_pts, 2)) 
    for i in range(n_data_pts): 
        nearest_cyclept_index = np.argmin(sub_AI_pdist[i, :]) 
        top_initialization[i, :] = crep_initializations[nearest_cyclept_index, :]

    bc1 = ph1[['birth', 'death']]

    return top_initialization, bc1, cycle_diameter 

def get_bc1(input_pdist): 

    n_data_pts = np.shape(input_pdist)[0] 

    # format pairwise distance matrix 
    formatted_dist_mat = csr_matrix(input_pdist)
    for i in range(n_data_pts):
        for j in range(n_data_pts):
            if input_pdist[i, j] == 0:
                formatted_dist_mat[i, j] = 0  # need explicit zeros

    # perform TDA  
    boundary = oat.rust.FactoredBoundaryMatrixVr(dissimilarity_matrix = formatted_dist_mat, homology_dimension_max = 1)   
    ph = boundary.homology(return_cycle_representatives = True, return_bounding_chains = False)   
    ph1 = ph[ph.dimension == 1].reset_index() 

    return ph1[['birth', 'death']]
"""
"""
import numpy as np
import scipy.stats as st
import networkx as nx
import sklearn.metrics.pairwise as skpw
import PyBDGK.IntermRepresentation.IntermRepresentation as ir
from PyBDGK.GraphEncoding.GE_Base import GE_Base
import os
import pdb

class GE_FunctionalConnectivity(GE_Base):
    """
    Graph encoding where each voxel represents a node and there is an edge
    between two nodes if the voxels timeseries are more similar than a given
    threshold.
    """

    def encode(self, interm_rep, similarity_measure="linear", threshold=0.5, n_jobs=-1, **kwds):
        """
        Parameters
        ----------
        interm_rep : Data in the intermediate representation

        similarity_measure : Function to be used to compute the similarity.
        Based on the kernel measures on scikit-learn, the possible values are:
        ===============   ========================================
        metric            Function
        ===============   ========================================
        'additive_chi2'   sklearn.pairwise.additive_chi2_kernel
        'chi2'            sklearn.pairwise.chi2_kernel
        'linear'          sklearn.pairwise.linear_kernel
        'poly'            sklearn.pairwise.polynomial_kernel
        'polynomial'      sklearn.pairwise.polynomial_kernel
        'rbf'             sklearn.pairwise.rbf_kernel
        'sigmoid'         sklearn.pairwise.sigmoid_kernel
        'cosine'          sklearn.pairwise.cosine_similarity
        ===============   ========================================

        threshold : Threshold value to be used to determine whether to include
        an edge between to nodes.

        **kwds: optional parameters for the similarity measure.

        Returns
        -------
        A Networkx graph with the encoding.

        """
        #Computing similarity matrix
        sim_mat = skpw.pairwise_kernels(interm_rep.arr_voxels, interm_rep.arr_voxels,
                              metric = similarity_measure, filter_params = False,
                              n_jobs = n_jobs, **kwds)

        #Normalizing similarity in [0,1] and applying threshold.
        adj_mat = np.zeros((sim_mat.shape[0], sim_mat.shape[0]), dtype = np.byte)
        for j in range(sim_mat.shape[0] - 1):
            for k in range(j + 1, sim_mat.shape[1]):
                aux = sim_mat[j, k] / np.sqrt(sim_mat[j, j] * sim_mat[k, k])
                if aux >= threshold:
                    adj_mat[j,k] = 1
                    adj_mat[k,j] = 1

        #Building the graph from the adjacency matrix
        g = nx.from_numpy_matrix(adj_mat)

        return g



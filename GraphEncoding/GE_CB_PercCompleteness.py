"""
"""
import numpy as np
import scipy.stats as st
import networkx as nx
import sklearn.metrics.pairwise as skpw
from sklearn.cluster import MiniBatchKMeans, KMeans
import PyBDGK.IntermRepresentation.IntermRepresentation as ir
from PyBDGK.GraphEncoding.GE_Base import GE_Base
import os
import pdb

class GE_CB_PercCompleteness(GE_Base):
    """  
    Graph encoding based on clustering algorithms. Given a set of voxels, a 
    clustering algorithm is applied and then a node is assigned to each 
    cluster. Each node takes the average voxel time serie of the voxels 
    belonging to the node. All similarity values between every pair
    of node time series are computed. An edge is built between two nodes
    if the similarity value between them is one of the x highest values. x is
    a parameter of the algorithm.
    """      

    def encode(self, interm_rep, clust_alg='MiniBatchKMeans', 
               n_clusters=-1,
               clust_ratio=10,  
               similarity_measure="pearson", 
               completeness = 0.5, n_jobs=1, **kwds):
        """
        interm_rep : Data in the intermediate representation
        
        clust_alg : KMeans or MiniBatchKMeans.
        
        n_clusters : Number of clusters to be computed. If n_clusters = -1,
                     then n/clust_ratio is used.
                                 
        clust_ratio :  The number of clusters will be computed as n/clust_ratio
                       if n_cluster not equal to -1, this parameter is ignored.
        
        similarity_measure : pearson or one of the similarity available in 
        scikit-learn, i.e.:
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
        
        completeness : Percentage of completeness we want in the resulting 
                       graph. For example if the value is 0.2, then we will
                       keep the edges with 20% higher similarity values.
        
        **kwds: optional parameters for the similarity measure.
        
        Returns
        -------
        A Networkx graph with the encoding.
        """
        
        if n_clusters == -1:
            n_clusters = len(interm_rep.arr_xyz) / clust_ratio
        
        ca = None
        if clust_alg == 'MiniBatchKMeans':
            ca = MiniBatchKMeans(init='k-means++', n_clusters = n_clusters, 
                                 batch_size=n_clusters/2, n_init=10, 
                                 max_no_improvement=10, verbose=0) 
        elif clust_alg == 'KMeans':
            ca = KMeans(init='k-means++', n_clusters = n_clusters, 
                        n_init = 10, n_jobs = n_jobs)
        else:
            raise Exception("Invalid algorithm name, only KMeans and MiniBatchKMeans are supported.")
        
        #Applying the clustering algorithm        
        ca.fit(interm_rep.arr_xyz)
        labels = ca.labels_  
        
        #Computing the unique cluster indentifiers
        l_unique = np.unique(labels)
        
        mean_voxels = np.zeros((len(l_unique), interm_rep.arr_voxels.shape[1]))
        mean_xyz = np.zeros((len(l_unique), interm_rep.arr_xyz.shape[1]))      
        
        cont = 0
        for i in l_unique:
            #Taking the possitions corresponding to the same cluster.
            pos = np.where(labels == i)[0]
            #Taking data from these possitions and computing the mean time serie
            m_voxel = interm_rep.arr_voxels[pos].mean(0)
            #Taking the xyz from these positions and computing the mean value
            m_xyz = interm_rep.arr_xyz[pos].mean(0)
            
            mean_voxels[cont] = m_voxel
            mean_xyz[cont] = m_xyz
            
            cont += 1
            
        #The new intermediate representation is given by mean_voxels and 
        # mean_xyz.
        #pdb.set_trace()
        
        #Computing similarity matrix 
        adj_mat = np.zeros((len(mean_voxels), len(mean_voxels)))
        for j in range(len(mean_voxels) - 1):
            for k in range(j + 1, len(mean_voxels)):
                if similarity_measure == 'pearson':
                    adj_mat[j,k] = adj_mat[k,j] = st.pearsonr(mean_voxels[j], mean_voxels[k])[0]
                else:
                    adj_mat[j,k] = adj_mat[k,j] = skpw.pairwise_kernel(mean_voxels[j], 
                                                                       mean_voxels[k], 
                                                                       metric = similarity_measure,
                                                                       n_jobs = n_jobs)
                                          
        n_edges = int(completeness * ((len(adj_mat) * (len(adj_mat) -1)) / 2))
        r = np.triu_indices_from(adj_mat, 1)
        l = np.array([adj_mat[r[0][i],r[1][i]] for i in range((len(adj_mat) * (len(adj_mat) -1)) / 2)])
        l.sort()
        threshold = l[-1 * n_edges]
        new_adj_mat = np.where(adj_mat > threshold, 1, 0)
                                                        
        
        #Building the graph from the adjacency matrix
        g = nx.from_numpy_matrix(new_adj_mat)
        
        return g
        


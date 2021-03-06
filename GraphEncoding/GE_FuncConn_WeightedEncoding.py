"""
"""
import numpy as np
import scipy.stats as st
import networkx as nx
import sklearn.metrics.pairwise as skpw
from sklearn.cluster import MiniBatchKMeans, KMeans
import PyBDGK.IntermRepresentation.IntermRepresentation as ir
from PyBDGK.GraphEncoding.GE_Base import GE_Base
import PyBDGK.Visualization.Plot as pp
import os
import pdb

class GE_FuncConn_WeightedEncoding(GE_Base):
    """
    Graph encoding based on clustering algorithms. Given a set of voxels, a
    clustering algorithm is applied and then a node is assigned to each
    cluster. Each node takes the average voxel time serie of the voxels
    belonging to the node. Edges are computed by applying a similarity
    measure between time series and keeping the values higher than a fixed
    threshold. The edges that survive the thresholding will keep the weight
    value in an attribute called 'weight'.
    """

    def encode(self, interm_rep, clust_alg='MiniBatchKMeans',
               n_clusters=-1,
               clust_ratio=100,
               similarity_measure="pearson",
               threshold=0.0, n_jobs=1, **kwds):
        """
        Parameters
        ----------
        interm_rep: IntermRep
            Data in the intermediate representation

        clust_alg: string
            KMeans or MiniBatchKMeans.

        n_clusters: int
            Number of clusters to be computed. If n_clusters = -1,
            then n/clust_ratio is used.

        clust_ratio: int
            The number of clusters will be computed as n/clust_ratio.
            If n_cluster not equal to -1, this parameter is ignored.

        similarity_measure: string
            pearson or one of the similarity available in scikit-learn, i.e.:
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

        threshold: float
            Threshold value to be used to determine whether to include
            an edge between to nodes.

        **kwds: optional parameters for the similarity measure.

        Returns
        -------
        g: Graph
            A Networkx graph with the encoding.
        """

        if n_clusters == -1:
            n_clusters = len(interm_rep.arr_xyz) / clust_ratio

        ca = None
        if clust_alg == 'MiniBatchKMeans':
            ca = MiniBatchKMeans(init='k-means++', n_clusters = n_clusters,
                                 batch_size=10*n_clusters, n_init=10,
                                 max_no_improvement=10, verbose=0)
        elif clust_alg == 'KMeans':
            ca = KMeans(init='k-means++', n_clusters = n_clusters,
                        n_init = 10, n_jobs = n_jobs)
        else:
            raise Exception("Invalid algorithm name, only KMeans and MiniBatchKMeans are supported.")

        #Applying the clustering algorithm
        ca.fit(interm_rep.arr_xyz)
        labels = ca.labels_

        #Plotting the voxels with the cluster labels.
#        pp.plot_clustering_intermediate_representation(interm_rep, labels)

        #Computing the unique cluster indentifiers
        l_unique = np.unique(labels)

        mean_voxels = np.zeros((len(l_unique), interm_rep.arr_voxels.shape[1]))
        mean_xyz = np.zeros((len(l_unique), interm_rep.arr_xyz.shape[1]), np.float32)

        clust_sizes = []

        cont = 0
        for i in l_unique:
            #Taking the possitions corresponding to the same cluster.
            pos = np.where(labels == i)[0]
            clust_sizes.append(len(pos))
            #Taking data from these possitions and computing the mean time serie
            m_voxel = interm_rep.arr_voxels[pos].mean(0)
            #Taking the xyz from these positions and computing the mean value
            m_xyz = interm_rep.arr_xyz[pos].mean(0)

            mean_voxels[cont] = m_voxel
            mean_xyz[cont] = m_xyz

            cont += 1

        print "min cluster size %i and max cluster size %i" %(np.min(clust_sizes), np.max(clust_sizes))
        #The new intermediate representation is given by mean_voxels and
        # mean_xyz.
        #pdb.set_trace()

        #Computing similarity matrix and applying the threshold
        adj_mat = np.zeros((len(mean_voxels), len(mean_voxels)),
                           dtype = np.float)
        for j in range(len(mean_voxels) - 1):
            for k in range(j + 1, len(mean_voxels)):
                if similarity_measure == 'pearson':
                    aux = st.pearsonr(mean_voxels[j], mean_voxels[k])[0]
                else:
                    aux = skpw.pairwise_kernel(mean_voxels[j], mean_voxels[k],
                                               metric = similarity_measure,
                                               n_jobs = n_jobs)
                if aux >= threshold:
                    adj_mat[j,k] = aux
                    adj_mat[k,j] = aux

        #Building the graph from the adjacency matrix
        g = nx.from_numpy_matrix(adj_mat)

        #Drawing the graph
#        p = dict(zip(g, np.asarray(mean_xyz[:,:2])))
#        nx.draw_networkx(g, pos = p, labels = g.degree())

        #Plot Graphs
#        pp.plot_graph(mean_xyz, g)

        return g



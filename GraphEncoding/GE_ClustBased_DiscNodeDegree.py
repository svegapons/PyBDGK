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

class GE_ClustBased_DiscNodeDegree(GE_Base):
    """
    Graph encoding based on clustering algorithms. Given a set of voxels, a
    clustering algorithm is applied and then a node is assigned to each
    cluster. Each node takes the average voxel time serie of the voxels
    belonging to the node. Edges are computed by applying a similarity
    measure between time series and keeping the values higher than a fixed
    threshold. After nodes and edges are computed, for each node the node
    degree value is computed. According to a parameter n_degree_cat every node
    degree is assigned to a specific category and these category values are
    used as node labels.
    """

    def encode(self, interm_rep, clust_alg='MiniBatchKMeans',
               n_clusters=-1,
               clust_ratio=10,
               similarity_measure="pearson",
               threshold=0.5, n_categ=10, n_jobs=1, **kwds):
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
            If n_cluster not equal to -1 this parameter is ignored.

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

        n_categ: int
            The number of categories to associate node degree values.
            Each category value will be used as node label.

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

        #Computing similarity matrix and applying the threshold
        adj_mat = np.zeros((len(mean_voxels), len(mean_voxels)),
                           dtype = np.byte)
        for j in range(len(mean_voxels) - 1):
            for k in range(j + 1, len(mean_voxels)):
                if similarity_measure == 'pearson':
                    aux = st.pearsonr(mean_voxels[j], mean_voxels[k])[0]
                else:
                    aux = skpw.pairwise_kernel(mean_voxels[j], mean_voxels[k],
                                               metric = similarity_measure,
                                               n_jobs = n_jobs)
                if aux >= threshold:
                    adj_mat[j,k] = 1
                    adj_mat[k,j] = 1

        #Building the graph from the adjacency matrix
        g = nx.from_numpy_matrix(adj_mat)

        #Taking all values of node degrees
        node_deg_unique = np.unique(g.degree().values())
        #Making n_categ number of categories with this values.
        cats = np.array_split(node_deg_unique, n_categ)
        #Creating an array where for each node degree value the corresponding
        #category is associated.
        deg_to_categs = np.zeros(np.max(node_deg_unique) + 1, dtype = np.int32)
        for i in range(len(cats)):
            for j in cats[i]:
                deg_to_categs[j] = i

        #Writing the value of the category in the attribute 'node_label' of the
        #node.
        for node in g.nodes():
            g.node[node]['node_label'] = deg_to_categs[g.degree(node)]
        return g



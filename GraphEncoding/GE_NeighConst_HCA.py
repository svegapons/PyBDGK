"""
"""
import numpy as np
import scipy.stats as st
import networkx as nx
import sklearn.metrics.pairwise as skpw
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.cluster import MiniBatchKMeans, KMeans, Ward
import PyBDGK.IntermRepresentation.IntermRepresentation as ir
from PyBDGK.GraphEncoding.GE_Base import GE_Base
import PyBDGK.Visualization.Plot as pp
import os
import pdb

class GE_NeighConst_HCA(GE_Base):
    """
    Graph encoding based on geometrical connectivity constraints and
    hierarchical clustering algorithms.
    """

    def encode(self, interm_rep, neighborhood_size = 26,
               clust_ratio=10,
               encoding='geometrical',
               similarity_measure='pearson',
               threshold=0.3, n_jobs=1, **kwds):
        """
        Parameters
        ----------
        interm_rep: IntermRep
            IntermRep object containing the arr_xyz and arr_voxel matrixes.
        neighborhood_size: int
            Number of neighbors each voxel will be connected to.
        clust_ratio: int
            The number of clusters will be equal to n/clust_ratio, where n is
            the number of voxels.
        encoding: string
            Type of encoding. 'geometrical' and 'functional' are allowed.
        similarity_measure: string
            Similarity measure used to compare the representative value of each
            parcel (cluster). 'pearson' or the measures available in scikit-learn
            are allowed.
        threshold: float
            Threshold applied to the similarity values in order to define the
            edges in the graph.

        Returns
        -------
        g: Graph
            Networkx graph representing the graph encoding of the data.
        """

        #computing the connectivity matrix, each voxel is connected to
        #"neighborhood_size" neighbors.
        #
        conn = kneighbors_graph(interm_rep.arr_xyz, n_neighbors=neighborhood_size)
#        conn_n = kneighbors_graph(interm_rep.arr_xyz, n_neighbors=neighborhood_size)
#        conn_r = radius_neighbors_graph(interm_rep.arr_xyz, radius=10)
#        conn = conn_n * conn_r

        #Hierarchical clustering algorithm. The number of clusters is defined
        #accoring to the parameter "clust_ratio".
        ward = Ward(n_clusters=len(interm_rep.arr_xyz)/clust_ratio, connectivity=conn)
        #ward = Ward(n_clusters=60, connectivity=conn)

        #Type of encoding: geometrical (only xyz data is used) or
        # functional (voxel time series is used).
        if encoding=='geometrical':
            ward.fit(interm_rep.arr_xyz)
        elif encoding=='functional':
            ward.fit(interm_rep.arr_voxels)

        labels = ward.labels_

        #Plotting the voxels with the cluster labels.
        #pp.plot_clustering_intermediate_representation(interm_rep, labels*10)


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


        #plotting the voxels time series for each cluster
        #pp.plot_interm_representation_time_series(ir.IntermRep(mean_voxels, mean_xyz))

        #The new intermediate representation is given by mean_voxels and
        # mean_xyz.

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


#        #Weighted encoding (for graph kernels that work with weighted graphs)
#        #------------------------------------
#        adj_mat = np.zeros((len(mean_voxels), len(mean_voxels)),
#                           dtype = np.float)
#        for j in range(len(mean_voxels) - 1):
#            for k in range(j + 1, len(mean_voxels)):
#                if similarity_measure == 'pearson':
#                    aux = st.pearsonr(mean_voxels[j], mean_voxels[k])[0]
#                else:
#                    aux = skpw.pairwise_kernel(mean_voxels[j], mean_voxels[k],
#                                               metric = similarity_measure,
#                                               n_jobs = n_jobs)
##                if aux >= threshold:
##                    adj_mat[j,k] = aux
##                    adj_mat[k,j] = aux
#                adj_mat[j,k] = adj_mat[k,j] = aux
#        adj_mat = (adj_mat - np.mean(adj_mat))/np.std(adj_mat)
#        adj_mat = (adj_mat - np.min(adj_mat))/(np.max(adj_mat) - np.min(adj_mat))
#        adj_mat = np.where(adj_mat>=threshold, 1, 0)
#        #------------------------------------


        #Building the graph from the adjacency matrix
        g = nx.from_numpy_matrix(adj_mat)

        #Spliting the node degrees into some categories and using them as node labels.
#        num_lab = 5
        deg = g.degree()
#        for k in deg:
#            deg[k]/= num_lab
        nx.set_node_attributes(g, 'node_label', deg)

        ############
        #Storing the mean time-series of each parcell as a node attribute
        ts_att = {}
        mv = mean_voxels.tolist()
        for pos in range(len(mv)):
            ts_att[pos] = mv[pos]
        nx.set_node_attributes(g, 'time_series', ts_att)



        #Saving the graphs for CLFR subject (the one for which I have the structural data)
#        if interm_rep.subj_name == 'CLFR':
#            nx.write_gexf(g, 'graph_gephi_format.gexf')
#            np.savetxt('CLFR_clusters_xyz.txt', mean_xyz, fmt='%1d', delimiter=' ')
#            edges = np.array(np.where(adj_mat==1)).T
#            np.savetxt('CLFR_clusters_timeseries_cond%s.txt' %(interm_rep.cls), edges, fmt='%1d', delimiter=' ')


        #Plot Graphs
        #pp.plot_graph(mean_xyz, g)

        return g



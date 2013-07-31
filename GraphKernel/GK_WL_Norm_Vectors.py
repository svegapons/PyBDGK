import numpy as np
import scipy as sc
import networkx as nx
import scipy.interpolate as interp
from PyBDGK.GraphKernel.GK_Base import GK_Base
import copy
import pdb


class GK_WL_NV(GK_Base):
    """
    Weisfeiler_Lehman graph kernel
    """
    def compare(self, g1, g2, h=1, nl=False):
        """
        Compute the kernel value between the two graphs.     
        @param g1: First graph (GraphW)
        @param g2: Second graph (GraphW)
        @param h: a natural number (number of iterations of WL)
        @param nl: a boolean (True if we want to use original node labels,
                   False otherwise)
        @return: [k, runtime], where @k is the similarity value between the 
        graphs in graphList and @runtime is the total runtime in seconds.
        """
        gl = [g1, g2]
        return self.compare_list(gl, h, nl)[0,1]
        
    def compare_normalized(self, g1, g2, h=1, nl=False):
        """
        Compute the kernel value between two graphs. A normalized version of
        the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        """
        gl = [g1, g2]
        return self.compare_list_normalized(gl, h, nl)[0,1]
       
        
    def compare_list(self, graph_list, h=1, nl=False):
        """
        Compute the all-pairs kernel values for a list of graphs.     
        @param graph_list: A list of graphs to be compared
        @param h: a natural number (number of iterations of WL)
        @param nl: a boolean (True if we want to use original node labels,
                   False otherwise)
        @return: [K, runtime], where @K is the similarity matrix of all the 
        graphs in graphList and @runtime is the total runtime in seconds.
        """
        
        self.graphs = graph_list
        n = len(graph_list)
        lists = [0]*(n)
        k = [0]*(h+1)
        n_nodes = 0
        n_max = 0
        
        #Compute adjacency lists and n_nodes, the total number of nodes in
        #the dataset.
        for i in range(n):
            #the graph should be a networkx graph or having the same methods.
            lists[i] = graph_list[i].adjacency_list()
            n_nodes = n_nodes + graph_list[i].number_of_nodes()
            
            #Computing the maximum number of nodes in the graphs. It will be 
            #used in the computation of vectorial representation.
            if(n_max < graph_list[i].number_of_nodes()):
                n_max = graph_list[i].number_of_nodes()
            
        phi = np.zeros((n_max, n), dtype = np.uint64)
        #each column j of phi will be the explicit feature representation
        # for the graph j.
        #n_max is enough to store all possible labels
        
        #INITIALIZATION
        #initialize the nodes labels for each graph with their labels or 
        #with degrees (for unlabeled graphs)
        
        labels = [0] * n
        label_lookup = {}
        label_counter = 0

        # label_lookup is an associative array, which will contain the
        # mapping from multiset labels (strings) to short labels (integers)
        
        if nl == True:
            for i in range(n):
                l_aux = nx.get_node_attributes(graph_list[i], 
                                               'node_label').values()
                #It is assumed that the graph has an attribute 'node_label'
                labels[i] = np.zeros(len(l_aux), dtype = np.int32) 
                
                for j in range(len(l_aux)):
                    if not label_lookup.has_key(l_aux[j]):
                        label_lookup[l_aux[j]] = label_counter
                        labels[i][j] = label_counter
                        label_counter += 1
                    else:
                        labels[i][j] = label_lookup[l_aux[j]]
                    #labels are associated to a natural number starting with 0.
                    phi[labels[i][j], i] += 1
        else:
            for i in range(n):
                labels[i] = np.array(graph_list[i].degree().values())    
                for j in range(len(labels[i])):
                    phi[labels[i][j], i] += 1
                print(str(i + 1) + " from " + str(n) + " completed")
                                   
        #Vectorial representation of graphs. It is just taken the original 
        #nodes degree.
        vects = np.copy(phi.transpose())   
        self.vectors = np.zeros(vects.shape)
        
        ave_length = 0
        for i in vects:
            ave_length += len(i)
        ave_length = ave_length / len(vects)        
        mat = np.zeros((len(vects), ave_length))
        for i in range(len(vects)):
            r = interp.interp1d(range(len(vects[i])), vects[i], 
                                kind = 'quadratic')
            for j in range(ave_length):
                mat[i,j] = r(j * (len(vects[i])) / ave_length)  
            self.vectors[i] = mat[i]
    
        #Selecting part of the vectorial representation
        mat = mat[:, len(mat[0])*3/4 :]
        #    
    
        k = np.dot(mat, mat.T)  
        
        return k
        

    def compare_list_normalized(self, graph_list, h, nl):      
        """
        Using the normalized kernel trying to avoid the effect of size differences on graphs.
        k'(x, y) = k(x, y) / sqrt(k(x, x) * k(y, y))
        """
        k = self.compare_list(graph_list, h, nl)
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i,j] = k[i,j] / np.sqrt(k[i,i] * k[j,j])
        
        return k_norm
                    
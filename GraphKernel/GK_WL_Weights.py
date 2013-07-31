import numpy as np
import scipy as sc
import networkx as nx
from PyBDGK.GraphKernel.GK_Base import GK_Base
import copy
import pdb


class GK_WL_Weights(GK_Base):
    """
    Simple weighted version of the Weisfeiler_Lehman graph kernel.
    """
    def compare(self, g1, g2, h=1, nl=False, verbose=False):
        """
        Compute the kernel value between the two graphs.     
        @param g1: First graph (Networkx)
        @param g2: Second graph (Networkx)
        @param h: a natural number (number of iterations of WL)
        @param nl: a boolean (True if we want to use original node labels,
                   False otherwise)
        @return: k: the similarity value between the two graphs. 
        """
        gl = [g1, g2]
        return self.compare_list(gl, h, nl, verbose)[0,1]
        
        
    def compare_normalized(self, g1, g2, h=1, nl=False, verbose=False):
        """
        Compute the kernel value between two graphs. A normalized version of
        the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        """
        gl = [g1, g2]
        return self.compare_list_normalized(gl, h, nl, verbose)[0,1]
       
        
    def compare_list(self, graph_list, h=1, nl=False, verbose=False):
        """
        Compute the all-pairs kernel values for a list of graphs.     
        @param graph_list: A list with the graphs to be compared.
        @param h: a natural number (number of iterations of WL)
        @param nl: a boolean (True if we want to use original node labels,
                   False otherwise)
        @return: K: the similarity matrix of all the 
        graphs in graph_list.
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
            
        phi = np.zeros((n_max, n), dtype = np.float)
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
            raise Exception("Not implemented for this graph kernel.")
        else:
            for i in range(n):
                e_aux = nx.get_edge_attributes(graph_list[i], 'weight')
                #It is assumed that the graph has an attribute 'weight'.
                for j in range(len(lists[i])):
                    sm = 0
                    for k in lists[i][j]:
                        if j < k:
                            sm += e_aux[j,k]
                    phi[int(sm), i] += 1
                if verbose:
                    print(str(i + 1) + " from " + str(n) + " completed")
                                   
        #Simplified vectorial representation of graphs (just taking the 
        #vectors before the kernel iterations), i.e., it is just the original 
        #nodes degree.
        self.vectors = np.copy(phi.transpose())   
        
        k = np.dot(phi.transpose(), phi)
        
        ### MAIN LOOP
        it = 0
        new_labels = copy.deepcopy(labels)
        
        
        while it < h:
            if verbose:                
                print("iter=", str(it))
            # create an empty lookup table
            label_lookup = {}
            label_counter = 0

            phi = np.zeros((n_nodes, n), dtype = np.float)
            for i in range(n):
                for v in range(len(lists[i])):
                    # form a multiset label of the node v of the i'th graph
                    # and convert it to a string
#                    pdb.set_trace()
                    long_label = np.concatenate((np.array([labels[i][v]]), np.sort(labels[i][lists[i][v]])))
                    long_label_string = str(long_label)
                    # if the multiset label has not yet occurred, add it to the
                    # lookup table and assign a number to it
                    if not label_lookup.has_key(long_label_string):
                        label_lookup[long_label_string] = label_counter
                        new_labels[i][v] = label_counter
                        label_counter += 1
                    else:
                        new_labels[i][v] = label_lookup[long_label_string]
                # fill the column for i'th graph in phi
                aux = np.bincount(new_labels[i])
                #NOTA estoy asumiendo q np.bincount hace lo mismo q accumarray de Matlab. Verificar!!
                phi[new_labels[i], i] += aux[new_labels[i]]
            
            if verbose:
                print("Number of compressed labels: ", str(label_counter))
            #pdb.set_trace()
            
            k += np.dot(phi.transpose(), phi)
            labels = copy.deepcopy(new_labels)
            it = it + 1
        return k
        

    def compare_list_normalized(self, graph_list, h, nl, verbose=False):      
        """
        Normalized version of the kernel trying to avoid the effect of 
        big differences in sizes on the graphs. The normalized kernel is 
        obtained by the following equation:
        k'(x, y) = k(x, y) / sqrt(k(x, x) * k(y, y))
        """
        k = self.compare_list(graph_list, h, nl)
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i,j] = k[i,j] / np.sqrt(k[i,i] * k[j,j])
        
        return k_norm
                    
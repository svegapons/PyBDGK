# -*- coding: utf-8 -*-
"""
"""
import networkx as nx
import numpy as np
import PyBDGK.IntermRepresentation as ir
import os
import pdb

class GK_Base(object):
    """Base class for graph kernels

    Notes
    -----
    """
    
    graphs = None
    vectors = None

    def compare(self, g1, g2):
        """
        Compute the kernel value between the two graphs.     
        @param g1: First graph (networkx)
        @param g2: Second graph (networkx)
        @return: k, the similarity value between g1 and g2.
        """

        
    def compare_normalized(self, g1, g2):
        """
        Compute the kernel value between two graphs. A normalized version of
        the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        """
        
    def compare_list(self, graph_list):
        """
        Compute the all-pairs kernel values for a list of graphs.     
        @param graph_list: A list of graphs to be compared
        @return: K, the similarity matrix of all the 
        graphs in graph_list.
        """
        
    def compare_list_normalized(self, graph_list):
        """
        Compute the all-pairs kernel values for a list of graphs. A normalized
        version of the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        """
    
    def get_vectorial_representation(self):
        """
        Gives the vectorial representation of the previously compared graphs.
        """
        return self.vectors
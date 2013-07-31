
import numpy as np
import networkx as nx
import PyBDGK.IntermRepresentation.IntermRepresentation as ir
import PyBDGK.GraphEncoding.GE_FunctionalConnectivity as ge
import PyBDGK.GraphKernel.GK_WL as gk
import os
import pdb


def test():
    """
    """
    d = "C:\WinPython-64bit-2.7.3.3\python-2.7.3.amd64\Lib\site-packages\PyBDGK\Data\Test"
    p1 = os.path.join(d, "blur.1.ANGO.steadystate.TRIM_graymask_dump")
    p2 = os.path.join(d, "xyz_coords_graymattermask_ANGO")
    
    interm = ir.IntermRep.load_from_file(p1, p2)
    fc = ge.GE_FunctionalConnectivity()
    
    g = fc.encode(interm)
    return g
    
def simple_test():
    """
    """
    a1 = np.array([[12,11,3],[14,25,4]])
    interm = ir.IntermRep(a1, None)
    fc = ge.GE_FunctionalConnectivity()    
    g = fc.encode(interm, n_jobs = 1)
    return g
    
    
def simple_graph_kernel_test():
    """
    """
    a1 = np.array([[12,11,3],[14,25,4],[67,3,5],[67,34,67],[89,9,123]])
    interm1 = ir.IntermRep(a1, None)
    fc1 = ge.GE_FunctionalConnectivity()    
    g1 = fc1.encode(interm1, n_jobs = 1)
    
    a2 = np.array([[123,121,33],[45,5,5],[45,2,56],[567,3,24],[67,4,3]])
    interm2 = ir.IntermRep(a2, None)
    fc2 = ge.GE_FunctionalConnectivity()    
    g2 = fc2.encode(interm2, n_jobs = 1)
    
    gk_wl = gk.GK_WL()
    k = gk_wl.compare_normalized(g1, g2)
    return k
    
    
def paper_example():
    """
    Experiment with the toy example in the original paper of the WL kernel:
    Shervashidze, N., et al.: Weisfeiler_Lehman Graph Kernels, Journal of 
    Machine Learning Research 12, pp. 2539 - 2561, 2011.
    """
    #Creating the first graph
    g1 = nx.Graph()
    g1.add_edges_from([(1,3),(2,3),(3,4),(3,0),(4,5),(4,0),(5,0)])    
    g1.node[0]["node_label"] = 3
    g1.node[1]["node_label"] = 1
    g1.node[2]["node_label"] = 1
    g1.node[3]["node_label"] = 4
    g1.node[4]["node_label"] = 5
    g1.node[5]["node_label"] = 2
#    plot(g1, layout = "circle")
    
    #Creating the second graph
    g2 = nx.Graph()
    g2.add_edges_from([(1,0),(2,3),(3,4),(3,5),(3,0),(4,5),(5,0)])    
    g2.node[0]["node_label"] = 3
    g2.node[1]["node_label"] = 2
    g2.node[2]["node_label"] = 1
    g2.node[3]["node_label"] = 4
    g2.node[4]["node_label"] = 2
    g2.node[5]["node_label"] = 5
#    plot(g2, layout = "circle")
    #Computing the graph kernel
#    pdb.set_trace()
    wl = gk.GK_WL()
    return wl.compare(g1, g2, h =1, nl = True)
    
    
    
    
    
    
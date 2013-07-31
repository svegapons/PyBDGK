
import numpy as np
import scipy as sc
import matplotlib.pyplot as pp
import pylab
from mpl_toolkits.mplot3d import Axes3D
import PyBDGK.IntermRepresentation.IntermRepresentation as ir



def plot_intermediate_representation(interm_rep):
    """
    plotting an intermediate representation of brain data...
    """
    fig = pylab.figure()
    ax = Axes3D(fig)

    ax.scatter(interm_rep.arr_xyz[:,0], interm_rep.arr_xyz[:,1], interm_rep.arr_xyz[:,2])
    pp.show()
    

def plot_clustering_intermediate_representation(interm_rep, labels):
    """
    plotting the clustering of an intermediate representation...
    """
    fig = pylab.figure()
    ax = Axes3D(fig)

    ax.scatter(interm_rep.arr_xyz[:,0], interm_rep.arr_xyz[:,1], interm_rep.arr_xyz[:,2], c = labels)
    pp.show()
    
    
def plot_graph(arr_xyz, g):
    """
    """
    
    fig = pylab.figure()
    ax = Axes3D(fig)
    
    #Plotting the nodes
    ax.scatter(arr_xyz[:,0], arr_xyz[:,1], arr_xyz[:,2], c = 'k', marker = 'o', linewidth=10)
#    tx = ["10"] * len(arr_xyz)
#    ax.text3D(arr_xyz[:,0], arr_xyz[:,1], arr_xyz[:,2], tx)
        
    for e in g.edges():
        ax.plot(arr_xyz[e,0], arr_xyz[e,1], arr_xyz[e,2], 'b-')
    pp.show()
    

import numpy as np
import scipy as sc
import matplotlib.pyplot as pp



def PlotMatrix(matrix):
    """
    Plot a similarity matrix.
    """
    iplot = pp.imshow(matrix)
    iplot.set_cmap('spectral')
    pp.colorbar()
    pp.show()
    
    
def PlotFeatureVectors(n_nodes, n_edges, vectors, vectors_norm, classes):
    """
    """
    cls = np.unique(classes)
    fig = pp.figure()
    fig.subplots_adjust(wspace = 1.0)
    pp.subplot(211)    
    pp.title('Number of edges vs. Number of non-trivial nodes')
    for i in range(len(n_nodes)):
        if(classes[i] == cls[0]):
            pp.plot(n_nodes[i], n_edges[i], 'ro')
            #pp.plot(range(vectors.shape[1]), vectors[i, :], 'ro', range(vectors.shape[1]), vectors[i, :], 'r-')
        else:
            pp.plot(n_nodes[i], n_edges[i], 'b^')  
    pp.ylabel('Number of edges')
    pp.xlabel('Number of non-trivial nodes')
    pp.legend()
    
    pp.subplot(212)
    #mi = vectors.min()
    ma = int(vectors.max())
    pp.title('Feature vectors (Only first iteration = number of nodes for each node degree value)')
    for i in range(len(vectors)):
        if(classes[i] == cls[0]):
            pp.plot(range(vectors.shape[1]), vectors[i, :], 'ro')
            #pp.plot(range(vectors.shape[1]), vectors[i, :], 'ro', range(vectors.shape[1]), vectors[i, :], 'r-')
        else:
            pp.plot(range(vectors.shape[1]), vectors[i, :], 'b^')
            #pp.plot(range(vectors.shape[1]), vectors[i, :], 'b^', range(vectors.shape[1]), vectors[i, :], 'b-')
            
#    pp.subplot(313)
#    maN = int(vectors_norm.max())
#    pp.title('Normalized feature vectors (after first iteration)')
#    for i in range(len(vectors_norm)):
#        if(classes[i] == cls[0]):
#            pp.plot(range(vectors_norm.shape[1]), vectors_norm[i, :], 'ro')
#        else:
#            pp.plot(range(vectors_norm.shape[1]), vectors_norm[i, :], 'b^')
    pp.xticks(range(0, vectors_norm.shape[1] + 11, 50))
    pp.yticks(range(0, ma + 6, 20))
    pp.ylabel('')
    pp.xlabel('vector values')
    pp.show()
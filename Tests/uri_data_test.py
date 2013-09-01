
import numpy as np
import networkx as nx
import PyBDGK.IntermRepresentation.IntermRepresentation as ir
import PyBDGK.GraphEncoding.GE_ClusterBased as ge
import PyBDGK.GraphEncoding.GE_ClustBased_PercCompleteness as gpc
import PyBDGK.GraphEncoding.GE_ClustBased_DiscNodeDegree as gnd
import PyBDGK.GraphEncoding.GE_FuncConn_WeightedEncoding as gwe
import PyBDGK.GraphEncoding.GE_NeighConst_HCA as gnc
import PyBDGK.GraphKernel.GK_WL_Weights as gkw
import PyBDGK.GraphKernel.GK_WL as gk
import PyBDGK.GraphKernel.GK_WL_Norm_Vectors as gknv
import PyBDGK.Classsification.Classification as cl
import PyBDGK.Visualization.V_Vectors as vs
import PyBDGK.Utils.Utils as ut
import PyBDGK.Visualization.Plot as pp
import os
import pdb


def uri_data_experiment(directory):
    """
    Experiment with the Uri dataset.
    Different encodings and graph kernels are tested!

    Parameters
    ----------
    directory: string
        The path of the directory containing all data files.

    """

    #######################################################################
    #LOADING DATA
    #######################################################################

    #Loading all files in the folder
    xyz_files = []
    blur_files = []

    #Spliting files in xyz coordinates and voxels data.
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            if f.startswith('xyz'):
                xyz_files.append(f)
            if f.startswith('blur'):
                blur_files.append(f)

    #Loading xyz data
    dict_xyz = {}
    for f in xyz_files:
        #The name of the subject is given by the four last letter.
        dict_xyz[f[-4:]] = np.genfromtxt(os.path.join(directory, f),
                                         dtype = float, delimiter = ' ')
        print "xyz_file for subject %s was loaded." %(f[-4:])

    #Loading voxels data and creating the intermediate representation objects
    inter_reps = []
    for f in blur_files:
        #Name of the subject is always in positions [7:11]
        s_name = f[7:11]
        #Class is in possition 5
        cls = int(f[5])
        arr_voxels = np.genfromtxt(os.path.join(directory, f), dtype = float,
                                       delimiter = ' ')
        inter_reps.append(ir.IntermRep(arr_voxels, dict_xyz[s_name], s_name,
                                       cls))

        print "Intermediate representation for subject %s and class %s created." %(s_name, cls)

    #######################################################################


    #######################################################################
    #Computing the Graph Encoding
    #######################################################################

    graphs = []
    classes = []
    subjects = []

    #Cluster based graph encoding with a fixed threshold
    #-----------------------------
#    fc = ge.GE_ClusterBased()
#    for i_rep in inter_reps:
#        graphs.append(fc.encode(i_rep, clust_alg = 'MiniBatchKMeans',
#                                n_clusters = -1, clust_ratio = 170,
#                                similarity_measure="pearson",
#                                threshold=0.1, n_jobs = 1))
#        classes.append(i_rep.cls)
#        subjects.append(i_rep.subj_name)
#        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
#        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
#                                                           graphs[-1].number_of_edges())
#        print ""
    #---------------------------

    #Cluster based graph encoding with percentage of completeness
    #-----------------------------
#    fc = gpc.GE_ClustBased_PercCompleteness()
#    for i_rep in inter_reps:
#        graphs.append(fc.encode(i_rep, clust_alg = 'MiniBatchKMeans',
#                                n_clusters = -1, clust_ratio = 110,
#                                similarity_measure="pearson",
#                                completeness=0.3, n_jobs = 1))
#        classes.append(i_rep.cls)
#        subjects.append(i_rep.subj_name)
#        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
#        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
#                                                           graphs[-1].number_of_edges())
#        print ""
    #---------------------------

    #Cluster based graph encoding with categories for node degree
    #-----------------------------
#    fc = gnd.GE_ClustBased_DiscNodeDegree()
#    for i_rep in inter_reps:
#        graphs.append(fc.encode(i_rep, clust_alg = 'MiniBatchKMeans',
#                                n_clusters = -1, clust_ratio = 170,
#                                similarity_measure="pearson",
#                                threshold=0.1, n_categ=10, n_jobs = 1))
#        classes.append(i_rep.cls)
#        subjects.append(i_rep.subj_name)
#        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
#        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
#                                                           graphs[-1].number_of_edges())
#        print ""
    #---------------------------


    #Weighted version of graph encoding with a fixed threshold
    #-----------------------------
#    fc = gwe.GE_FuncConn_WeightedEncoding()
#    for i_rep in inter_reps:
#        graphs.append(fc.encode(i_rep, clust_alg = 'MiniBatchKMeans',
#                                n_clusters = -1, clust_ratio = 200,
#                                similarity_measure="pearson",
#                                threshold=0.5, n_jobs = 1))
#        classes.append(i_rep.cls)
#        subjects.append(i_rep.subj_name)
#        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
#        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
#                                                           graphs[-1].number_of_edges())
#        print ""
    #---------------------------

    # Graph encoding based on Neirghboring connections and hierarchical clustering algorithm.
    #-----------------------------
    fc =  gnc.GE_NeighConst_HCA()
    for i_rep in inter_reps:
        graphs.append(fc.encode(i_rep))
        classes.append(i_rep.cls)
        subjects.append(i_rep.subj_name)
        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
                                                           graphs[-1].number_of_edges())
        print ""
    #---------------------------

    #######################################################################


    #######################################################################
    #Reordering data for the leave-one-subject-out cross-validation
    #######################################################################

    #Permutting elements for a further leave-two-out cv (leaving out
    #two samples corresponding to the same subject avoiding problems with
    #unbalanced data).
    nm_graphs = [None] * len(graphs)
    nm_classes = [None] * len(classes)
    nm_subjects = [None] * len(subjects)

    for i in range(len(graphs) / 2):
        nm_graphs[i*2] = graphs[i]
        nm_graphs[i*2 + 1] = graphs[(len(graphs) / 2) + i]
        nm_classes[i*2] = classes[i]
        nm_classes[i*2 + 1] = classes[(len(classes) / 2) + i]
        nm_subjects[i*2] = subjects[i]
        nm_subjects[i*2 + 1] = subjects[(len(subjects) / 2) + i]

    print nm_subjects
    print nm_classes

#    #Testing if I get chance level when I permutted the class label...
#    np.random.shuffle(nm_classes)
#    np.random.shuffle(nm_classes)
#    print nm_classes

    #######################################################################


    #######################################################################
    #Computing the Graph Kernel
    #######################################################################

    #Computing the kernel matrix by using WL graph kernel.
    gk_wl = gk.GK_WL()
    k_matrix = gk_wl.compare_list_normalized(nm_graphs, h = 1, nl = False)

    #Computing the kernel matrix with the normalized vectors graph kernel
#    gk_wl = gknv.GK_WL_NV()
#    k_norm_mat = gk_wl.compare_list_normalized(nm_graphs, h = 2, nl = False)

    #Computing the kernel matrix by using the weighted version of WL.
#    gk_wl = gkw.GK_WL_Weights()
#    k_matrix = gk_wl.compare_list_normalized(nm_graphs, h = 0, nl = False)

    #######################################################################


    #######################################################################
    #Ploting the similarity matrix
    #######################################################################

    #Ploting the similarity matrix, the matrix is permuted to have all
    #samples belonging to the first class at the beggining.
#    perm_matrix = ut.PermuteMatrix(k_matrix, nm_classes)
#    vs.PlotMatrix(perm_matrix)

    #Making a list with number of nodes and edges of all graphs. They will be
    #used in the plotting.
    n_nodes = []
    n_edges = []
    for g in nm_graphs:
        n_nodes.append(g.number_of_nodes())
        n_edges.append(g.number_of_edges())

    #Plotting the vectorial representation of each graph. In the picture we
    #include number_of_nodes and number_of_edges, original_vectors and
    #normalized_vectors.
    #vs.PlotFeatureVectors(n_nodes, n_edges, gk_wl.vectors, gk_wl_nv.vectors, nm_classes)
    vs.PlotFeatureVectors(n_nodes, n_edges, gk_wl.vectors, gk_wl.vectors, nm_classes)

    #######################################################################


    #######################################################################
    #Leave-one-subject-out cross-validation
    #######################################################################

    preds, scores = cl.subject_fold_cv(k_matrix, np.array(nm_classes),
                                       n_subjects = 19)
    print ""
    print "Predictions: "
    print preds
    print ""
    print "Scores:"
    print scores

    #######################################################################


    return scores





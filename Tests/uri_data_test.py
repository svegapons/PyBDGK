
import numpy as np
import networkx as nx
import scipy.stats as st
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
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import normalize
import os
import pdb
from scipy.io import loadmat
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pylab as pl
import matplotlib.pyplot as pp

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
        inter_reps.append(ir.IntermRep(arr_voxels,dict_xyz[s_name],s_name,cls))

#        print inter_reps[-1].arr_voxels.shape
#        inter_reps.append(ir.IntermRep(arr_voxels, dict_xyz[s_name], s_name,
#                                       cls))

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
#    fc = gnd.GE_ClustBased_DiscNodeDegree()Just to leave you the location of code+data in a written way ;-)
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
#                                n_clusters = -1, clust_ratio = 100,
#                                similarity_measure="pearson",
#                                threshold=0.0, n_jobs = 1))
#        np.savetxt("g.txt",nx.adj_matrix(graphs[-1]))
#        pdb.set_trace()
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
        graphs.append(fc.encode(i_rep, clust_ratio=120, threshold=0.4))
        #np.savetxt("g.txt",nx.adj_matrix(graphs[-1]))
        #pdb.set_trace()
        classes.append(i_rep.cls)
        subjects.append(i_rep.subj_name)
        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
                                                           graphs[-1].number_of_edges())
        print ""


#        #saving the graph for CLFR subject
#        if i_rep.sub_name == 'CLFR':
#            if i_rep.cls == 1:

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


    #########################
    #FEATURE SELECTION!!
    #########################
#    fs_graphs = []
#    for i in range(len(nm_graphs) / 2):
#        g1 = nm_graphs[i * 2]
#        g2 = nm_graphs[i * 2 + 1]
#        evals = np.zeros(g1.number_of_nodes())
#        ts1 = nx.get_node_attributes(g1,'time_series').values()
#        ts2 = nx.get_node_attributes(g2,'time_series').values()
#        #pdb.set_trace()
#        for nd in range(len(ts1)):
##            evals[nd] = np.linalg.norm(np.array(ts1[nd] - np.mean(ts1[nd])) - np.array(ts2[nd] - np.mean(ts2[nd])))
#            evals[nd] = np.abs(st.pearsonr(ts1[nd], ts2[nd])[0])
#            evals[nd] = st.pearsonr(ts1[nd], ts2[nd])[0]
#        print evals
#        m = np.mean(evals)
#        idx = np.where(evals > m)[0]
#        gg1 = nx.from_numpy_matrix(nx.adj_matrix(nx.subgraph(g1, np.array(g1.nodes())[idx])))
#        nx.set_node_attributes(gg1, 'node_label', gg1.degree())
#        fs_graphs.append(gg1)
#        gg2 = nx.from_numpy_matrix(nx.adj_matrix(nx.subgraph(g2, np.array(g2.nodes())[idx])))
#        nx.set_node_attributes(gg2, 'node_label', gg2.degree())
#        #pdb.set_trace()
#        fs_graphs.append(gg2)



    fs_graphs = []
    for i in range(len(nm_graphs) / 2):
        g1 = nm_graphs[i * 2]
        g2 = nm_graphs[i * 2 + 1]
        evals = np.zeros(g1.number_of_nodes())
        ts1 = g1.degree().values()
        ts2 = g2.degree().values()
        #pdb.set_trace()
        evals = np.abs(np.array(ts1) - np.array(ts2))
        print evals
        m = np.mean(evals)
        idx = np.where(evals > m)[0]
        gg1 = nx.from_numpy_matrix(nx.adj_matrix(nx.subgraph(g1, np.array(g1.nodes())[idx])))
        nx.set_node_attributes(gg1, 'node_label', gg1.degree())
        fs_graphs.append(gg1)
        gg2 = nx.from_numpy_matrix(nx.adj_matrix(nx.subgraph(g2, np.array(g2.nodes())[idx])))
        nx.set_node_attributes(gg2, 'node_label', gg2.degree())
        #pdb.set_trace()
        fs_graphs.append(gg2)
    #########################


    #######################################################################
    #Computing the Graph Kernel
    #######################################################################

    #Computing the kernel matrix by using WL graph kernel.
    gk_wl = gk.GK_WL()
    k_matrix = gk_wl.compare_list_normalized(nm_graphs, h = 1, nl = True)
    #k_matrix = gk_wl.compare_list_normalized(fs_graphs, h = 1, nl = True)

    #########
#    ll = len(gk_wl.vectors[0])
#    num = 5
#    means = []
#    for i in range(ll/num -1):
#        means.append(np.mean(gk_wl.vectors[:, i*num : i*num + num], axis=1))
#    means.append(np.mean(gk_wl.vectors[:, (ll/num -1):], axis=1))
#    v_means = np.vstack(means).T
#    k_matrix = np.dot(v_means, v_means.T)
#    print k_matrix.shape

    ########

    #Computing the kernel matrix with the normalized vectors graph kernel
#    gk_wl = gknv.GK_WL_NV()
#    k_matrix = gk_wl.compare_list_normalized(nm_graphs, h = 1, nl = False)

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
    #for g in nm_graphs:
    for g in nm_graphs:
        n_nodes.append(g.number_of_nodes())
        n_edges.append(g.number_of_edges())

    #Plotting the vectorial representation of each graph. In the picture we
    #include number_of_nodes and number_of_edges, original_vectors and
    #normalized_vectors.
    #vs.PlotFeatureVectors(n_nodes, n_edges, v_means, v_means, nm_classes)
    vs.PlotFeatureVectors(n_nodes, n_edges, gk_wl.vectors, gk_wl.vectors, nm_classes)

    #######################################################################


    #######################################################################
    #Leave-one-subject-out cross-validation
    #######################################################################

    preds, scores = cl.subject_fold_cv(k_matrix, np.array(nm_classes),
                                       n_subjects = 19)

#    selector = SelectPercentile(f_classif, percentile=10)
#    selector.fit(gk_wl.vectors, nm_classes)
#    print selector.pvalues_
#    print selector.scores_

    print ""
    print "Predictions: "
    print preds
    print ""
    print "Scores:"
    print scores

    #######################################################################
    return scores




def mice_experiment(dir_cl1='E:\Dropbox\Codes\Matlab\BTBR', dir_cl2='E:\Dropbox\Codes\Matlab\WT', threshold=0.4):
    """
    """
    #Building graphs for class 0
    graphs_cl1 = []
    for f in os.listdir(dir_cl1):
        mat = loadmat(os.path.join(dir_cl1, f))['nw_re_arranged']
        #Applying a threshold
        bin_mat = np.where(mat>=threshold, mat, 0)
        graphs_cl1.append(nx.from_numpy_matrix(bin_mat))
        #Plotting the graphs
#        fig = pl.figure()
#        nx.draw_spring(graphs_cl1[-1])
    cls1 = np.zeros(len(graphs_cl1))

    #Building graphs for class 1
    graphs_cl2 = []
    for f in os.listdir(dir_cl2):
        mat = loadmat(os.path.join(dir_cl2, f))['nw_re_arranged']
        #Applying a threshold
        bin_mat = np.where(mat>=threshold, mat, 0)
        graphs_cl2.append(nx.from_numpy_matrix(bin_mat))
        #Plotting the graphs
#        fig = pl.figure()
#        nx.draw_circular(graphs_cl2[-1])
    cls2 = np.ones(len(graphs_cl2))

    pp.show()
    #Puttin together both classes
    graphs = graphs_cl1 + graphs_cl2
    classes = np.concatenate((cls1, cls2))

    #Computing the kernel matrix by using WL graph kernel.
    gk_wl = gk.GK_WL()
#    gk_wl = gkw.GK_WL_Weights()
    k_matrix = gk_wl.compare_list_normalized(graphs, h = 3, nl = False)

    clf = SVC(kernel='precomputed')
    cv_scores = cross_val_score(clf, k_matrix, classes, cv=StratifiedKFold(classes, n_folds=len(classes)/2))
    print np.mean(cv_scores)
    print np.std(cv_scores)
    return cv_scores
















def uri_all_data_experiment(directory):
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
        inter_reps.append(ir.IntermRep(arr_voxels,dict_xyz[s_name],s_name,cls))
#        inter_reps.append(ir.IntermRep(arr_voxels, dict_xyz[s_name], s_name,
#                                       cls))

        print "Intermediate representation for subject %s and class %s created." %(s_name, cls)
    #######################################################################


    #######################################################################
    #Computing the Graph Encoding
    #######################################################################
    graphs = []
    classes = []
    subjects = []

    # Graph encoding based on Neirghboring connections and hierarchical clustering algorithm.
    #-----------------------------
    fc =  gnc.GE_NeighConst_HCA()
    for i_rep in inter_reps:
        graphs.append(fc.encode(i_rep, clust_ratio=120, threshold=0.4))
        #np.savetxt("g.txt",nx.adj_matrix(graphs[-1]))
        #pdb.set_trace()
        classes.append(i_rep.cls)
        subjects.append(i_rep.subj_name)
        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
                                                           graphs[-1].number_of_edges())
        print ""

    classes = np.array(classes)
    #Computing the kernel matrix by using WL graph kernel.
    gk_wl = gk.GK_WL()
#    gk_wl = gkw.GK_WL_Weights()
    k_matrix = gk_wl.compare_list_normalized(graphs, h = 1, nl = False)

    clf = SVC(kernel='precomputed')
    cv_scores = cross_val_score(clf, k_matrix, classes, cv=StratifiedKFold(classes, n_folds=len(classes)/len(np.unique(classes))))
    print np.mean(cv_scores)
    return cv_scores

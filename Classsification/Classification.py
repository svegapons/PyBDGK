import numpy as np
import sklearn
import sklearn.cross_validation
import sklearn.svm
import sklearn.neighbors
import random
import pdb

   
def leave_one_out_cv(gram_matrix, labels, alg = 'SVM'):
    """
    leave-one-out cross-validation
    """
    scores = []
    preds = []
    loo = sklearn.cross_validation.LeaveOneOut(len(labels))
    for train_index, test_index in loo:
        X_train, X_test = gram_matrix[train_index][:,train_index], gram_matrix[test_index][:, train_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if(alg == 'SVM'):
            svm = sklearn.svm.SVC(kernel = 'precomputed')
            svm.fit(X_train, y_train)
            preds += svm.predict(X_test).tolist()
            score = svm.score(X_test, y_test)
        elif(alg == 'kNN'):
            knn = sklearn.neighbors.KNeighborsClassifier()
            knn.fit(X_train, y_train)
            preds += knn.predict(X_test).tolist()
            score = knn.score(X_test, y_test)
        scores.append(score)

    print "Mean accuracy: %f" %(np.mean(scores))
    print "Stdv: %f" %(np.std(scores))
    
    return preds, scores
    
    
def k_fold_cv(gram_matrix, labels, folds = 10, alg = 'SVM', shuffle = True):
    """
    K-fold cross-validation
    """
    pdb.set_trace()
    scores = []
    preds = []
    loo = sklearn.cross_validation.KFold(len(labels), folds, shuffle = shuffle, random_state = random.randint(0,100))
    #loo = sklearn.cross_validation.LeaveOneOut(len(labels))
    for train_index, test_index in loo:
        X_train, X_test = gram_matrix[train_index][:,train_index], gram_matrix[test_index][:, train_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if(alg == 'SVM'):
            svm = sklearn.svm.SVC(kernel = 'precomputed')
            svm.fit(X_train, y_train)
            preds += svm.predict(X_test).tolist()
            score = svm.score(X_test, y_test)
        elif(alg == 'kNN'):
            knn = sklearn.neighbors.KNeighborsClassifier()
            knn.fit(X_train, y_train)
            preds += knn.predict(X_test).tolist()
            score = knn.score(X_test, y_test)
            
        scores.append(score)
        
    print "Mean accuracy: %f" %(np.mean(scores))
    print "Stdv: %f" %(np.std(scores))
    
    return preds, scores
    
    
def subject_fold_cv(gram_matrix, labels, n_subjects, alg = 'SVM'):
    """
    Leave-one-subject-out cross-validation
    """
    scores = []
    preds = []
    loo = sklearn.cross_validation.KFold(len(labels), n_subjects)
    #loo = sklearn.cross_validation.LeaveOneOut(len(labels))
    for train_index, test_index in loo:
        X_train, X_test = gram_matrix[train_index][:,train_index], gram_matrix[test_index][:, train_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if(alg == 'SVM'):
            svm = sklearn.svm.SVC(kernel = 'precomputed')
            svm.fit(X_train, y_train)
            preds += svm.predict(X_test).tolist()
            score = svm.score(X_test, y_test)
        elif(alg == 'kNN'):
            knn = sklearn.neighbors.KNeighborsClassifier(1)
            knn.fit(X_train, y_train)
            preds += knn.predict(X_test).tolist()
            score = knn.score(X_test, y_test)
            
        scores.append(score)
        
    print "Mean accuracy: %f" %(np.mean(scores))
    print "Stdv: %f" %(np.std(scores))
    
    return preds, scores
        

        
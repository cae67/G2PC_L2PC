# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:17:25 2021

@author: antho
"""

#%% Import Libraries

# General
import numpy as np

# Loading Data
from scipy.io import loadmat, savemat

# Clustering and Explainability
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict

# Logistic Regression
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Stats
from scipy.stats import ttest_1samp

import os

#%% Make a Prediction for DBScan
# Found function at:
# https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
def dbscan_predict(model, X):

    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1 # start with assumption that all points are noise points

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting 
        
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist) # find index of point with shortest distance to the new sample

        if dist[shortest_dist_idx] < model.eps: # if the shortest distance to the sample is less than epsilon
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new

#%% Make a Prediction for Agglomerative Clustering
    
def ag_predict(model,X1,X2): # X1 is original data used for clustering, X2 is new data
    n_samples_pred = np.shape(X2)[0]
    
    y_new = np.ones(shape=n_samples_pred, dtype=int)
    
    for i in range(n_samples_pred):
        diff = X1 - X2[i,:] # difference between new sample and each of the original points
        
        dist = np.linalg.norm(diff,axis=1) # Euclidean distance
        
        shortest_dist_idx = np.argmin(dist) # find index of original sample with shortest distance from new sample
        
        y_new[i] = model.labels_[shortest_dist_idx]
        
    return(y_new)

#%% My G2PC Feature Importance Function - This function calculates the percent change in the clustering after permutation

# add functionality that transposes X if the 0 dimension isn't the same length as Y

def G2PC(mdl,X,Y,n_repeats,groups,random_state,check_var):
    Pct_Chg = np.zeros((n_repeats,len(np.unique(groups)))) # preallocate output matrix number of repeats x number of features
    if check_var == 1:
        R = np.zeros((np.shape(X)[0],len(np.unique(groups)))) # preallocate matrix of ratio of cluster label changes to number of repeats
        VarData = np.zeros_like(R) # preallocate matrix of variance of permuted values for each sample and group
        MeanData = np.zeros_like(R)
        
    for j in np.unique(groups): # for j feature groups
        if check_var == 1:
            Record = np.zeros((np.shape(X)[0],np.sum(list(groups == j*np.ones_like(groups))),n_repeats)) # N Samples x N Features per Group x N Repeats
        for k in range(n_repeats): # for k repeats
            np.random.seed(seed=k)
            X_2 = np.copy(X); X_2[:] = X[:]; # duplicate data array
            Sub_Data = np.random.permutation(X_2[:,np.squeeze(list(groups == j*np.ones_like(groups)))]) # shuffle feature
            X_2[:,np.squeeze(list(groups == j*np.ones_like(groups)))] = Sub_Data # add shuffled data to data matrix
            if check_var == 1:
                Record[:,:,k] = Sub_Data # keep track of values for each permuted feature
            
            if type(mdl).__name__ == "DBSCAN": # For DBScan
                Y_2 = dbscan_predict(mdl,X_2) 
            elif type(mdl).__name__ == "AgglomerativeClustering": # For Agglomerative Clustering
                Y_2 = ag_predict(mdl,X,X_2)
            elif type(mdl).__name__ == "KMeans": # For K-means
                Y_2 = mdl.predict(X_2) # would be same for mean-shift clustering
            elif type(mdl).__name__ == "GaussianMixture":
                Y_2 = mdl.predict(X_2)
            elif type(mdl).__name__ == 'dict': # for Fuzzy C-Means
                u = cmeans_predict(X_2.T, mdl["cntr"], mdl["m"], error=mdl["error"], maxiter=mdl["maxiter"],seed=random_state)[0]
                Y_2 = np.argmax(u,axis=0)
            else:
                print('This Model-type is not supported.')
            # print(np.sum(np.array(Y)!=np.array(Y_2))/len(np.squeeze(Y)))
            Pct_Chg[k,j] = np.sum(np.array(Y)!=np.array(Y_2))/len(np.squeeze(Y)) # calculate percent change
            if check_var == 1:
                R[:,j] += np.squeeze(np.array(Y)!=np.array(Y_2))/n_repeats
                VarData[:,j] = np.mean(np.var(Record,axis= 2),axis=1)
                MeanData[:,j] = np.mean(np.mean(Record,axis= 2),axis=1)

                
    if check_var == 1:
        return(Pct_Chg,R,VarData,MeanData)
    else:
        return(Pct_Chg)

#%% My L2PC Feature Importance Function - This function calculates the percent change in the clustering of a sample after repeated Perturbation

def L2PC(mdl,X,Y,M, n_repeats,groups,random_state): # M = number of samples per repetition
    Pct_Chg = np.zeros((np.shape(X)[0],n_repeats,len(np.unique(groups)))) # preallocate output matrix number of repeats x number of features
    
    for n in range(np.shape(X)[0]): # for n subjects
        
        for j in np.unique(groups): # for j feature groups
            
            for m in range(M): # for the number of samples per repetition
                if m == 0:
                    X_2 = np.expand_dims(X[n,:],axis=0)
                else:
                    X_2 = np.concatenate((X_2,np.expand_dims(X[n,:],axis=0)),axis=0) # form a matrix of the the sample
            
            for k in range(n_repeats): # for k repeats
                np.random.seed(seed=n*k) # find new way to set this
                perm_idx = np.random.permutation(np.shape(X)[0])
                xval = X[np.squeeze(perm_idx)<M*np.ones((np.shape(X)[0],)),:] # select permuted samples to substitute
                X_2[:,np.squeeze(list(groups == j*np.ones_like(groups)))] =  xval[:,np.squeeze(list(groups == j*np.ones_like(groups)))] # substitute particular features of permuted samples

                if type(mdl).__name__ == "DBSCAN": # For DBScan
                    Y_2 = dbscan_predict(mdl,X_2) 
                elif type(mdl).__name__ == "AgglomerativeClustering": # For Agglomerative Clustering
                    Y_2 = ag_predict(mdl,X,X_2)
                elif type(mdl).__name__ == "KMeans": # For K-means
                    Y_2 = mdl.predict(X_2) # would be same for mean-shift clustering
                elif type(mdl).__name__ == "GaussianMixture":
                    Y_2 = mdl.predict(X_2)
                elif type(mdl).__name__ == 'dict': # for Fuzzy C-Means
                    u = cmeans_predict(X_2.T, mdl["cntr"], mdl["m"], error=mdl["error"], maxiter=mdl["maxiter"],seed=random_state)[0]
                    Y_2 = np.argmax(u,axis=0)
                else:
                    print('This Model-type is not supported.')
                # print(len(np.squeeze(Y_2)))
                # print(np.sum(np.array(Y[n]*np.ones((M,)))!=np.array(Y_2)))
                # if n == 0: print(np.shape(x))

                Pct_Chg[n,k,j] = np.sum(np.array(Y[n]*np.ones((M,)))!=np.array(Y_2))/len(np.squeeze(Y_2)) # calculate percent change
                # print(Pct_Chg[n,k,j])
        # print(Y[n])

    return(Pct_Chg)

#%% Performance Assessment Function
def Performance_Assessment(y_test,y_pred,threshold_opt):
    # Given the positive column of the test labels and predicted scores, 
    # this function computes an interpolated ROC curve (for exactly 100 FPR points between 0 and 1), 
    # the AUC, the Sensitivity, the Specificity, and the Accuracy of a classifier.
    
    AUC = (roc_auc_score(y_test,y_pred,average='weighted'))
    
    y_pred[y_pred>=threshold_opt] = 1
    y_pred[y_pred<threshold_opt] = 0
    ConfMat = np.zeros((2,2))
    for i in np.arange(len(y_test)):
        ConfMat[int(y_pred[i]),int(y_test[i])] += 1
    SPEC = ((ConfMat[0,0]/(ConfMat[0,0] + ConfMat[1,0])))
    SENS = ((ConfMat[1,1]/(ConfMat[1,1] + ConfMat[0,1])))
    return(AUC,SPEC,SENS)

#%% LR-ENR

def LR_ENR(X_z,Y_pred,groups,GroupNames):
    cv_method = StratifiedShuffleSplit(n_splits=10,train_size=0.8,test_size=0.2,random_state=0)
    cv_method2 = StratifiedShuffleSplit(n_splits=10,train_size=0.8,test_size=0.2,random_state=0)
    
    
    group_effect_vals = []; label_act = [];
    AUC = []; SPEC = []; SENS = [];
    
    step = 0
    for train_index, test_index in cv_method.split(X_z,Y_pred):
        # Select data
        x_train = X_z[train_index,:]; y_train = np.squeeze(Y_pred[train_index])
        x_test = X_z[test_index,:]; y_test = np.squeeze(Y_pred[test_index])
        label_act.append(y_test)
        
        # Define parameter values to be considered in analysis.
        l1_ratios = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99]; #np.linspace(0,1.0,25)
        Cs = np.geomspace(10**-4,10**4,num=100)
        
        # Train LR-ENR classifier
        clf = LogisticRegressionCV(penalty='elasticnet',solver='saga',l1_ratios=list(l1_ratios),Cs = Cs,cv=cv_method2,max_iter=200000,random_state=step,n_jobs=8).fit(x_train,y_train)
        
        # Output permutation feature importance
        effect = x_train*clf.coef_ # effect of each feature (coefficent * feature value)
        
        mean_effect = np.mean(np.absolute(effect),axis=0) # mean effect of each feature across all test samples
        
        group_effect = np.zeros((1,len(GroupNames))) # mean of mean effect for each group of features
        for i in np.unique(groups):
            group_effect[:,i-1] = np.mean(mean_effect[np.squeeze(groups==i*np.ones_like(groups))])
        
        group_effect_vals.append(group_effect);
        predicted_labels = clf.predict_proba(x_test)[:,1]
        
        # Compute Classification Performance
        [auc,spec,sens] = Performance_Assessment(y_test, predicted_labels, 0.5)
        
        AUC.append(auc); SPEC.append(spec); SENS.append(sens);
        
    group_effect_vals2 = np.zeros((np.shape(group_effect_vals)[0],np.shape(group_effect_vals[0])[1]))
    for i in range(np.shape(group_effect_vals)[0]):
        group_effect_vals2[i,:] = np.squeeze(group_effect_vals[i])
        
    return(group_effect_vals2,AUC)

#%% Choose Dataset to Analyze

SZ_Only = 0 

# 0 = analyze both the schizophrenia and control groups
# 1 = analyze only the schizophrenia group

#%% Load and Format Data


file_loc = "C:/Users/antho/OneDrive/Documents/Calhoun_Lab/Projects/Clustering_Explainability/JMLR/KDD2021/DataAndResults/FBIRN_formatted.mat"
X = loadmat(file_loc, appendmat=True)['sFNC']
groups = loadmat(file_loc, appendmat=True)['groups2']
Y_real = loadmat(file_loc,appendmat=True)['analysis_SCORE'][:,2]-1
GroupNames = loadmat(file_loc, appendmat=True)['GroupNames2']

# Select Samples
if SZ_Only == 1:
    X = X[Y_real == np.zeros_like(Y_real),:]

# Z-Score Data
# x_mean = np.mean(X,axis=0)
# x_std = np.std(X,axis=0)
# X_z = (X-x_mean)/x_std
X_z = X

#%% K-Means Clustering
  
n_clusters = np.arange(2,16)  
silhouette_vals = np.zeros((len(n_clusters),1))

count = 0
for i in n_clusters:
    kcluster = KMeans(n_clusters=i,n_init = 50, random_state=0).fit(X_z)
    
    Y_pred = np.transpose(kcluster.labels_)
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    

fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals)
opt_cluster = n_clusters[max_idx]

kcluster = KMeans(n_clusters=opt_cluster,n_init = 100, random_state=0).fit(X_z)
Y_pred_kmeans = np.transpose(kcluster.labels_)

# G2PC

# permut_kmeans_G2PC,R,VarData,MeanData = G2PC(kcluster,X_z,Y_pred_kmeans,100, np.squeeze(groups-1),1,1)
permut_kmeans_G2PC = G2PC(kcluster,X_z,Y_pred_kmeans,100, np.squeeze(groups-1),1,0)


fig = plt.figure(figsize=(10,10))
ax = plt.boxplot((permut_kmeans_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - K-Means Results',fontsize=20)

# L2PC

permut_kmeans_L2PC = L2PC(kcluster, X_z, Y_pred_kmeans, 30, 100,np.squeeze(groups-1), 0)

fig = plt.figure(figsize=(10,10))
for i in range(np.shape(permut_kmeans_L2PC)[0]):
    if Y_pred_kmeans[i] == 0:
        c = 'green'
    elif Y_pred_kmeans[i] == 1:
        c = 'blue'
    elif Y_pred_kmeans[i] == 2:
        c = 'red'
    elif Y_pred_kmeans[i] == 3:
        c = 'gray'
    elif Y_pred_kmeans[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_kmeans_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.mean(permut_kmeans_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_kmeans_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(28)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - K-Means Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

ACC_kmeans = np.sum(np.array(np.squeeze(Y_real))!=np.array(Y_pred_kmeans))/len(Y_pred_kmeans)

# Logistic Regression

Mean_Effect_Kmeans,AUC_Kmeans = LR_ENR(X_z,Y_pred_kmeans,groups,GroupNames)

#%% DBScan

np.random.seed(seed=i)

# Determine Optimal Epsilon Parameter Value
ep_vals = np.arange(0.5,20,0.5)
silhouette_vals = np.zeros((len(ep_vals),1)) # set default value of 0 for silhouette
count = 0
for ep in ep_vals:
    clustering = DBSCAN(eps=ep,min_samples=3).fit(X_z) # train DBScan clustering
    Y_pred = np.transpose(clustering.labels_) # obtain predicted labels
    if len(np.unique(clustering.labels_)) != 1: # if not all points are noise points
        silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    count += 1

# Train DBScan Clustering with Best Epsilon Value
best_ep = np.array(ep_vals)[np.argmax(silhouette_vals)] # get epsilon with best silhouette value
clustering = DBSCAN(eps=best_ep,min_samples=3).fit(X_z) # train DBScan clustering 
Y_pred_dbscan = np.transpose(clustering.labels_) # obtain predicted labels

# G2PC
permut_dbscan_G2PC = G2PC(clustering,X_z,Y_pred_dbscan,100, np.squeeze(groups-1),1,0)
        
fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_dbscan_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - DBScan Results',fontsize=20)

# L2PC
permut_dbscan_L2PC = L2PC(clustering,  X_z, Y_pred_dbscan, 30, 100,np.squeeze(groups-1), 0)

fig = plt.figure(figsize=(10,10))
for i in range(np.shape(permut_dbscan_L2PC)[0]):
    if Y_pred_dbscan[i] == 0:
        c = 'green'
    elif Y_pred_dbscan[i] == 1:
        c = 'blue'
    elif Y_pred_dbscan[i] == 2:
        c = 'red'
    elif Y_pred_dbscan[i] == 3:
        c = 'gray'
    elif Y_pred_dbscan[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_dbscan_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.median(permut_dbscan_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_dbscan_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(1,29)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - DBScan Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

ACC_dbscan = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_dbscan))/len(Y_pred_dbscan)

#%% GMM


# Determine Optimal Number of Clusters
n_clusters = np.arange(2,16)  
silhouette_vals = np.zeros((len(n_clusters),1))

count = 0
for i in n_clusters:
    gmm = GaussianMixture(n_components=i, max_iter = 500, random_state=0).fit(X_z)
    
    Y_pred = np.transpose(gmm.predict(X_z))    
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    

fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals)
opt_cluster = n_clusters[max_idx]

gmm = GaussianMixture(n_components=opt_cluster, max_iter = 500, random_state=0).fit(X_z)

Y_pred_gmm = np.transpose(gmm.predict(X_z))

permut_gmm_G2PC = G2PC(gmm,X_z,Y_pred_gmm,100, np.squeeze(groups-1),1,0)

ACC_gmm = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_gmm))/len(Y_pred_gmm)

fig = plt.figure(figsize=(10,10))
ax = plt.boxplot((permut_gmm_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - GMM Results',fontsize=20)

# L2PC
permut_gmm_L2PC = L2PC(gmm,  X_z, Y_pred_gmm, 30, 100,np.squeeze(groups-1), 0)

fig = plt.figure(figsize=(10,10))
for i in range(np.shape(permut_gmm_L2PC)[0]):
    if Y_pred_gmm[i] == 0:
        c = 'green'
    elif Y_pred_gmm[i] == 1:
        c = 'blue'
    elif Y_pred_gmm[i] == 2:
        c = 'red'
    elif Y_pred_gmm[i] == 3:
        c = 'gray'
    elif Y_pred_gmm[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_gmm_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.median(permut_dbscan_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_gmm_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(1,29)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - GMM Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% Agglomerative Clustering

# Determine Optimal Number of Clusters
n_clusters = np.arange(2,16)  
silhouette_vals = np.zeros((len(n_clusters),1))

count = 0
for i in n_clusters:
    
    ag = AgglomerativeClustering(n_clusters=i).fit(X_z)
    
    Y_pred = np.transpose(ag.labels_)
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    

fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals)
opt_cluster = n_clusters[max_idx]

ag = AgglomerativeClustering(n_clusters=opt_cluster).fit(X_z)

Y_pred_ag = np.transpose(ag.labels_)

permut_ag_G2PC = G2PC(ag,X_z,Y_pred_ag,100, np.squeeze(groups-1),1,0)

ACC_ag = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_ag))/len(Y_pred_ag)

fig = plt.figure(figsize=(10,10))
ax = plt.boxplot((permut_ag_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - AGC Results',fontsize=20)

# L2PC
permut_ag_L2PC = L2PC(ag,  X_z, Y_pred_ag, 30, 100,np.squeeze(groups-1), 0)

fig = plt.figure(figsize=(10,10))
for i in range(np.shape(permut_ag_L2PC)[0]):
    if Y_pred_ag[i] == 0:
        c = 'green'
    elif Y_pred_ag[i] == 1:
        c = 'blue'
    elif Y_pred_ag[i] == 2:
        c = 'red'
    elif Y_pred_ag[i] == 3:
        c = 'gray'
    elif Y_pred_ag[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_ag_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.median(permut_dbscan_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_ag_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(1,29)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - AGC Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% C-Means Clustering

# Determine Optimal Number of Clusters
n_clusters = np.arange(2,16)  
silhouette_vals = np.zeros((len(n_clusters),1))

m = 2 # model parameter
error = 0.005 # model parameter
maxiter = 1000 # model parameter

count = 0
for i in n_clusters:
    
    cntr,u = cmeans(np.transpose(X_z),i,m, error=error, maxiter=maxiter,seed=0)[:2]
    Y_pred = np.argmax(u,axis=0)
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    
fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals)
opt_cluster = n_clusters[max_idx]

cntr,u = cmeans(np.transpose(X_z),opt_cluster,m, error=error, maxiter=maxiter,seed=0)[:2]

Y_pred_cmeans = np.argmax(u,axis=0)

cmeans_mdl = {"cntr":cntr,"nclusters":opt_cluster,"m":m,"error":error,"maxiter": maxiter, "Y_pred":Y_pred_cmeans}

permut_cmeans_G2PC = G2PC(cmeans_mdl,X_z,Y_pred_cmeans,100, np.squeeze(groups-1),0,0)

# permut_cmeans_G2PC,R,VarData = G2PC(cmeans_mdl,X_z,Y_pred_kmeans,100, np.squeeze(groups-1),1,1)

ACC_cmeans = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_cmeans))/len(Y_pred_cmeans)

fig = plt.figure(figsize=(10,10))
ax = plt.boxplot((permut_cmeans_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - Fuzzy C-means Results',fontsize=20)

# L2PC
permut_cmeans_L2PC = L2PC(cmeans_mdl,  X_z, Y_pred_cmeans, 30, 100,np.squeeze(groups-1), 0)

fig = plt.figure(figsize=(10,10))
for i in range(np.shape(permut_cmeans_G2PC)[0]):
    if Y_pred_cmeans[i] == 0:
        c = 'green'
    elif Y_pred_cmeans[i] == 1:
        c = 'blue'
    elif Y_pred_cmeans[i] == 2:
        c = 'red'
    elif Y_pred_cmeans[i] == 3:
        c = 'gray'
    elif Y_pred_cmeans[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_cmeans_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.median(permut_dbscan_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_cmeans_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(0,28)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - Fuzzy C-means Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

Mean_Effect_Cmeans,AUC_Cmeans = LR_ENR(X_z,Y_pred_cmeans,groups,GroupNames)

#%% Save Results

output_loc = "C:/Users/antho/OneDrive/Documents/Calhoun_Lab/Projects/Clustering_Explainability/JMLR/KDD2021/DataAndResults/G2PC_L2PC_Results_nz_sFC_all_V2.mat"
output = {"permut_cmeans_G2PC":permut_cmeans_G2PC,"permut_cmeans_L2PC":permut_cmeans_L2PC,"permut_kmeans_G2PC":permut_kmeans_G2PC,"permut_kmeans_L2PC":permut_kmeans_L2PC,"Y_pred_cmeans":Y_pred_cmeans,"Y_pred_kmeans":Y_pred_kmeans,"Y_real":Y_real}
# output = {"permut_kmeans_L2PC":permut_kmeans_L2PC}

savemat(output_loc,output,appendmat=True)

output_loc = "C:/Users/antho/OneDrive/Documents/Calhoun_Lab/Projects/Clustering_Explainability/JMLR/LR_Results_nz_sFC_all_V2.mat"
output = {"Mean_Effect_Cmeans":Mean_Effect_Cmeans,"Mean_Effect_Kmeans":Mean_Effect_Kmeans,"AUC_Cmeans":AUC_Cmeans,"AUC_Kmeans":AUC_Kmeans}

savemat(output_loc,output,appendmat=True)
    
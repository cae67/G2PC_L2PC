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

# Plotting
import matplotlib.pyplot as plt

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

def G2PC(mdl,X,Y,K,groups,random_state): # K = number of repeats
    Pct_Chg = np.zeros((K,len(np.unique(groups)))) # preallocate output matrix K x number of groups
            
    for j in np.unique(groups): # for j feature groups
        
        for k in range(K): # for k in K repeats
            np.random.seed(seed=k)
            X_2 = np.copy(X); X_2[:] = X[:]; # duplicate data array
            Sub_Data = np.random.permutation(X_2[:,np.squeeze(list(groups == j*np.ones_like(groups)))]) # shuffle feature
            X_2[:,np.squeeze(list(groups == j*np.ones_like(groups)))] = Sub_Data # add shuffled data to data matrix
            
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

#%% Choose Dataset to Analyze

SZ_Only = 0 

# 0 = analyze both the schizophrenia and control groups
# 1 = analyze only the schizophrenia group

#%% Load and Format Data


file_loc = "C:/Users/antho/OneDrive/Documents/Calhoun_Lab/Projects/Clustering_Explainability/DataAndResults/FBIRN_formatted.mat"
X = loadmat(file_loc, appendmat=True)['sFNC']
groups = loadmat(file_loc, appendmat=True)['groups']
Y_real = loadmat(file_loc,appendmat=True)['analysis_SCORE'][:,2]-1
GroupNames = loadmat(file_loc, appendmat=True)['GroupNames']

# Select Samples
if SZ_Only == 1:
    X = X[Y_real == np.zeros_like(Y_real),:]

# Z-Score Data
# x_mean = np.mean(X,axis=0)
# x_std = np.std(X,axis=0)
# X_z = (X-x_mean)/x_std
X_z = X

#%% K-Means Clustering

# Optimize Number of Clusters with Silhouette Method
n_clusters = np.arange(2,16)  # number of clusters to try
silhouette_vals = np.zeros((len(n_clusters),1)) # preallocate array for silhouette values

count = 0
for i in n_clusters:
    kcluster = KMeans(n_clusters=i,n_init = 50, random_state=0).fit(X_z)
    
    Y_pred = np.transpose(kcluster.labels_)
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    
# Plot Silhouette Values
fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

# Find Optimal Silhouette Value, its Index, and its corresponding number of clusters
max_idx = np.argmax(silhouette_vals)
opt_cluster = n_clusters[max_idx]

# Redo the clustering with the optimal number of clusters
kcluster = KMeans(n_clusters=opt_cluster,n_init = 100, random_state=0).fit(X_z)
Y_pred_kmeans = np.transpose(kcluster.labels_)

# G2PC
permut_kmeans_G2PC = G2PC(kcluster,X_z,Y_pred_kmeans,100, np.squeeze(groups-1),1)

# Plot G2PC Results
fig = plt.figure(figsize=(10,10))
ax = plt.boxplot((permut_kmeans_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - K-Means Results',fontsize=20)

# L2PC
permut_kmeans_L2PC = L2PC(kcluster, X_z, Y_pred_kmeans, 30, 100,np.squeeze(groups-1), 0)

# Plot L2PC Results
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
plt.plot(np.mean(np.mean(permut_kmeans_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(28)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - K-Means Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

ACC_kmeans = np.sum(np.array(np.squeeze(Y_real))!=np.array(Y_pred_kmeans))/len(Y_pred_kmeans) # Calculate accuracy of clustering, note that if cluster labels don't align correctly, accuracy may be below 50%

#%% DBScan

np.random.seed(seed=i)

# Determine Optimal Epsilon Parameter Value with Silhouette Method
ep_vals = np.arange(0.5,20,0.5) # epsilon values to try
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
permut_dbscan_G2PC = G2PC(clustering,X_z,Y_pred_dbscan,100, np.squeeze(groups-1),1)

# Plot G2PC Results
fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_dbscan_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - DBScan Results',fontsize=20)

# L2PC
permut_dbscan_L2PC = L2PC(clustering,  X_z, Y_pred_dbscan, 30, 100,np.squeeze(groups-1), 0)

# Plot L2PC Results
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
plt.plot(np.mean(np.mean(permut_dbscan_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(1,29)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - DBScan Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

ACC_dbscan = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_dbscan))/len(Y_pred_dbscan) # Calculate accuracy of clustering, note that if cluster labels don't align correctly, accuracy may be below 50%


#%% GMM


# Determine Optimal Number of Clusters
n_clusters = np.arange(2,16)  # number of clustes to try
silhouette_vals = np.zeros((len(n_clusters),1)) # preallocate array for silhouette values

count = 0
for i in n_clusters:
    gmm = GaussianMixture(n_components=i, max_iter = 500, random_state=0).fit(X_z)
    
    Y_pred = np.transpose(gmm.predict(X_z))    
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    
# Plot the Silhouette Values
fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals) # Calculate Index of Max Silhouette Value
opt_cluster = n_clusters[max_idx] # Find Optimal Number of Clusters Using Index

# Retrain the GMM with Optimal Number of Clusters
gmm = GaussianMixture(n_components=opt_cluster, max_iter = 500, random_state=0).fit(X_z)

# Obtain Predicted Labels
Y_pred_gmm = np.transpose(gmm.predict(X_z))

# G2PC
permut_gmm_G2PC = G2PC(gmm,X_z,Y_pred_gmm,100, np.squeeze(groups-1),0)

# Calculate Accuracy of GMM (note that accuracy may be below 50% if cluster labels don't match real class labels)
ACC_gmm = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_gmm))/len(Y_pred_gmm)

# Plot G2PC Results
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
plt.plot(np.mean(np.mean(permut_gmm_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(1,29)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - GMM Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% Agglomerative Clustering

# Determine Optimal Number of Clusters
n_clusters = np.arange(2,16) # number of clusters to try
silhouette_vals = np.zeros((len(n_clusters),1)) # preallocate array for silhouette values

count = 0
for i in n_clusters:
    
    ag = AgglomerativeClustering(n_clusters=i).fit(X_z)
    
    Y_pred = np.transpose(ag.labels_)
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    
# Plot Silhouette Values
fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals) # Find Index of Max Silhouette Value
opt_cluster = n_clusters[max_idx] # Find Optimal Number of Clusters with Index

# Redo Clustering with Optimal Number of Clusters
ag = AgglomerativeClustering(n_clusters=opt_cluster).fit(X_z)

Y_pred_ag = np.transpose(ag.labels_) # Obtain Cluster Labels

# G2PC
permut_ag_G2PC = G2PC(ag,X_z,Y_pred_ag,100, np.squeeze(groups-1),0)

# Calculate Accuracy
ACC_ag = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_ag))/len(Y_pred_ag)

# Plot G2PC Results
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
plt.plot(np.mean(np.mean(permut_ag_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(1,29)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - AGC Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% C-Means Clustering

# Determine Optimal Number of Clusters
n_clusters = np.arange(2,16)  # Number of Clusters to Try
silhouette_vals = np.zeros((len(n_clusters),1)) # Preallocate Array for Silhouette Values

m = 2 # model parameter
error = 0.005 # model parameter
maxiter = 1000 # model parameter

count = 0
for i in n_clusters:
    
    cntr,u = cmeans(np.transpose(X_z),i,m, error=error, maxiter=maxiter,seed=0)[:2]
    Y_pred = np.argmax(u,axis=0)
    
    silhouette_vals[count] = silhouette_score(X_z,Y_pred)
    
    count += 1
    
# Plot Silhouette Values
fig = plt.figure(figsize=(5,5))
plt.plot(n_clusters,silhouette_vals)

max_idx = np.argmax(silhouette_vals) # Find Index of Max Silhouette Value
opt_cluster = n_clusters[max_idx] # Find Optimal Number of Clusters with Index

# Redo Clustering
cntr,u = cmeans(np.transpose(X_z),opt_cluster,m, error=error, maxiter=maxiter,seed=0)[:2]

Y_pred_cmeans = np.argmax(u,axis=0)

# Place Cluster Parameters into Dictionary to be Passed to G2PC and L2PC Functions
cmeans_mdl = {"cntr":cntr,"nclusters":opt_cluster,"m":m,"error":error,"maxiter": maxiter, "Y_pred":Y_pred_cmeans}

# G2PC
permut_cmeans_G2PC = G2PC(cmeans_mdl,X_z,Y_pred_cmeans,100, np.squeeze(groups-1),0)

# Clustering Accuracy
ACC_cmeans = np.sum(np.array(np.squeeze(Y_real))==np.array(Y_pred_cmeans))/len(Y_pred_cmeans)

# Plot G2PC Results
fig = plt.figure(figsize=(10,10))
ax = plt.boxplot((permut_cmeans_G2PC))
plt.xticks(np.arange(1,len(GroupNames)+1),GroupNames,rotation=90,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - Fuzzy C-means Results',fontsize=20)

# L2PC
permut_cmeans_L2PC = L2PC(cmeans_mdl,  X_z, Y_pred_cmeans, 30, 100,np.squeeze(groups-1), 0)

# Plot L2PC Results
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
plt.plot(np.mean(np.mean(permut_cmeans_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = np.arange(0,28)
plt.xticks(x_tick_pts, GroupNames,rotation=90,fontsize=14)
plt.title('L2PC - Fuzzy C-means Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% Save Results

output_loc = "C:/Users/antho/OneDrive/Documents/Calhoun_Lab/Projects/Clustering_Explainability/DataAndResults/G2PC_L2PC_Results_nz_sFC_all.mat"
output = {"permut_cmeans_G2PC":permut_cmeans_G2PC,"permut_cmeans_L2PC":permut_cmeans_L2PC,"permut_kmeans_G2PC":permut_kmeans_G2PC,"permut_kmeans_L2PC":permut_kmeans_L2PC,"Y_pred_cmeans":Y_pred_cmeans,"Y_pred_kmeans":Y_pred_kmeans,"Y_real":Y_real}
# output = {"permut_kmeans_L2PC":permut_kmeans_L2PC}

savemat(output_loc,output,appendmat=True)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Import Libraries

# General
import numpy as np

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

#%% Make Synthetic Data

def Gen_Data(dataset,n_samples_perclass,random_state):
    # np.random.seed(random_state)
    
    if dataset == 1: # Synthetic Dataset 1
        # n_samples_perclass = 50;
        x1 = np.concatenate((np.random.randn(n_samples_perclass,1)+11,np.random.randn(n_samples_perclass,1)+3),axis=0)
        x2 = np.concatenate((np.random.randn(n_samples_perclass,1)+9,np.random.randn(n_samples_perclass,1)+3),axis=0)
        x3 = np.concatenate((np.random.randn(n_samples_perclass,1)+7,np.random.randn(n_samples_perclass,1)+3),axis=0)
        x4 = np.concatenate((np.random.randn(n_samples_perclass,1)+5,np.random.randn(n_samples_perclass,1)+3),axis=0)
        x5 = np.random.randn(2*n_samples_perclass,1)+3
        X = np.concatenate((x1,x2,x3,x4,x5),axis=1)
        Y = np.concatenate((np.zeros((50,1)),np.ones((50,1))),axis=0)    
    elif dataset == 2: # Synthetic Dataset 2
        # n_samples_perclass = 50;
        np.random.seed(random_state)
        x1 = 0.5*np.concatenate((np.random.randn(n_samples_perclass,1)+27,np.random.randn(n_samples_perclass,1)+19,np.random.randn(n_samples_perclass,1)+11,np.random.randn(n_samples_perclass,1)+3),axis=0)
        np.random.seed(random_state*2)
        x2 = 0.5*np.concatenate((np.random.randn(n_samples_perclass,1)+21,np.random.randn(n_samples_perclass,1)+15,np.random.randn(n_samples_perclass,1)+9,np.random.randn(n_samples_perclass,1)+3),axis=0)
        np.random.seed(random_state*3)
        x3 = 0.5*np.concatenate((np.random.randn(n_samples_perclass,1)+15,np.random.randn(n_samples_perclass,1)+11,np.random.randn(n_samples_perclass,1)+7,np.random.randn(n_samples_perclass,1)+3),axis=0)
        np.random.seed(random_state*4)
        x4 = 2*np.concatenate((np.random.randn(n_samples_perclass,1)+9,np.random.randn(n_samples_perclass,1)+7,np.random.randn(n_samples_perclass,1)+5,np.random.randn(n_samples_perclass,1)+3),axis=0)
        np.random.seed(random_state*5)
        x5 = 2*np.concatenate((np.random.randn(n_samples_perclass,1)+6,np.random.randn(n_samples_perclass,1)+5,np.random.randn(n_samples_perclass,1)+4,np.random.randn(n_samples_perclass,1)+3),axis=0)
        X = np.concatenate((x1,x2,x3,x4,x5),axis=1)
        Y = np.concatenate((3*np.ones((n_samples_perclass,1)),2*np.ones((n_samples_perclass,1)),np.ones((n_samples_perclass,1)),np.zeros((n_samples_perclass,1))),axis=0)
    return(X,Y)

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

def G2PC(mdl,X,Y,n_repeats,random_state):
    Pct_Chg = np.zeros((n_repeats,np.shape(X)[1])) # preallocate output matrix number of repeats x number of features
    for j in range(np.shape(X)[1]): # for j features
        for k in range(n_repeats): # for k repeats
            np.random.seed(seed=k)
            X_2 = np.copy(X); X_2[:] = X[:]; # duplicate data array
            X_2[:,j] = np.random.permutation(X_2[:,j]) # shuffle feature
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
            Pct_Chg[k,j] = np.sum(np.array(Y)!=np.array(Y_2))/len(np.squeeze(Y)) # calculate percent change
    return(Pct_Chg)

#%% Set Run Parameters
dataset = 1 # choose either dataset 1 or dataset 2
n_clusters = 4
n_samples_perclass = 50

n_repeats = 100


#%% G2PC with K-Means Clustering
ACC_kmeans = []
Y_all = []
X_all = []
n_iter = 100
# n_repeats = 100
permut_kmeans = np.zeros((n_iter*n_repeats,5))


for i in range(n_iter):
    np.random.seed(seed=i)
    X,Y = Gen_Data(dataset,n_clusters,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    kcluster = KMeans(n_clusters=n_clusters,n_init = 50, random_state=i).fit(X_z)

    Y_pred = np.transpose(kcluster.labels_)
    
    permut_kmeans[i*n_repeats:(i+1)*n_repeats,:] = (G2PC(kcluster,X_z,Y_pred,n_repeats,1))
        
    ACC_kmeans.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters
    

fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_kmeans))
#plt.xticks(np.arange(1,len(FeatureNames)+1),FeatureNames,rotation=0,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - K-Means Results',fontsize=20)

# p_kmeans = ttest_1samp(permut_kmeans,0,axis=0)[1]


#%% G2PC with DBScan

ACC_dbscan = []
Y_all = []
X_all = []
n_iter = 100
# n_repeats = 100
permut_dbscan = np.zeros((int(n_iter*n_repeats),5)) 


for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std
    
    # Determine Optimal Epsilon Parameter Value
    ep_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    silhouette_vals = np.zeros((len(ep_vals),1)) # set default value of 0 for silhouette
    count = 0
    for ep in ep_vals:
        clustering = DBSCAN(eps=ep,min_samples=4).fit(X_z) # train DBScan clustering
        Y_pred = np.transpose(clustering.labels_) # obtain predicted labels
        if len(np.unique(clustering.labels_)) != 1: # if not all points are noise points
            silhouette_vals[count] = silhouette_score(X_z,Y_pred)
        count += 1
    
    # Train DBScan Clustering with Best Epsilon Value
    best_ep = np.array(ep_vals)[np.argmax(silhouette_vals)] # get epsilon with best silhouette value
    clustering = DBSCAN(eps=best_ep,min_samples=4).fit(X_z) # train DBScan clustering 
    Y_pred = np.transpose(clustering.labels_) # obtain predicted labels

    permut_dbscan[i*n_repeats:(i+1)*n_repeats,:] = (G2PC(clustering,X_z,Y_pred,n_repeats,1))
    
    ACC_dbscan.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters
    
fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_dbscan))
#plt.xticks(np.arange(1,len(FeatureNames)+1),FeatureNames,rotation=0,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - DBScan Results',fontsize=20)

# p_dbscan = ttest_1samp(permut_dbscan,0,axis=0)[1]

#%% G2PC with GMM

ACC_gmm = []
Y_all = []
X_all = []
n_iter = 100
# n_repeats = 100
permut_gmm = np.zeros((int(n_iter*n_repeats),5)) 

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    gmm = GaussianMixture(n_components=n_clusters, max_iter = 500, random_state=i).fit(X_z)

    Y_pred = np.transpose(gmm.predict(X_z))

    permut_gmm[i*n_repeats:(i+1)*n_repeats,:] = G2PC(gmm,X_z,Y_pred,n_repeats,1)
    
    if np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred) == 0:
        ACC_gmm.append(1.0)
    else:
        ACC_gmm.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters

fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_gmm))
#plt.xticks(np.arange(1,len(FeatureNames)+1),FeatureNames,rotation=0,fontsize=14)#
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - GMM Results',fontsize=20)

# plot_partial_dependence(gmm, X_z, [0], grid_resolution=20)

# p_gmm = ttest_1samp(permut_gmm,0,axis=0)[1]

#%% G2PC with Agglomerative Clustering

ACC_ag = []
Y_all = []
X_all = []
n_iter = 100
# n_repeats = 100
permut_ag = np.zeros((int(n_iter*n_repeats),5)) 

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    ag = AgglomerativeClustering(n_clusters=n_clusters).fit(X_z)

    Y_pred = np.transpose(ag.labels_)

    permut_ag[i*n_repeats:(i+1)*n_repeats,:] = G2PC(ag,X_z,Y_pred,n_repeats,1)
    
    if np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred) == 0:
        ACC_ag.append(1.0)
    else:
        ACC_ag.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters

fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_ag))
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - AGC Results',fontsize=20)

# p_ag = ttest_1samp(permut_ag,0,axis=0)[1]

#%% G2PC with Fuzzy C-Means

ACC_cmeans = []
Y_all = []
X_all = []
n_iter = 100
# n_repeats = 100
n_clusters = n_clusters # model parameter
m = 2 # model parameter
error = 0.005 # model parameter
maxiter = 1000 # model parameter
permut_cmeans = np.zeros((int(n_iter*n_repeats),5)) 

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    cntr,u = cmeans(np.transpose(X_z),n_clusters,m, error=error, maxiter=maxiter,seed=i)[:2]
    Y_pred = np.argmax(u,axis=0)
    cmeans_mdl = {"cntr":cntr,"nclusters":n_clusters,"m":m,"error":error,"maxiter": maxiter, "Y_pred":Y_pred}

    permut_cmeans[i*n_repeats:(i+1)*n_repeats,:] = G2PC(cmeans_mdl,X_z,Y_pred,n_repeats,i)
    
    if np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred) == 0:
        ACC_cmeans.append(1.0)
    else:
        ACC_cmeans.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters

fig = plt.figure(figsize=(5,5))
ax = plt.boxplot((permut_cmeans))
ylabel = plt.ylabel('Permutation Percent Change',fontsize=16)
title = plt.title('G2PC - Fuzzy C-Means Results',fontsize=20)

# p_ag = ttest_1samp(permut_ag,0,axis=0)[1]


#%% My L2PC Feature Importance Function - This function calculates the percent change in the clustering of a sample after repeated Perturbation

def L2PC(mdl,X,Y,M, n_repeats,random_state): # M = number of samples per repetition
    Pct_Chg = np.zeros((np.shape(X)[0],n_repeats,np.shape(X)[1])) # preallocate output matrix number of repeats x number of features
    
    for n in range(np.shape(X)[0]): # for n subjects
        
        for j in range(np.shape(X)[1]): # for j features
            
            for m in range(M): # for the number of samples per repetition
                if m == 0:
                    X_2 = np.expand_dims(X[n,:],axis=0)
                else:
                    X_2 = np.concatenate((X_2,np.expand_dims(X[n,:],axis=0)),axis=0) # form a matrix of the the sample
            
            for k in range(n_repeats): # for k repeats
                np.random.seed(seed=n*k) # find new way to set this
                perm_idx = np.random.permutation(np.shape(X)[0])
                
                X_2[:,j] =  X[np.squeeze(perm_idx)<M*np.ones((np.shape(X)[0],)),j]
                
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

                Pct_Chg[n,k,j] = np.sum(np.array(Y[n]*np.ones((M,)))!=np.array(Y_2))/len(np.squeeze(Y_2)) # calculate percent change
                
    return(Pct_Chg)


#%% L2PC with K-Means

ACC_kmeans_L2PC = []
Y_all = []
X_all = []
n_iter = 1
n_repeats = 100
permut_kmeans_L2PC = np.zeros((int(n_samples_perclass*n_clusters),int(n_iter*n_repeats),5))

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    kcluster = KMeans(n_clusters=n_clusters,n_init = 50, random_state=i).fit(X_z)

    Y_pred = np.transpose(kcluster.labels_)
    
    permut_kmeans_L2PC[:,i*n_repeats:(i+1)*n_repeats,:] = L2PC(kcluster, X_z, Y_pred, 30, n_repeats, i)
        
    ACC_kmeans_L2PC.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters
    

fig = plt.figure(figsize=(5,5))
for i in range(np.shape(permut_kmeans_L2PC)[0]):
    if Y_pred[i] == 0:
        c = 'green'
    elif Y_pred[i] == 1:
        c = 'blue'
    elif Y_pred[i] == 2:
        c = 'red'
    elif Y_pred[i] == 3:
        c = 'gray'
    elif Y_pred[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_kmeans_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.mean(permut_kmeans_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_kmeans_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = [0,1,2,3,4]
my_xticks = ['1', '2', '3', '4', '5']
plt.xticks(x_tick_pts, my_xticks)
plt.title('L2PC - K-Means Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)


#%% L2PC with DBScan

ACC_dbscan = []
Y_all = []
X_all = []
n_iter = 1
n_repeats = 100
permut_dbscan_L2PC = np.zeros((int(n_samples_perclass*n_clusters),int(n_iter*n_repeats),5)) 


for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std
    
    # Determine Optimal Epsilon Parameter Value
    ep_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    silhouette_vals = np.zeros((len(ep_vals),1)) # set default value of 0 for silhouette
    count = 0
    for ep in ep_vals:
        clustering = DBSCAN(eps=ep,min_samples=4).fit(X_z) # train DBScan clustering
        Y_pred = np.transpose(clustering.labels_) # obtain predicted labels
        if len(np.unique(clustering.labels_)) != 1: # if not all points are noise points
            silhouette_vals[count] = silhouette_score(X_z,Y_pred)
        count += 1

    # Train DBScan Clustering with Best Epsilon Value
    best_ep = np.array(ep_vals)[np.argmax(silhouette_vals)] # get epsilon with best silhouette value
    clustering = DBSCAN(eps=best_ep,min_samples=4).fit(X_z) # train DBScan clustering 
    Y_pred = np.transpose(clustering.labels_) # obtain predicted labels

    permut_dbscan_L2PC[:,i*n_repeats:(i+1)*n_repeats,:] = (L2PC(clustering, X_z, Y_pred, 30, n_repeats, i))
    
    ACC_dbscan.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred)) # this accuracy calculation will only work for 2 clusters
   
fig = plt.figure(figsize=(5,5))
for i in range(np.shape(permut_dbscan_L2PC)[0]):
    if Y_pred[i] == 0:
        c = 'green'
    elif Y_pred[i] == 1:
        c = 'blue'
    elif Y_pred[i] == 2:
        c = 'red'
    elif Y_pred[i] == 3:
        c = 'gray'
    elif Y_pred[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_dbscan_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.median(permut_dbscan_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_dbscan_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
x_tick_pts = [0,1,2,3,4]
my_xticks = ['1', '2', '3', '4', '5']
plt.xticks(x_tick_pts, my_xticks)
plt.title('L2PC - DBScan Results',fontsize=20)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% L2PC with GMM

ACC_gmm_L2PC = []
Y_all = []
X_all = []
n_iter = 1
n_repeats = 100
permut_gmm_L2PC = np.zeros((int(n_samples_perclass*n_clusters),int(n_iter*n_repeats),5)) 

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    gmm = GaussianMixture(n_components=n_clusters, max_iter = 500, random_state=i).fit(X_z)

    Y_pred = np.transpose(gmm.predict(X_z))

    permut_gmm_L2PC[:,i*n_repeats:(i+1)*n_repeats,:] = L2PC(gmm, X_z, Y_pred, 30, n_repeats, i)
    
    if np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred) == 0: # this accuracy calculation will only work for 2 clusters
        ACC_gmm_L2PC.append(1.0)
    else:
        ACC_gmm_L2PC.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred))

fig = plt.figure(figsize=(5,5))
for i in range(np.shape(permut_gmm_L2PC)[0]):
    if Y_pred[i] == 0:
        c = 'green'
    elif Y_pred[i] == 1:
        c = 'blue'
    elif Y_pred[i] == 2:
        c = 'red'
    elif Y_pred[i] == 3:
        c = 'gray'
    elif Y_pred[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_gmm_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.mean(permut_gmm_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_gmm_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
plt.title('L2PC - GMM Results',fontsize=20)
x_tick_pts = [0,1,2,3,4]
my_xticks = ['1', '2', '3', '4', '5']
plt.xticks(x_tick_pts, my_xticks)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% L2PC with Agglomerative Clustering

ACC_ag_L2PC = []
Y_all = []
X_all = []
n_iter = 1
n_repeats = 100
permut_ag_L2PC = np.zeros((int(n_samples_perclass*n_clusters),int(n_iter*n_repeats),5)) 

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    ag = AgglomerativeClustering(n_clusters=n_clusters).fit(X_z)

    Y_pred = np.transpose(ag.labels_)

    permut_ag_L2PC[:,i*n_repeats:(i+1)*n_repeats,:] = L2PC(ag, X_z, Y_pred, 30, n_repeats, i)
    
    if np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred) == 0: # this accuracy calculation will only work for 2 clusters
        ACC_ag_L2PC.append(1.0)
    else:
        ACC_ag_L2PC.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred))

fig = plt.figure(figsize=(5,5))
for i in range(np.shape(permut_ag_L2PC)[0]):
    if Y_pred[i] == 0:
        c = 'green'
    elif Y_pred[i] == 1:
        c = 'blue'
    elif Y_pred[i] == 2:
        c = 'red'
    elif Y_pred[i] == 3:
        c = 'gray'
    elif Y_pred[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_ag_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.mean(permut_ag_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_ag_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
plt.title('L2PC - AGC Results',fontsize=20)
x_tick_pts = [0,1,2,3,4]
my_xticks = ['1', '2', '3', '4', '5']
plt.xticks(x_tick_pts, my_xticks)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)

#%% L2PC with Fuzzy C-Means Clustering

ACC_cmeans_L2PC = []
Y_all = []
X_all = []
n_iter = 1
n_repeats = 100
permut_cmeans_L2PC = np.zeros((int(n_samples_perclass*n_clusters),int(n_iter*n_repeats),5)) 

for i in range(n_iter):
    np.random.seed(seed=i)
    
    X,Y = Gen_Data(dataset,n_samples_perclass,i)
    
    X_all.append(X)
    Y_all.append(Y)
    
    x_mean = np.mean(X,axis=0)
    x_std = np.std(X,axis=0)
    X_z = (X-x_mean)/x_std

    cntr,u = cmeans(np.transpose(X_z),n_clusters,m, error=error, maxiter=maxiter,seed=i)[:2]
    Y_pred = np.argmax(u,axis=0)
    cmeans_mdl = {"cntr":cntr,"nclusters":n_clusters,"m":m,"error":error,"maxiter": maxiter, "Y_pred":Y_pred}

    permut_cmeans_L2PC[:,i*n_repeats:(i+1)*n_repeats,:] = L2PC(cmeans_mdl, X_z, Y_pred, 30, n_repeats, i)
    
    if np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred) == 0: # this accuracy calculation will only work for 2 clusters
        ACC_cmeans_L2PC.append(1.0)
    else:
        ACC_cmeans_L2PC.append(np.sum(np.array(np.squeeze(Y))==np.array(Y_pred))/len(Y_pred))

fig = plt.figure(figsize=(5,5))
for i in range(np.shape(permut_cmeans_L2PC)[0]):
    if Y_pred[i] == 0:
        c = 'green'
    elif Y_pred[i] == 1:
        c = 'blue'
    elif Y_pred[i] == 2:
        c = 'red'
    elif Y_pred[i] == 3:
        c = 'gray'
    elif Y_pred[i] == 4:
        c = 'yellow'
    plt.plot(np.mean(permut_cmeans_L2PC[i,:,:],axis=0),marker='o',color=c,alpha=0.3)
# plt.plot(np.mean(permut_cmeans_L2PC,axis=(0,1)),marker='o',color='black',linewidth=2,markersize=12)
plt.plot(np.mean(np.mean(permut_cmeans_L2PC,axis=(1)),axis=0),marker='o',color='black',linewidth=2,markersize=12)
plt.title('L2PC - Fuzzy C-Means Results',fontsize=20)
x_tick_pts = [0,1,2,3,4]
my_xticks = ['1', '2', '3', '4', '5']
plt.xticks(x_tick_pts, my_xticks)
ylabel = plt.ylabel('Perturbation Percent Change',fontsize=16)
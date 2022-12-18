#%% 設定工作目錄

import os
wkDir = "F:/Machine Learning/homework1/";   os.chdir(wkDir) #設定工作目錄

#%%Read Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv('data/data.csv')
colnames = list(data.columns[:])
data.head()

# %%Define k to initiate the centroids

def initiate_centroids(k, dset):
    centroids = dset.sample(k)
    return centroids

np.random.seed(42)
k = 4
df = data[['column1','column2','column3','column4']]
centroids = initiate_centroids(k, df)
centroids

# %%Calculate distance between centroids and data points

def rsserr(a,b):
    return np.sqrt(np.sum((a-b)**2))

for i, centroid in enumerate(range(centroids.shape[0])):
    err = rsserr(centroids.iloc[centroid,:], df.iloc[36,:])
    print('Error for centroid {0}: {1:.2f}'.format(i, err))

# %%Assign data to centroids

def centroid_assignation(dset, centroids):
    k = centroids.shape[0]
    n = dset.shape[0]
    assignation = []
    assign_errors = []

    for obs in range(n):
        #Estimate error
        all_errors = np.array([])
        for centroid in range(k):
            err = rsserr(centroids.iloc[centroid, :], dset.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors

df['centroid'], df['error'] = centroid_assignation(df, centroids)
df.head()

# %% kmeans

def kmeans(dset, k, tol=1e-4):

    working_dset = dset.copy()

    err = []
    goahead = True
    j = 0
    
    # Step 2: Initiate clusters by defining centroids 
    centroids = initiate_centroids(k, dset)

    while(goahead):
        # Step 3 and 4 - Assign centroids and calculate error
        working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids) 
        err.append(sum(j_err))
        
        # Step 5 - Update centroid position
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)

        # Step 6 - Restart the iteration
        if j>0:
            # Is the error less than a tolerance (1E-4)
            if err[j-1]-err[j]<=tol:
                goahead = False
        j+=1

    working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
    return working_dset['centroid'], j_err, centroids

np.random.seed(42)
df['centroid'], df['error'], centroids =  kmeans(df[['column1','column2','column3','column4']], 4)
df.head()

centroids


# %%elbow to find best k
err_total = []
n = 10

for i in range(n):
    label, my_errs, centroid = kmeans(df[['column1','column2','column3','column4']], i+1)
    err_total.append(sum(my_errs))
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1,n+1), err_total, linewidth=3, marker='o')
ax.set_xlabel(r'Number of clusters', fontsize=14)
ax.set_ylabel(r'Total error', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# %%

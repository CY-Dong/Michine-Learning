#%% 設定工作目錄
import os
wkDir = "D:/Machine Learning/homework1/";   os.chdir(wkDir) #設定工作目錄

#%%載入所需軟件包
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#%%匯入DATA
data = pd.read_csv( 'data.csv', encoding = "big5" )


#%%求分群數
silhouette_avg = []#側影
distortions = []#手肘
for i in range(2,11):
    KM_fit = KMeans(n_clusters = i).fit(data)
    silhouette_avg.append(silhouette_score(data, KM_fit.labels_))
    distortions.append(KM_fit.inertia_)
plt.plot(range(2,11), silhouette_avg)
plt.plot(range(2, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

#%%K-means
KM = KMeans(n_clusters=4).fit(data)
#KM.fit(data)
KM.predict(data)

#中心點座標
centers = KM.cluster_centers_

#%%視覺化
plt.subplot()
#plt.title(f'KMeans={selected_K} groups')
plt.scatter(data.column1, data.column2, data.column3, c=KM.predict(data), cmap=plt.cm.Set3)
plt.scatter(centers.T[0], centers.T[1], marker='^', color='orange')
for i in range(centers.shape[0]): # 標上各分組中心點
    plt.text(centers.T[0][i], centers.T[1][i], str(i + 1),
             fontdict={'color': 'red', 'weight': 'bold', 'size': 24})
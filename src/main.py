"""
Unsupervised Learning with PCA & K-Means Clustering — TPS Jul 2022
An unsupervised machine learning project leveraging Principal Component Analysis for dimensional compression and K-Means/GMM algorithms for cluster discovery in competitive tabular data.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/tps-pca-k-means-simplified-4-everyone
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Cell 2 ---
df_og = pd.read_csv("../input/tabular-playground-series-jul-2022/data.csv")
df = df_og.copy()
df.head()

# --- Cell 3 ---
df.drop('id',axis=1,inplace=True)

# --- Cell 4 ---
df.info()

# --- Cell 5 ---
df.describe()

# --- Cell 6 ---
df.isnull().sum()

# --- Cell 8 ---
sns.heatmap(df.corr());

# --- Cell 9 ---
#lets scale the data for further use.
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# --- Cell 10 ---
pca = PCA(0.95) 

# --- Cell 11 ---
df_pca = pca.fit_transform(df_scaled)

# --- Cell 12 ---
df_pca.shape

# --- Cell 13 ---
df_pca = pd.DataFrame(df_pca)
df_pca.head()

# --- Cell 14 ---
sns.heatmap(df_pca.corr());

# --- Cell 15 ---
from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(df_pca)     
visualizer.show()        

# --- Cell 16 ---
km = KMeans(n_clusters=7)
y = km.fit_predict(df_pca)

# --- Cell 17 ---
y

# --- Cell 18 ---
df_og['Predicted']=y

# --- Cell 19 ---
submission=df_og.loc[:,['id','Predicted']]
submission.shape

# --- Cell 20 ---
submission.head()

# --- Cell 21 ---
submission.to_csv('submission.csv',index=False)

# --- Cell 22 ---
### Using Gausian mixture with same number of clusters
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 7)
 
# Fit the GMM model for the dataset
gmm.fit(df_pca)
 
# Assign a label to each sample
labels = gmm.predict(df_pca)
df['Clusters']= labels

# --- Cell 23 ---
df.head()

# --- Cell 24 ---
submission['id']=df.index
submission['Predicted']=df.loc[:,['Clusters']]
submission.shape

# --- Cell 25 ---
submission.head()

# --- Cell 26 ---
submission.to_csv('submission.csv',index=False)



if __name__ == "__main__":
    print("Pipeline execution complete.")

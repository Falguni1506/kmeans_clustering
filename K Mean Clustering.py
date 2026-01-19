import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\DELL\Downloads\Simple RFM Customer Segmentation - K Means Practice.csv")
#Here, use the path where your file is uploaded.

df.shape


df.head()


df.info()


df.describe()


plt.scatter(df.monetary_purchase_amt,df["frequency_purchase"])
plt.xlabel("monetary_purchase_amt")
plt.ylabel("frequency_purchase")


from sklearn.preprocessing import MinMaxScaler #scaling the datapoints in order to avoid mis-clustering since the distance between datapoints is high 

X = df[['monetary_purchase_amt', 'frequency_purchase']]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42) 
df['Cluster'] = kmeans.fit_predict(X_scaled)
#as we can see Elbow plotting is not required here since the data is clearly dense at 3 different points, suggesting for 3 clusters.


X_scaled.min(axis=0)
X_scaled.max(axis=0)


Km = KMeans(n_clusters=3)
y_predicted = Km.fit_predict(df[["monetary_purchase_amt","frequency_purchase"]])
y_predicted


df['cluster'] = y_predicted
df.head()


df = df.drop(columns=['cluster']) #dropping the additional Cluster column since I ran the code twice


df.head()


Km.cluster_centers_


df1=df[df.Cluster==0]
df2=df[df.Cluster==1]
df3=df[df.Cluster==2]
plt.scatter(df1.monetary_purchase_amt,df1["frequency_purchase"],color='green')
plt.scatter(df2.monetary_purchase_amt,df2["frequency_purchase"],color='red')
plt.scatter(df3.monetary_purchase_amt,df3["frequency_purchase"],color='blue')
plt.scatter(Km.cluster_centers_[:,0],Km.cluster_centers_[:,1],color="purple",marker="*",label="centroid")
plt.xlabel("monetary_purchase_amt")
plt.ylabel("frequency_purchase")
plt.legend()





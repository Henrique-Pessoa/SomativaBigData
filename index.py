import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.ensemble import IsolationForest


data = pd.read_csv("./dadosacoes.csv")

print(data.head())
print(data.describe())
print(data.isna())
print(data.isnull())

sns.boxplot(data=data,  x="nome ação",y="valor de mercado R$ -(Bilhões)")

plt.show()

newData = pd.get_dummies(data,columns=['nome ação'],drop_first=True)


kmeans = KMeans(n_clusters=4) 
kmeans.fit(newData) 

sse = kmeans.inertia_

labels = kmeans.labels_ 
centroids = kmeans.cluster_centers_

plt.scatter(data["valor de mercado R$ -(Bilhões)"],data["qtde cotas"],data['preço ação R$'],c=labels) 
plt.scatter(centroids[:, 4], centroids[:, 4], marker='x', color='red') 
plt.title(label="cluster = 4")
plt.show()

kmeans = KMeans(n_clusters=5) 
kmeans.fit(newData) 

sse = kmeans.inertia_

labels = kmeans.labels_ 
centroids = kmeans.cluster_centers_

plt.scatter(data["valor de mercado R$ -(Bilhões)"],data["qtde cotas"],data['preço ação R$'],c=labels) 
plt.scatter(centroids[:, 4], centroids[:, 4], marker='x', color='red') 
plt.title(label="cluster = 5")
plt.show()

kmeans = KMeans(n_clusters=8) 
kmeans.fit(newData) 

sse = kmeans.inertia_

labels = kmeans.labels_ 
centroids = kmeans.cluster_centers_

plt.scatter(data["valor de mercado R$ -(Bilhões)"],data["qtde cotas"],data['preço ação R$'],c=labels) 
plt.scatter(centroids[:, 4], centroids[:, 4], marker='x', color='red') 
plt.title(label="cluster = 8")
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(data['preço ação R$'], data['qtde cotas'], data['valor de mercado R$ -(Bilhões)'], c=labels)


plt.show()

print(">>>>>>>>>>>>>>>>>>>>>>>Resposta 5<<<<<<<<<<<<<<<<<<<<<<<")

print("Dentre varias caracteristicas diferentes eu acredito que o aprendizado não supervisionado, ajuda quando os dados estão bem espalhados e não estão bem definidos,a maior vantagem do aprendizado não supervisioado eu acredito que seja a facilidade de encontrar padrões")
print("#############################################################################")

price = np.array(data["preço ação R$"])
value = np.array(data["valor de mercado R$ -(Bilhões)"])
qtdeCotas = np.array(data["qtde cotas"])

data = np.column_stack((price,value,qtdeCotas))

iforest = IsolationForest(n_estimators = 100, contamination = 0.03, max_samples ='auto')
prediction = iforest.fit_predict(data)
print(prediction[:20])
print("Number of outliers detected: {}".format(prediction[prediction < 0].sum()))
print("Number of normal samples detected: {}".format(prediction[prediction > 0].sum()))

normal_data = data[np.where(prediction > 0)]
outliers = data[np.where(prediction < 0)]
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(normal_data[:, 0], normal_data[:, 1])
ax.scatter(outliers[:, 0], outliers[:, 1])
plt.show()
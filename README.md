# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and print head,info of the dataset
2. check for null values
3. Import kmeans and fit it to the dataset
4. Plot the graph using elbow method
5. Print the predicted array
6. Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: YAMUNA M
RegisterNumber: 212223230248


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("/content/Mall_Customers.csv")

# Basic checks
print(data.head())
print(data.info())
print(data.isnull().sum())

# Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])  # Use only 'Annual Income' and 'Spending Score'
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply KMeans with optimal clusters
km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["cluster"] = y_pred

# Separate data by clusters
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Visualize clusters
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

plt.legend()
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

*/
```

## Output:

## DATA.HEAD():

![image](https://github.com/user-attachments/assets/6beaa101-b007-4603-abfe-3a688a562bdd)


## DATA.INF0():

![image](https://github.com/user-attachments/assets/63b5452c-dc83-4e5e-a37b-30fce5df0de1)



## DATA.ISNULL().SUM():

![image](https://github.com/user-attachments/assets/cd972c66-a02d-4f55-b94b-d3dfeb699fa4)



## PLOT USING ELBOW METHOD:

![image](https://github.com/user-attachments/assets/7f673ca6-c624-4e22-ab47-356c1892170d)



## CUSTOMER SEGMENT:

![image](https://github.com/user-attachments/assets/1ae6527e-8855-4263-a679-e8aaca340b5f)




## Result:

Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.









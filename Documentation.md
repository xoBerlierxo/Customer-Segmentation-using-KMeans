### K-Means Clustering And its usage on a customer segemnetation database

**How it Works**:

1. Select the number of clusters `k`.
2. Initialize `k` centroids randomly.
3. Assign each data point to the nearest centroid.
4. Recompute the centroids as the mean of all data points in a cluster.
5. Repeat steps 3–4 until convergence (centroids no longer change).

**Objective Function**:
Minimize the sum of squared distances between points and their corresponding cluster centroids.

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

**Advantages**:

* Simple to implement and interpret
* Fast for small to medium-sized datasets

**Disadvantages**:

* Requires specifying `k` in advance
* Sensitive to initial centroid placement
* Struggles with clusters of non-spherical shapes or varying densities

**Typical Applications**:

* Customer segmentation
* Market research
* Image compression

**Example in Python (Mall Customer Dataset)**:

```python
# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
customerdf = pd.read_csv("Mall_Customers.csv")

# Step 3: Visualize relationships
plt.plot(customerdf['Annual Income (k$)'], customerdf['Spending Score (1-100)'])
plt.title("Annual Income vs Spending Score (Line Plot)")
plt.show()

sns.scatterplot(
    data=customerdf,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    style='Gender'
)
plt.title("Spending Score by Gender")
plt.show()

# Step 4: Encode Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
customerdf['Gender'] = le.fit_transform(customerdf['Gender'])

# Step 5: Apply KMeans
from sklearn.cluster import KMeans
X = customerdf[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 6: Visualize clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X')
plt.title("K-Means Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.show()
```

**Elbow Method for Optimal k**:

```python
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
```

---

### 2. Hierarchical Clustering

...

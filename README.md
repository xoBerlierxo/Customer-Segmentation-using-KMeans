# **Customer-Segmentation-using-KMeans**

## 📌 What is Clustering?

Clustering is an unsupervised machine learning technique that groups a set of objects in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups. It is commonly used for exploratory data analysis, customer segmentation, and anomaly detection.

---

## 🔹 Types of Clustering Algorithms

### 1. K-Means Clustering

**Definition**:
K-Means is a partition-based clustering algorithm that divides the data into `k` non-overlapping subsets (clusters).

**How it Works**:

1. Choose the number of clusters `k`.
2. Initialize `k` centroids randomly.
3. Assign each data point to the nearest centroid (using Euclidean distance).
4. Compute new centroids as the mean of all points assigned to that cluster.
5. Repeat steps 3–4 until convergence (centroids no longer change significantly).

**Mathematical Objective**:
Minimize the within-cluster sum of squares (WCSS):

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$

**Pros**:

* Simple and fast
* Works well with spherical clusters

**Cons**:

* Need to specify `k`
* Sensitive to initial centroid positions
* Poor performance on non-convex shapes or varying densities

**Use Cases**:

* Customer segmentation
* Market basket analysis
* Image compression

---

### 2. Hierarchical Clustering

**Definition**:
Builds a tree of clusters (dendrogram). Can be:

* Agglomerative: Bottom-up approach
* Divisive: Top-down approach

**How it Works (Agglomerative)**:

1. Treat each data point as a single cluster.
2. Merge the two closest clusters.
3. Repeat until only one cluster remains.

**Linkage Criteria**:

* Single linkage (minimum distance)
* Complete linkage (maximum distance)
* Average linkage

**Pros**:

* No need to specify number of clusters initially
* Dendrogram provides deep insight into data

**Cons**:

* Computationally expensive (O(n^3))
* Not scalable for large datasets

**Use Cases**:

* Gene expression analysis
* Taxonomy

---

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Definition**:
Clusters data based on density. Capable of discovering clusters of arbitrary shape and identifying noise (outliers).

**Key Parameters**:

* `eps`: Maximum radius of the neighborhood
* `min_samples`: Minimum number of points required to form a dense region

**Types of Points**:

* Core point
* Border point
* Noise point

**Pros**:

* Does not require `k`
* Can find arbitrarily shaped clusters
* Robust to outliers

**Cons**:

* Choosing `eps` and `min_samples` can be tricky
* Struggles with varying densities

**Use Cases**:

* Spatial data
* Anomaly detection

---

### 4. Mean Shift Clustering

**Definition**:
A centroid-based algorithm that updates centroids to the mean of the points within a sliding window.

**How it Works**:

1. Place a window on a data point
2. Calculate the mean of points in the window
3. Shift the window to the mean
4. Repeat until convergence
5. Merge overlapping windows as clusters

**Pros**:

* No need to specify number of clusters
* Can find clusters of any shape

**Cons**:

* Computationally intensive
* Bandwidth selection is critical

**Use Cases**:

* Image segmentation
* Object tracking

---

### 5. Gaussian Mixture Models (GMM)

**Definition**:
Probabilistic model assuming data is generated from a mixture of several Gaussian distributions.

**How it Works**:

1. Initialize means, covariances, and mixture weights
2. E-Step: Compute probability that each point belongs to each cluster
3. M-Step: Update parameters using these probabilities
4. Repeat until convergence

**Pros**:

* Soft clustering (points belong to multiple clusters with probabilities)
* Flexible in cluster shape due to covariance matrices

**Cons**:

* Requires knowing `k`
* Sensitive to initialization

**Use Cases**:

* Speaker recognition
* Background subtraction in videos

---

### 6. OPTICS (Ordering Points To Identify Clustering Structure)

**Definition**:
A density-based clustering algorithm similar to DBSCAN but handles varying densities better.

**Pros**:

* Handles varying cluster density
* Generates reachability plot for visual cluster structure

**Cons**:

* More complex and slower than DBSCAN

---

### 7. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)

**Definition**:
Efficient for large datasets, builds a CF (Clustering Feature) tree incrementally.

**Pros**:

* Scalable to large datasets
* Can be combined with other clustering methods

**Cons**:

* May not work well with non-spherical clusters

---

## 📊 Summary Table

| Algorithm    | Requires k | Handles Outliers | Arbitrary Shape | Scalability | Main Idea                   |
| ------------ | ---------- | ---------------- | --------------- | ----------- | --------------------------- |
| K-Means      | Yes        | No               | No              | High        | Partition into k clusters   |
| Hierarchical | No         | No               | Somewhat        | Low         | Merge closest clusters      |
| DBSCAN       | No         | Yes              | Yes             | Medium      | Density-based clustering    |
| Mean Shift   | No         | Yes              | Yes             | Low         | Move to data density peaks  |
| GMM          | Yes        | No               | Yes             | Medium      | Probabilistic clustering    |
| OPTICS       | No         | Yes              | Yes             | Low         | Order-based density scan    |
| BIRCH        | Yes        | No               | No              | High        | CF tree-based summarization |

---

## 🧠 Final Tips

* Always **scale your data** (StandardScaler) before clustering.
* Use **visualizations** (scatter plots, silhouette scores, dendrograms) to evaluate clustering quality.
* Run clustering multiple times with different parameters for best results.

---

## 🔹 Types of Clustering Algorithms

### 1. K-Means Clustering

**Definition**:
K-Means is a partition-based clustering algorithm that divides the data into `k` non-overlapping subsets (clusters).

**How it Works**:

1. Choose the number of clusters `k`.
2. Initialize `k` centroids randomly.
3. Assign each data point to the nearest centroid (using Euclidean distance).
4. Compute new centroids as the mean of all points assigned to that cluster.
5. Repeat steps 3–4 until convergence (centroids no longer change significantly).

**Mathematical Objective**:
Minimize the within-cluster sum of squares (WCSS):

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$

**Pros**:

* Simple and fast
* Works well with spherical clusters

**Cons**:

* Need to specify `k`
* Sensitive to initial centroid positions
* Poor performance on non-convex shapes or varying densities

**Use Cases**:

* Customer segmentation
* Market basket analysis
* Image compression

**Python Usage**:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Example data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and fit KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Predict cluster labels
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.title("K-Means Clustering")
plt.show()
```

**Choosing `k`: Elbow Method**

```python
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```

---

### 2. Hierarchical Clustering

**Definition**:
Builds a tree of clusters (dendrogram). Can be:

* Agglomerative: Bottom-up approach
* Divisive: Top-down approach

**How it Works (Agglomerative)**:

1. Treat each data point as a single cluster.
2. Merge the two closest clusters.
3. Repeat until only one cluster remains.

**Linkage Criteria**:

* Single linkage (minimum distance)
* Complete linkage (maximum distance)
* Average linkage

**Pros**:

* No need to specify number of clusters initially
* Dendrogram provides deep insight into data

**Cons**:

* Computationally expensive (O(n^3))
* Not scalable for large datasets

**Use Cases**:

* Gene expression analysis
* Taxonomy

...


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#-------------------------------------
# We have
N = 2 # characteristics, namely, x and y coordinate

# Number os random points generated in each ellipse
d_1 = 10
d_2 = 10
d_3 = 10
# We generate few points because the manual implementations is very slow

#Sample number
d = d_1 + d_2 + d_3

#-------------------------------------
# Pseudo-data generation - we generate d_points random numbers in [x0 - a, x0 + a] through an uniform distribution

def xy_gen(d_points, x0, y0, a, b):

    x_min = x0 - a
    x_max = x0 + a

    # np.random.uniform(limite_inferior, limite_superior, quantidade)
    x = np.random.uniform(x_min, x_max, d_points)

    y_min = y0 - b*np.sqrt(1.0 - ((x-x0)/a)**2)
    y_max = y0 + b*np.sqrt(1.0 - ((x-x0) /a)**2)

    y = np.random.uniform(y_min, y_max, d_points)

    return (x, y)

#-------------------------------------
# Graph generator

# 1. Generating the pseudo-data for 3 ellipses (clusters)
x1, y1 = xy_gen(d_1, x0=1, y0=2, a=1.5, b=1.0)
c1 = np.full(d_1, 'royalblue')

x2, y2 = xy_gen(d_2, x0=2, y0=3, a=3.0, b=1.8)
c2 = np.full(d_2, 'darkorange')

x3, y3 = xy_gen(d_3, x0=4, y0=5, a=2.2, b=2.5)
c3 = np.full(d_3, 'forestgreen')

#The color label c is only for the final comparison of the results

#-------------------------------------
# We create the X: all points in a single matrix

x_all = np.concatenate([x1, x2, x3])
y_all = np.concatenate([y1, y2, y3])
c_all = np.concatenate([c1, c2, c3])

X = np.column_stack((x_all, y_all)) # Ordered pairs [[x, y], [x, y], ...]

# We create a vector with the color label for the comparison of the figures in the final
X_label = np.column_stack((x_all, y_all, c_all)) # Ordered pairs [[x, y], [x, y], ...]

# Extracting colors directly from X_label (Your logic)
colors_labeled = []

for j in range(len(X)):
    coordinate_color = X_label[j, 2]
    colors_labeled.append(coordinate_color)

#------------------------------------------------------------------
# HIERARCHICAL CLUSTERING MANUAL (SINGLE LINKAGE)
#------------------------------------------------------------------

# 2. Initialization: Each point is a cluster
clusters = [[i] for i in range(len(X))]

# We choose the metric: euclidean in this case
def euclidean_distance(p1, p2):
    """Euclidean Metric"""
    return np.sqrt(np.sum((p1 - p2)**2))

def D(cluster_i, cluster_j, data):
    """Classifier (Single Linkage): Shortest distance between two sets"""
    min_dist = np.inf
    for idx_i in cluster_i:
        for idx_j in cluster_j:
            dist = euclidean_distance(data[idx_i], data[idx_j])
            if dist < min_dist:
                min_dist = dist
    return min_dist

# 3. Manual Clustering Process
# We will reduce until only 'target_k' clusters remain
target_k = 3

print(f"Starting with {len(clusters)} clusters...")
print("Warning: The manual process with 300 points might take a few minutes. Please wait...")

while len(clusters) > target_k:
    min_dist_global = np.inf
    to_merge = (0, 0)

    # Find the two closest clusters according to classifier D
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist_clusters = D(clusters[i], clusters[j], X)
            if dist_clusters < min_dist_global:
                min_dist_global = dist_clusters
                to_merge = (i, j)

    # Merge the two sets (A_i U A_j)
    idx_i, idx_j = to_merge
    clusters[idx_i].extend(clusters[idx_j])
    clusters.pop(idx_j)

    if len(clusters) % 50 == 0:
        print(f"Remaining clusters: {len(clusters)}")

#------------------------------------------------------------------
# METRICS EVALUATION
#------------------------------------------------------------------
# 4. Assigning Labels for the Plot
labels_manual = np.zeros(len(X))
for cluster_id, points_indices in enumerate(clusters):
    for idx in points_indices:
        labels_manual[idx] = cluster_id

hc_sil_score = silhouette_score(X, labels_manual)
hc_ari_score = adjusted_rand_score(colors_labeled, labels_manual)

print("\n" + "="*50)
print("     HIERARCHICAL CLUSTERING EVALUATION METRICS")
print("="*50)
print(f"1. Silhouette Score:          {hc_sil_score:.4f}  (Closer to 1.0 is better)")
print(f"2. Adjusted Rand Index (ARI): {hc_ari_score:.4f}  (Closer to 1.0 is better)")
print("="*50)

#-------------------------------------------------------------------
# FIGURES
#-------------------------------------------------------------------

# 2. Figure configuration
plt.figure(figsize=(8, 6))

# 3. Plot. The scatter don't join the points with a line as the plot. Alpha channel controls the transparency of the points alpha = 0, the point is invisible
plt.scatter(x1, y1, color=c1[0], alpha=1, edgecolors='w', linewidth=0.5, label='Cluster 1')
plt.scatter(x2, y2, color=c2[0], alpha=1, edgecolors='w', linewidth=0.5, label='Cluster 2')
plt.scatter(x3, y3, color=c3[0], alpha=1, edgecolors='w', linewidth=0.5, label='Cluster 3')

# 4. Labels
plt.title('Synthetic Data Generation (3 Ellipses)', fontsize=14, fontweight='bold')
plt.xlabel(r'$\lambda_1 =$ X coordinate', fontsize=12)
plt.ylabel(r'$\lambda_2 =$ Y coordinate', fontsize=12)

# Axis x and y have the same scale
plt.axis('equal')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 5. Save and show the graph
plt.tight_layout()
plt.savefig('outputs/plots/pseudo_data_ellipses.png', dpi=300)

#---------------------------------------------------------------------

# 5. Visualization: Original vs Manual Hierarchical Clustering
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Graph 1: Original Data ---
# We use 'colors_labeled' to show the true ground truth of the generated ellipses
ax1.scatter(X[:, 0], X[:, 1], color=colors_labeled, alpha=0.8, edgecolors='w', linewidth=0.5)
ax1.set_title('Original Data (True Classes)', fontsize=14, fontweight='bold')
ax1.set_xlabel("X coordinate", fontsize=12)
ax1.set_ylabel("Y coordinate", fontsize=12)
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Graph 2: Manual Hierarchical Clustering Result ---
# We use 'labels_manual' to show the output of our Single Linkage algorithm
# --- Graph 2: Manual Hierarchical Clustering Result ---
# 1. Calculate centroids for the formed clusters and match colors
centroid_colors_manual = {}
for cluster_id in np.unique(labels_manual):
    # Calculate the mean (centroid) of the points in this cluster
    cluster_points = X[labels_manual == cluster_id]
    centroid = np.mean(cluster_points, axis=0)
    
    # Find the closest original point to this centroid
    distances = [euclidean_distance(centroid, X[i]) for i in range(len(X))]
    closest_idx = np.argmin(distances)
    centroid_colors_manual[cluster_id] = colors_labeled[closest_idx]

# 2. Map the labels to the true colors
pred_colors_manual = [centroid_colors_manual[label] for label in labels_manual]

# 3. Plot with the matched colors
ax2.scatter(X[:, 0], X[:, 1], color=pred_colors_manual, alpha=0.8, edgecolors='w', linewidth=0.5)

ax2.set_title(f"Manual Hierarchical (Single Linkage, k={target_k})", fontsize=14, fontweight='bold')
ax2.set_xlabel("X coordinate", fontsize=12)
ax2.set_ylabel("Y coordinate", fontsize=12)
ax2.axis('equal')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('outputs/plots/hierarchical_manual_comparison.png', dpi=300)

#------------------------------------------------------------------
# HIERARCHICAL CLUSTERING SCIPY (WARD LINKAGE)
#------------------------------------------------------------------

# 2. Generating the Linkage Matrix
# 'ward' is great for ellipses as it minimizes the variance.
Z = linkage(X, method='ward')

# 3. Creating the comparative figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Graph 1: The Dendrogram ---
# 'truncate_mode' helps not to pollute the graph with 300 lines at the base
dendrogram(Z, ax=ax1, leaf_rotation=45)
ax1.set_title("Hierarchical Dendrogram (Ward)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Cluster Size (or Point Index)")
ax1.set_ylabel("Distance Threshold (d)")

# --- Graph 2: The Result in 2D Space ---
# Let's cut the tree at k=3 to compare with K-Means
k = 3
labels_h = fcluster(Z, k, criterion='maxclust')

# 1. Calculate centroids for SciPy clusters and match colors
centroid_colors_scipy = {}
for cluster_id in np.unique(labels_h):
    cluster_points = X[labels_h == cluster_id]
    centroid = np.mean(cluster_points, axis=0)
    
    distances = [euclidean_distance(centroid, X[i]) for i in range(len(X))]
    closest_idx = np.argmin(distances)
    centroid_colors_scipy[cluster_id] = colors_labeled[closest_idx]

# 2. Map the labels to the true colors
pred_colors_scipy = [centroid_colors_scipy[label] for label in labels_h]

# 3. Plot with the matched colors
ax2.scatter(X[:, 0], X[:, 1], color=pred_colors_scipy, alpha=0.6, edgecolors='w')

ax2.set_title(f"SciPy Hierarchical Clustering (Ward, k={k})", fontsize=14, fontweight='bold')
ax2.set_xlabel("X coordinate")
ax2.set_ylabel("Y coordinate")
ax2.axis('equal')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('outputs/plots/hierarchical_comparison.png', dpi=300)
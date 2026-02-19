import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# 1. Create 10x10 Grid
rows = 10
cols = 10
n = rows * cols

polygons = []
ids = []
for i in range(rows):
    for j in range(cols):
        # Create a square polygon for each grid cell
        poly = Polygon([(j, i), (j+1, i), (j+1, i+1), (j, i+1)])
        polygons.append(poly)
        ids.append(i * cols + j)

gdf = gpd.GeoDataFrame({'geometry': polygons, 'id': ids})

# 2. Define 7 Complex Clusters (Irregular Shapes)
# We define a 10x10 grid of cluster labels manually to ensure irregular "union of rectangles" shapes
# 7 Clusters: 0 to 6
cluster_grid = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 3, 3, 1, 1, 2, 2, 2, 2],
    [0, 0, 3, 3, 3, 1, 1, 4, 4, 4],
    [0, 0, 3, 3, 3, 5, 5, 4, 4, 4],
    [6, 6, 6, 3, 3, 5, 5, 4, 4, 4],
    [6, 6, 6, 5, 5, 5, 5, 4, 4, 4],
    [6, 6, 6, 5, 5, 5, 5, 5, 5, 5],
    [6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
    [6, 6, 6, 6, 6, 5, 5, 5, 5, 5]
])

def assign_complex_cluster(r, c):
    return cluster_grid[9-r, c] # Flip r to match visual grid if needed, or just standard. Let's start standard.
    # Actually, in the notebook loop: for i in range(rows): for j in range(cols). i is row (y), j is col (x).
    # Usually grid (0,0) is bottom-left in plots, but matrix (0,0) is top-left.
    # Let's just use direct mapping and see the plot.
    
    return cluster_grid[i, j]

cluster_labels = []
for i in range(rows):
    for j in range(cols):
        cluster_labels.append(cluster_grid[i, j])

gdf['true_cluster'] = cluster_labels

# 3. Simulate Covariates and Response
np.random.seed(42)
P = 2
# Reduced scale to 0.2 to ensure better separation
X = np.random.normal(loc=0.0, scale=0.2, size=(n, P))

# True Coefficients for 7 clusters
# Distinct betas to ensure separation in y
true_betas = np.array([
    [-5.0, -5.0],  # C0
    [-3.0, -3.0],  # C1
    [-1.0, -1.0],  # C2
    [ 1.0,  1.0],  # C3
    [ 3.0,  3.0],  # C4
    [ 5.0,  5.0],  # C5
    [ 7.0,  7.0]   # C6
])
# Check if 7 betas are defined
assert len(true_betas) == 7

true_alpha = 0

y = np.zeros(n)
for i in range(n):
    c = cluster_labels[i]
    if c >= 7: print(f"Error: cluster label {c} out of bounds")
    beta = true_betas[c]
    mu = true_alpha + np.dot(X[i], beta)
    y[i] = mu + np.random.randn() * 0.1 # Low noise

gdf['y'] = y
gdf['x1'] = X[:, 0]
gdf['x2'] = X[:, 1]

# Plot True Clusters
plt.figure(figsize=(6, 6))
gdf.plot(column='true_cluster', categorical=True, legend=True, 
         cmap='tab10', edgecolor='black', linewidth=0.5)
plt.title('True Synthetic Clusters (7 Irregular Shapes)')
plt.axis('off')
plt.savefig('true_clusters_v2.png')
plt.close()

# Simulate a "prediction" that matches perfectly but with permuted labels
# To verify confusion matrix matching
permuted_labels = np.array(cluster_labels).copy()
# Swap labels 0 and 1
mask0 = np.array(cluster_labels) == 0
mask1 = np.array(cluster_labels) == 1
permuted_labels[mask0] = 1
permuted_labels[mask1] = 0
gdf['predicted_cluster'] = permuted_labels

# Fix Confusion Matrix with Hungarian Algorithm
def match_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    # We want to maximize the sum of diagonal elements (matches)
    # linear_sum_assignment minimizes cost, so we use negative confusion matrix
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create a mapping dictionary {old_label: new_label}
    # col_ind[i] is the predicted label that corresponds to true label row_ind[i] (which is i)
    # So we want to map pred_label -> true_label
    # mapping: pred_label (col_ind[i]) -> true_label (row_ind[i])
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    return mapping

mapping = match_labels(gdf['true_cluster'], gdf['predicted_cluster'])
print("Label mapping (Pred -> True):", mapping)

# Apply mapping
gdf['matched_predicted'] = gdf['predicted_cluster'].map(mapping)

# Calculate ARI
ari = adjusted_rand_score(gdf['true_cluster'], gdf['matched_predicted'])
print(f"Adjusted Rand Index: {ari:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(gdf['true_cluster'], gdf['matched_predicted'])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Cluster (Matched)')
plt.ylabel('True Cluster')
plt.title('Confusion Matrix: True vs Predicted (Matched)')
plt.savefig('confusion_matrix_v2.png')
plt.close()

print("Verification complete. Images saved.")

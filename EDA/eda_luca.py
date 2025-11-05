# IMPORTS
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import libpysal
import esda
import matplotlib.patches as mpatches
import numpy as np


# data import
DATA_PATH = "../data/merged_ecec_province.geojson"
gdf = gpd.read_file(DATA_PATH)

# CHECK DATAFRAME DIMENSIONS AND TYPES
print(f"The dataframe has dimensions {gdf.shape}") # 107 provinces, 63 variables
print(f"The dataframe has {gdf.select_dtypes(include='number').shape[1]} numerical variables") # 47 actual variables
nan_cols = gdf.select_dtypes(exclude='number').columns # only codes and names in non-numeric
print(f"The non numerical columns are: {nan_cols.tolist()}")
print("\tNon-numerical are just codes and names")

# CHECK DISTRIBUTIONS OF NUMERICAL VARIABLES
summary = (
    gdf.select_dtypes(include='number').describe()
        .T[['mean', 'std', 'min', 'max']].round(2)
)

gdf.hist(figsize=(24, 20), bins=20)
plt.tight_layout()
plt.show()

# CORRELATION MATRIX
gdf.select_dtypes('number') \
    .corr(method='spearman') \
    .style.format(precision=2) \
    .background_gradient(cmap='coolwarm')
    
# FOCUS ON SPECIFIC VARIABLES

var_subset = [
    'prov_name',
    'rip_name',
    'geometry',
    'fem_empl_rate', 
    'unemp_rate', 
    'fem_edu_rate',
    'gdp',
    'service_empl_rate', 
    'dependency_rate',
    'graduate_mobility_rate',
    'per_capita_public_expenditure',
    'ecec_participation',
    'coverage',
    'per_capita_user_contribution',
    'fem_maj_empl_rate']

gdf_subset = gdf[[col for col in gdf.columns if col in var_subset]]

sns.pairplot(
    gdf_subset,
    hue='rip_name',  # color by this variable
    plot_kws={'alpha': 0.6}
)
plt.show()

# CHECK SPATIAL AUTOCORRELATION

gdf_check = gdf_subset.copy()

# Spatial weights
w = libpysal.weights.Queen.from_dataframe(gdf_check, use_index=True)
w.transform = 'r'

# Variable
y = gdf_check['fem_empl_rate'].values

# Local Moran
lm = esda.Moran_Local(y, w)

# Cluster classification
sig = lm.p_sim < 0.05
labels = np.array(['Not significant'] * len(lm.q))
labels[(lm.q == 1) & sig] = 'HH'
labels[(lm.q == 2) & sig] = 'LH'
labels[(lm.q == 3) & sig] = 'LL'
labels[(lm.q == 4) & sig] = 'HL'
gdf_check['cluster'] = labels

# Plot
cluster_colors = {'HH':'red', 'LL':'blue', 'HL':'pink', 'LH':'cyan'}
cluster_labels = {
    'HH': 'High-High (hotspot)',
    'LL': 'Low-Low (coldspot)',
    'HL': 'High-Low (outlier)',
    'LH': 'Low-High (outlier)'
}

fig, ax = plt.subplots(figsize=(12, 12))
gdf_check.plot(color='lightgrey', ax=ax)

for cluster, color in cluster_colors.items():
    subset = gdf_check[gdf_check['cluster'] == cluster]
    if not subset.empty:
        subset.plot(color=color, ax=ax)

patches = [mpatches.Patch(color=color, label=cluster_labels[cluster])
           for cluster, color in cluster_colors.items()
           if not gdf_check[gdf_check['cluster']==cluster].empty]

plt.legend(handles=patches, title="Local Moran's I Clusters")
ax.set_title("Female employment: Local Moran's I Clusters")
ax.axis('off')
plt.show()

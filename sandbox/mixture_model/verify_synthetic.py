
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
import arviz as az
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import AgglomerativeClustering
from numba import njit
import os

az.style.use("arviz-darkgrid")

# 1. Create 10x10 Grid
rows = 10
cols = 10
n = rows * cols

polygons = []
ids = []
for i in range(rows):
    for j in range(cols):
        poly = Polygon([(j, i), (j+1, i), (j+1, i+1), (j, i+1)])
        polygons.append(poly)
        ids.append(i * cols + j)

gdf = gpd.GeoDataFrame({'geometry': polygons, 'id': ids})

# 2. Define 4 Quadrant Clusters with Non-Smooth Boundaries
def assign_quadrant_cluster(r, c):
    # Base Quadrants
    # 0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right
    
    # Deterministic base
    if r < 5:
        if c < 5:
            base = 0
        else:
            base = 1
    else:
        if c < 5:
            base = 2
        else:
            base = 3
            
    # Add noise at boundaries (rows 4,5 and cols 4,5)
    # If we are close to the vertical boundary (c=4 or c=5)
    if (c == 4 or c == 5):
        if np.random.rand() < 0.3: # 30% chance to flip horizontally
            if base == 0: base = 1
            elif base == 1: base = 0
            elif base == 2: base = 3
            elif base == 3: base = 2
            
    # If we are close to the horizontal boundary (r=4 or r=5)
    if (r == 4 or r == 5):
        if np.random.rand() < 0.3: # 30% chance to flip vertically
             if base == 0: base = 2
             elif base == 2: base = 0
             elif base == 1: base = 3
             elif base == 3: base = 1
             
    return base

np.random.seed(42) # Ensure reproducible "random" boundaries
cluster_labels = []
for i in range(rows):
    for j in range(cols):
        cluster_labels.append(assign_quadrant_cluster(i, j))

gdf['true_cluster'] = cluster_labels

# 3. Simulate Covariates and Response
np.random.seed(42)
P = 2
# Reduced scale for X to make clusters tighter (variance reduction)
X = np.random.normal(loc=2.0, scale=0.5, size=(n, P))

# True Coefficients for 4 clusters
# Chosen to have distinct means
true_betas = np.array([
    [ -5.0,  -5.0],  # C0 (Top-Left)
    [ -1.0,  -1.0],  # C1 (Top-Right)
    [  3.0,   3.0],  # C2 (Bottom-Left)
    [  8.0,   8.0]   # C3 (Bottom-Right)
])
true_alpha = 1.0

y = np.zeros(n)
for i in range(n):
    c = cluster_labels[i]
    beta = true_betas[c]
    mu = true_alpha + np.dot(X[i], beta)
    y[i] = mu + np.random.randn() * 0.5

gdf['y'] = y
gdf['x1'] = X[:, 0]
gdf['x2'] = X[:, 1]

print("Data generated with complex clusters.")

# Preprocessing
scaler = StandardScaler()
X_star = scaler.fit_transform(X)
y_star = y
mean_y = np.mean(y_star)
std_y = np.std(y_star)
median_y = np.median(y_star)

# Adjacency
sindex = gdf.sindex
W = np.zeros((n, n), dtype=int)
for i, geom in enumerate(gdf.geometry):
    possible_matches = list(sindex.intersection(geom.bounds))
    for j in possible_matches:
        if j <= i: continue
        if geom.touches(gdf.geometry[j]):
            W[i, j] = 1
            W[j, i] = 1

# Stan Model
mix_model = """
functions {
  real car_leroux_lpdf(
    vector phi,
    real tau2,
    real rho,
    matrix W,
    vector D,
    vector lambda,
    int N
  ) {
    vector[N] Dphi;
    vector[N] Wphi;
    vector[N] det_terms;
    real quad;

    Dphi = D .* phi;        // D * phi
    Wphi = W * phi;         // W * phi

    for (i in 1 : N) {
      det_terms[i] = log(rho * lambda[i] + (1 - rho)); // eigenvalues of D-W
    }

    quad = rho * (dot_product(phi, Dphi) - dot_product(phi, Wphi)) + (1 - rho) * dot_product(phi, phi);

    return 0.5 * (
      N * log(tau2)
      + sum(det_terms)  
      - tau2 * quad
    );
  }
}

data {
  int<lower=1> N;          
  int<lower=1> P;          
  int<lower=1> K;  
  real std_y;
  real median_y;        
  vector[N] y_star;             
  matrix[N, P] X_star;           
  matrix<lower=0, upper=1>[N, N] W;  
}

transformed data {
  vector[N] D;            // diagonal entries of D
  vector[N] lambda;       // eigenvalues of D-W

  for (i in 1:N)
    D[i] = sum(W[i]);

  lambda = eigenvalues_sym(diag_matrix(D) - W);      // eigenvalues
}

parameters {
  simplex[K] pi;           
  vector[K] alpha;  
  vector[P] b0;
  vector<lower = 1e-6, upper=100>[P] lambda_b;      
  matrix[P, K] beta; 
  vector<lower=1e-6>[K] tau;  
  real<lower=1e-6> tau2;                  // CAR precision
  vector[N] phi_raw;                        // Spatial random effects
  real<lower=0, upper=0.95> rho;      // Leroux dependence parameter
}

transformed parameters {
  vector[N] phi;
  phi = phi_raw - mean(phi_raw);

  vector<lower=0>[K] sigma = inv_sqrt(tau); 

  vector<lower=0>[P] sd_beta = std_y * sqrt(lambda_b); 
}


model {
  // Priors

  pi ~ dirichlet(rep_vector(0.5, K)); 

  for (j in 1:P) {
    b0[j] ~ normal(0, 5);
    lambda_b[j] ~ gamma(0.5, 0.5);
  }

  for(k in 1:K) {
    tau[k] ~ gamma(2, 1);
  }

  for(k in 1:K) {
    alpha[k] ~ normal(0, 5);
  }

  for (j in 1:P) {
    for (k in 1:K) {
      beta[j, k] ~ normal(b0[j], sd_beta[j]);
    }
  }


  // CAR prior
  tau2  ~ gamma(2, 1);
  rho   ~ beta(1, 1);
  phi_raw ~ car_leroux(tau2, rho, W, D, lambda, N);

  // Likelihood
  for (i in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      real mu_ik;
      mu_ik = alpha[k] + dot_product(row(X_star, i), beta[, k]) + phi[i];
      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}

generated quantities {
  array[N] int z;

  for (i in 1:N) {
    vector[K] lps;
    vector[K] r_i;

    for (k in 1:K) {
      real mu_ik = alpha[k] + dot_product(row(X_star, i), beta[, k]) + phi[i];
      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma[k]);
    }

    r_i = softmax(lps);
    z[i] = categorical_rng(r_i);
    
  }
}

"""

with open("synthetic_mixture.stan", "w") as f:
    f.write(mix_model)

K_model = 10 

data = {
    "N": n,
    "P": P,
    "K": K_model,
    "std_y": std_y,
    "median_y": median_y,
    "y_star": y_star,
    "X_star": X_star,
    "W": W,
}

print("Running Stan model...")
mix_stan = CmdStanModel(stan_file="synthetic_mixture.stan")
fit = mix_stan.sample(
    data=data,
    chains=4,
    iter_warmup=500,
    iter_sampling=500,
    adapt_delta=0.99,
    max_treedepth=15,
    show_progress=True
)

z_draws = fit.stan_variable("z")

@njit
def build_psm(posterior):
    n_draws, n_obs = posterior.shape
    psm = np.zeros((n_obs, n_obs))
    for row in posterior:
        for r_idx, i in enumerate(row):
            for c_idx, j in enumerate(row):
                if i == j:
                    psm[r_idx, c_idx] += 1
    psm /= n_draws
    return psm

print("Building PSM...")
psm = build_psm(z_draws.astype(np.int64))

def binder_loss(psm, labels):
    labels = np.asarray(labels)
    A = (labels[:, None] == labels[None, :]).astype(float)
    iu = np.triu_indices(psm.shape[0], k=1)
    return float(np.sum(np.abs(A[iu] - psm[iu])))

def binder_optimal_partition(psm, k_min, k_max):
    psm = np.asarray(psm, dtype=float)
    psm = 0.5 * (psm + psm.T)
    np.fill_diagonal(psm, 1.0)
    dist = 1.0 - psm
    np.fill_diagonal(dist, 0.0)

    best_loss = np.inf
    best_labels = None

    for k in range(k_min, min(k_max, psm.shape[0]) + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="average",
            metric="precomputed",  
        )
        labels = model.fit_predict(dist)
        loss = binder_loss(psm, labels)
        if loss < best_loss:
            best_loss = loss
            best_labels = labels
    return best_labels, best_loss

print("Finding optimal partition...")
labels_opt, loss_opt = binder_optimal_partition(psm, k_min=2, k_max=12)
gdf['predicted_cluster'] = labels_opt

from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(gdf['true_cluster'], gdf['predicted_cluster'])
print(f"Adjusted Rand Index: {ari:.4f}")

if ari > 0.8:
    print("SUCCESS: Clusters recovered accurately.")
else:
    print("WARNING: Cluster recovery suboptimal. (Confusion Matrix below)")
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(gdf['true_cluster'], gdf['predicted_cluster']))

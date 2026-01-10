# imports
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import libpysal
import esda
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## DATA ##
DATA_PATH = "../data/updated_data.geojson"
gdf = gpd.read_file(DATA_PATH)
    
## ADJACENCY MATRIX ##
A = libpysal.weights.Rook.from_dataframe(gdf)

# I checked and this is equal to the one that Chi hardcoded
# np.array_equal(A.full()[0], W), where W is Chi's matrix

## COVARIATES SELECTION
selected_covariates = ['fem_house_rate',
                       'net_migration_rate', 
                        'avg_children_per_woman',
                        'n_members_family',
                        'fem_maj_empl_rate',
                        'ecec_diffusion',
                        'per_capita_public_expenditure',
                        'ecec_participation',  
                        'coverage',
                        'per_capita_user_contribution', 
                        'ageing_index', 
                        'service_empl_rate']

X = gdf.select_dtypes(include="number")
X.isna().sum() # no NAs
# Select only the desired covariates
X = X[selected_covariates]

#X = gdf.select_dtypes(include="number").drop(columns=["fem_empl_rate"])
y = gdf["fem_empl_rate"].values
y = y / 100
y = np.log(y / (1 - y))
# x_offset = gdf["mal_empl_rate"].values
# x_offset=x_offset/100   
# x_offset = np.log(x_offset / (1 - x_offset))
X_standardized = (X - X.mean()) / X.std()

# possibily we can get sparse adjacency matrix for more efficient computation
# A_sparse = A.sparse # sparse matrix
# adj_matrix = A_sparse.tocoo()
# node1 = adj_matrix.row + 1  # remember stan is 1-indexed
# node2 = adj_matrix.col + 1

stan_data = {
    "N": len(gdf),
    "P": X_standardized.shape[1],
    "y": y,
    "X": X_standardized.values,
    "W": A.full()[0] # maybe sparse representation is better, see above
}


## MODEL COMPILATION 
#TODO fix this by removing offset
#TODO extract the posterior sample

from cmdstanpy import CmdStanModel
stan_code = """
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
  int<lower=1> N;                       // number of regions
  int<lower=1> P;                       // number of covariates
  matrix[N, P] X;                        // design matrix
  vector[N] y;                          // response variable
  matrix<lower=0, upper=1>[N, N] W;    // adjacency matrix
}

transformed data {
  vector[N] D;            // diagonal entries of D
  vector[N] lambda;       // eigenvalues of D-W

  for (i in 1:N)
    D[i] = sum(W[i]);

  lambda = eigenvalues_sym(diag_matrix(D) - W);      // eigenvalues
}

parameters {
  real alpha;                          // intercept
  vector[P] beta;                      // coefficients
  real<lower=0> sigma;                 
  real<lower=0> tau2;                  // CAR precision
  vector[N] phi_raw;                   // Spatial random effects
  real<lower=0, upper=0.95> rho;      // Leroux dependence parameter
}

transformed parameters {
  vector[N] phi;
  phi = phi_raw - mean(phi_raw);
}


model {
  // Priors
  alpha ~ normal(0, 2);
  beta  ~ normal(0, 2);
  sigma ~ normal(0, 1);
  tau2  ~ gamma(1, 0.01);
  rho   ~ beta(2,2);

  // CAR Leroux prior
  phi_raw ~ car_leroux(tau2, rho, W, D, lambda, N);

  // Likelihood
  y ~ normal(alpha + X * beta + phi, sigma);

}

"""

# Write stan model to file
stan_file = "./car_leroux_model.stan"
with open(stan_file, "w") as f:
    print(stan_code, file=f)
    
# Compile stan model
from cmdstanpy import CmdStanModel
stan_file = "./car_leroux_model.stan"
model = CmdStanModel(stan_file=stan_file)

# run model
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    seed=42,
    adapt_delta=0.999,
    max_treedepth=30,
    force_one_process_per_chain=True
)

# plot the mean of Yi sampled from the full conditional
# 1. Extract the 8,000 iterations for each parameter
alphas = fit.stan_variable("alpha")  # Shape: (8000,)
betas = fit.stan_variable("beta")    # Shape: (8000, P)
phis = fit.stan_variable("phi")      # Shape: (8000, N)
X = X_standardized.values            # Shape: (N, P)

# posterior mean of Y
y_post_means = alphas[:, None] + np.dot(betas, X.T) + phis
y_post_means.shape

##### VISUALISE RESULTS: 3 plots #####

### 1. plot the posterior density of y in n random iterations, each line is one iteration
plt.figure(figsize=(10, 6))

# Plot 100 versions of the "Cleaned" Structural Mixture
n_iterations = 100
for i in np.random.choice(8000, n_iterations):
    sns.kdeplot(y_post_means[i, :], color="forestgreen", alpha=0.2, linewidth=1)
plt.title(f"Mean of Y across {n_iterations} posterior iterations")
plt.legend()
plt.show()

### 2. plot the posterior mean density across all iterations (the average value of the green lines above)
plt.figure(figsize=(10, 6))
sns.kdeplot(y_post_means.flatten(), color="forestgreen", linewidth=2, label="Average Posterior Structural Distribution")
# observed data data
sns.kdeplot(y, color="black", linewidth=2, linestyle="--", label="Observed Data (Y)")

plt.title("Average density across iterations")
plt.legend()
plt.show()

### 3. compute the mean for each province first, across the 8000 iterations, and then plot them
province_means = y_post_means.mean(axis=0) 
sns.kdeplot(province_means, color="red", label="Distribution of Province-Specific Means")


##### VISUALISE THE "CLUSTERS" #####
low_threshold = -0.6 # checked by viz 2 and viz 3 (above)
high_threshold = 0.0 # checked by viz 2 and viz 3
def identify_clusters(val):
  if val < low_threshold:
    return 'low employment'
  elif val < high_threshold:
    return 'medium employment'
  else:
    return 'high employment'

gdf['cluster'] = province_means
gdf['cluster_label'] = gdf['cluster'].apply(identify_clusters)

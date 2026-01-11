"""
All 32 Model Combinations for Normal Mixture

This script generates and runs all 32 combinations of 5 binary model features:
1. cluster_specific_beta - Beta coefficients vary by cluster
2. spatial_effect - CAR Leroux spatial effect
3. cluster_specific_sigma - Sigma varies by cluster
4. offset - Include offset term
5. cluster_specific_alpha - Alpha intercept varies by cluster

Author: Generated for Bayesian Statistics Project
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from cmdstanpy import CmdStanModel
from sklearn.preprocessing import StandardScaler
from numba import njit
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import AgglomerativeClustering
import itertools
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# UTILITY FUNCTIONS (from original notebook)
# =============================================================================

@njit
def build_psm(posterior):
    """Build posterior similarity matrix from sampled labels."""
    n_draws, n_obs = posterior.shape
    psm = np.zeros((n_obs, n_obs))
    for row in posterior:
        for r_idx, i in enumerate(row):
            for c_idx, j in enumerate(row):
                if i == j:
                    psm[r_idx, c_idx] += 1
    psm /= n_draws
    return psm


def order_by_hclust(psm, method="average"):
    """Order PSM by hierarchical clustering for visualization."""
    psm = np.asarray(psm)
    dist = 1.0 - psm
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    return leaves_list(Z)


def binder_loss(psm, labels):
    """Compute expected Binder loss from PSM."""
    labels = np.asarray(labels)
    A = (labels[:, None] == labels[None, :]).astype(float)
    iu = np.triu_indices(psm.shape[0], k=1)
    return float(np.sum(np.abs(A[iu] - psm[iu])))


def binder_optimal_partition(psm, k_min, k_max):
    """Find optimal partition minimizing Binder loss."""
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


# =============================================================================
# STAN CODE GENERATOR
# =============================================================================

def generate_stan_code(config):
    """
    Generate Stan model code based on configuration.
    
    Parameters:
    -----------
    config : dict with keys:
        - cluster_specific_beta: bool
        - spatial_effect: bool
        - cluster_specific_sigma: bool
        - offset: bool
        - cluster_specific_alpha: bool
    
    Returns:
    --------
    str : Stan model code
    """
    cluster_beta = config['cluster_specific_beta']
    spatial = config['spatial_effect']
    cluster_sigma = config['cluster_specific_sigma']
    use_offset = config['offset']
    cluster_alpha = config['cluster_specific_alpha']
    
    code_parts = []
    
    # =========================
    # FUNCTIONS BLOCK (for spatial effect)
    # =========================
    if spatial:
        code_parts.append("""
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

    Dphi = D .* phi;
    Wphi = W * phi;

    for (i in 1 : N) {
      det_terms[i] = log(rho * lambda[i] + (1 - rho));
    }

    quad = rho * (dot_product(phi, Dphi) - dot_product(phi, Wphi)) + (1 - rho) * dot_product(phi, phi);

    return 0.5 * (
      N * log(tau2)
      + sum(det_terms)  
      - tau2 * quad
    );
  }
}
""")
    
    # =========================
    # DATA BLOCK
    # =========================
    data_lines = [
        "data {",
        "  int<lower=1> N;",
        "  int<lower=1> P;",
        "  int<lower=1> K;",
        "  vector[N] y_star;",
        "  matrix[N, P] X_star;",
    ]
    if use_offset:
        data_lines.append("  vector[N] offset_star;")
    if spatial:
        data_lines.append("  matrix<lower=0, upper=1>[N, N] W;")
    data_lines.append("}")
    code_parts.append("\n".join(data_lines))
    
    # =========================
    # TRANSFORMED DATA BLOCK (for spatial)
    # =========================
    if spatial:
        code_parts.append("""
transformed data {
  vector[N] D;
  vector[N] lambda;
  matrix[N, N] DmW;
  
  for (i in 1:N) {
    D[i] = sum(W[i]);
  }
  
  DmW = diag_matrix(D) - W;
  lambda = eigenvalues_sym(DmW);
}
""")
    
    # =========================
    # PARAMETERS BLOCK
    # =========================
    param_lines = [
        "parameters {",
        "  simplex[K] pi;",
    ]
    
    # Alpha
    if cluster_alpha:
        param_lines.append("  vector[K] alpha;")
    else:
        param_lines.append("  real alpha;")
    
    # Beta
    if cluster_beta:
        param_lines.append("  matrix[K, P] beta;")
    else:
        param_lines.append("  vector[P] beta;")
    
    # Sigma
    if cluster_sigma:
        param_lines.append("  vector<lower=1e-6>[K] sigma;")
    else:
        param_lines.append("  real<lower=1e-6> sigma;")
    
    # Spatial parameters
    if spatial:
        param_lines.append("  vector[N] phi;")
        param_lines.append("  real<lower=1e-6> tau2;")
        param_lines.append("  real<lower=0, upper=1> rho;")
    
    param_lines.append("}")
    code_parts.append("\n".join(param_lines))
    
    # =========================
    # MODEL BLOCK
    # =========================
    model_lines = [
        "model {",
        "  // Priors",
        "  pi ~ dirichlet(rep_vector(1.0, K));",
    ]
    
    # Alpha priors
    if cluster_alpha:
        model_lines.append("  for (k in 1:K) {")
        model_lines.append("    alpha[k] ~ normal(0, 2);")
        model_lines.append("  }")
    else:
        model_lines.append("  alpha ~ normal(0, 2);")
    
    # Beta priors
    if cluster_beta:
        model_lines.append("  for (k in 1:K) {")
        model_lines.append("    for (j in 1:P) {")
        model_lines.append("      beta[k, j] ~ normal(0, 3);")
        model_lines.append("    }")
        model_lines.append("  }")
    else:
        model_lines.append("  for (j in 1:P) {")
        model_lines.append("    beta[j] ~ normal(0, 3);")
        model_lines.append("  }")
    
    # Sigma priors
    if cluster_sigma:
        model_lines.append("  for (k in 1:K) {")
        model_lines.append("    sigma[k] ~ inv_gamma(2, 1);")
        model_lines.append("  }")
    else:
        model_lines.append("  sigma ~ inv_gamma(2, 1);")
    
    # Spatial priors
    if spatial:
        model_lines.append("  tau2 ~ gamma(2, 1);")
        model_lines.append("  rho ~ beta(1, 1);")
        model_lines.append("  phi ~ car_leroux(tau2, rho, W, D, lambda, N);")
    
    # Likelihood
    model_lines.append("")
    model_lines.append("  // Likelihood")
    model_lines.append("  for (i in 1:N) {")
    model_lines.append("    vector[K] lps;")
    model_lines.append("    for (k in 1:K) {")
    model_lines.append("      real mu_ik;")
    
    # Build mu_ik based on configuration
    mu_parts = []
    if cluster_alpha:
        mu_parts.append("alpha[k]")
    else:
        mu_parts.append("alpha")
    
    if cluster_beta:
        mu_parts.append("dot_product(row(X_star, i), to_vector(beta[k]))")
    else:
        mu_parts.append("dot_product(row(X_star, i), beta)")
    
    if use_offset:
        mu_parts.append("offset_star[i]")
    
    if spatial:
        mu_parts.append("phi[i]")
    
    mu_expr = " + ".join(mu_parts)
    model_lines.append(f"      mu_ik = {mu_expr};")
    
    # Sigma in likelihood
    if cluster_sigma:
        model_lines.append("      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma[k]);")
    else:
        model_lines.append("      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma);")
    
    model_lines.append("    }")
    model_lines.append("    target += log_sum_exp(lps);")
    model_lines.append("  }")
    model_lines.append("}")
    code_parts.append("\n".join(model_lines))
    
    # =========================
    # GENERATED QUANTITIES BLOCK
    # =========================
    gq_lines = [
        "generated quantities {",
        "  array[N] int z;",
        "",
        "  for (i in 1:N) {",
        "    vector[K] lps;",
        "    vector[K] r_i;",
        "",
        "    for (k in 1:K) {",
    ]
    
    # Build mu_ik for generated quantities (same as model block)
    if cluster_alpha:
        mu_gq = "alpha[k]"
    else:
        mu_gq = "alpha"
    
    if cluster_beta:
        mu_gq += " + X_star[i] * to_vector(beta[k])"
    else:
        mu_gq += " + X_star[i] * beta"
    
    if use_offset:
        mu_gq += " + offset_star[i]"
    
    if spatial:
        mu_gq += " + phi[i]"
    
    gq_lines.append(f"      real mu_ik = {mu_gq};")
    
    if cluster_sigma:
        gq_lines.append("      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma[k]);")
    else:
        gq_lines.append("      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma);")
    
    gq_lines.extend([
        "    }",
        "",
        "    r_i = softmax(lps);",
        "    z[i] = categorical_rng(r_i);",
        "  }",
        "}",
    ])
    code_parts.append("\n".join(gq_lines))
    
    return "\n".join(code_parts)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(data_path, geojson_path, k):
    """Prepare all data needed for model fitting."""
    
    # Load data
    df = pd.read_csv(data_path)
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.reset_index(drop=True)
    
    n = len(df)
    
    # Response variable
    y = df["fem_empl_rate"].to_numpy() / 100
    y_star = np.log(y / (1 - y))
    
    # Covariates
    selected_covariates = [
        "coverage", "per_capita_user_contribution",
        "ecec_participation", "service_empl_rate", "ecec_diffusion", 
    ]
    X = df[selected_covariates].to_numpy()
    X_scaler = StandardScaler()
    X_star = X_scaler.fit_transform(X)
    
    # Offset
    offset = df["wr_men"].to_numpy() / 100
    offset_star = np.log(offset / (1 - offset))
    
    # Spatial weights matrix
    node1 = []
    node2 = []
    num_neighbors = np.zeros(n, dtype=int)
    sindex = gdf.sindex
    
    for i, geom in enumerate(gdf.geometry):
        possible_matches = list(sindex.intersection(geom.bounds))
        for j in possible_matches:
            if j <= i:
                continue
            if geom.touches(gdf.geometry[j]):
                node1.append(i + 1)
                node2.append(j + 1)
                num_neighbors[i] += 1
                num_neighbors[j] += 1
    
    node1 = np.asarray(node1, dtype=int)
    node2 = np.asarray(node2, dtype=int)
    
    W = np.zeros((n, n), dtype=float)
    for i, j in zip(node1 - 1, node2 - 1):
        W[i, j] = 1
        W[j, i] = 1
    
    return {
        'N': n,
        'P': len(selected_covariates),
        'K': k,
        'y_star': y_star.astype(float),
        'X_star': X_star.astype(float),
        'offset_star': offset_star.astype(float),
        'W': W.astype(float),
    }


# =============================================================================
# MODEL RUNNER
# =============================================================================

def run_model(config, config_id, data, output_dir):
    """
    Run a single model configuration.
    
    Returns dict with results or None if failed.
    """
    print(f"\n{'='*60}")
    print(f"Model {config_id}: {config}")
    print(f"{'='*60}")
    
    # Generate Stan code
    stan_code = generate_stan_code(config)
    
    # Write Stan file
    stan_file = os.path.join(output_dir, f"model_{config_id}.stan")
    with open(stan_file, "w") as f:
        f.write(stan_code)
    
    print(f"Stan file written: {stan_file}")
    
    # Prepare data for this configuration
    stan_data = {
        'N': int(data['N']),
        'P': int(data['P']),
        'K': int(data['K']),
        'y_star': data['y_star'],
        'X_star': data['X_star'],
    }
    
    if config['offset']:
        stan_data['offset_star'] = data['offset_star']
    
    if config['spatial_effect']:
        stan_data['W'] = data['W']
    
    try:
        # Compile model
        print("Compiling Stan model...")
        model = CmdStanModel(stan_file=stan_file)
        
        # Run sampling
        print("Running MCMC sampling...")
        fit = model.sample(
            data=stan_data,
            chains=4,
            iter_warmup=2000,
            iter_sampling=2000,
            adapt_delta=0.95,
            max_treedepth=12,
            show_progress=True
        )
        
        # Extract z draws
        z_draws = fit.stan_variable("z")
        
        # Build PSM
        print("Building PSM...")
        psm = build_psm(z_draws.astype(np.int64))
        
        # Get optimal partition
        print("Finding optimal partition...")
        labels_opt, loss_opt = binder_optimal_partition(psm, k_min=1, k_max=5)
        labels = labels_opt + 1
        
        # Count clusters
        n_clusters = len(np.unique(labels))
        
        result = {
            'config': config,
            'config_id': config_id,
            'psm': psm,
            'labels': labels,
            'binder_loss': loss_opt,
            'n_clusters': n_clusters,
            'success': True,
            'error': None
        }
        
        print(f"SUCCESS - Binder loss: {loss_opt:.2f}, Clusters: {n_clusters}")
        return result
        
    except Exception as e:
        print(f"FAILED: {str(e)}")
        return {
            'config': config,
            'config_id': config_id,
            'success': False,
            'error': str(e)
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_all_configs():
    """Generate all 32 model configurations."""
    features = ['cluster_specific_beta', 'spatial_effect', 'cluster_specific_sigma', 'offset', 'cluster_specific_alpha']
    configs = []
    
    for combo in itertools.product([False, True], repeat=5):
        config = dict(zip(features, combo))
        configs.append(config)
    
    return configs


def config_to_string(config):
    """Convert config to readable string."""
    parts = []
    for key, val in config.items():
        short_key = key.replace('cluster_specific_', 'cs_').replace('spatial_effect', 'spatial')
        parts.append(f"{short_key}={'Y' if val else 'N'}")
    return ", ".join(parts)


def main():
    """Main execution function."""
    for c in [2, 3]:  # Run for K=2, 3, 4, and 5
        print(f"\n\nRunning all models for K={c} clusters")
        
        print("="*70)
        print("ALL 32 MODEL COMBINATIONS - Normal Mixture Model")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/updated_data.csv')
        geojson_path = os.path.join(base_dir, '../data/updated_data.geojson')
        output_dir = os.path.join(base_dir, 'stan_models_32')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        print("\nPreparing data...")
        data = prepare_data(data_path, geojson_path, c)
        print(f"Data prepared: N={data['N']}, P={data['P']}, K={data['K']}")
        
        # Generate all configurations
        configs = generate_all_configs()
        print(f"\nGenerated {len(configs)} model configurations")
        
        # Run all models
        results = []
        for i, config in enumerate(configs):
            config_id = i + 1
            result = run_model(config, config_id, data, output_dir)
            results.append(result)
            
            # Save intermediate results
            with open(os.path.join(output_dir, 'results_partial.pkl'), 'wb') as f:
                pickle.dump(results, f)
        
        # Save final results
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        with open(os.path.join(output_dir, 'results_all.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary DataFrame
        summary_data = []
        for r in results:
            row = {
                'model_id': r['config_id'],
                'cluster_beta': r['config']['cluster_specific_beta'],
                'spatial': r['config']['spatial_effect'],
                'cluster_sigma': r['config']['cluster_specific_sigma'],
                'offset': r['config']['offset'],
                'cluster_alpha': r['config']['cluster_specific_alpha'],
                'success': r['success'],
                'binder_loss': r.get('binder_loss', None),
                'n_clusters': r.get('n_clusters', None),
                'error': r.get('error', None)
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, f'summary_{c}_cov.csv'), index=False)

        print(f"\nSummary saved to: summary_{c}.csv")
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(summary_df.to_string())
        
        # Print statistics
        n_success = sum(1 for r in results if r['success'])
        print(f"\n\nSuccessful models: {n_success}/{len(results)}")
        
        if n_success > 0:
            successful = [r for r in results if r['success']]
            losses = [r['binder_loss'] for r in successful]
            print(f"Binder loss range: {min(losses):.2f} - {max(losses):.2f}")
            
            best = min(successful, key=lambda x: x['binder_loss'])
            print(f"\nBest model (lowest Binder loss):")
            print(f"  Model {best['config_id']}: {config_to_string(best['config'])}")
            print(f"  Binder loss: {best['binder_loss']:.2f}")
            print(f"  Clusters: {best['n_clusters']}")
        
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    return results


if __name__ == "__main__":
    results = main()

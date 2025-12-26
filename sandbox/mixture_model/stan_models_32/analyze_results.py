"""
Analysis script for 320 Bayesian mixture models.
Combines all summary CSVs and performs exploratory analysis.
"""

import pandas as pd
import numpy as np
import os

# %% Load and concatenate all data
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

all_data = []
# Skip K=1 (trivially produces 1 cluster with binder_loss=0)
for k in range(2, 6):
    for cov_type in ['full', 'cov']:
        suffix = '' if cov_type == 'full' else '_cov'
        filepath = os.path.join(base_dir, f'summary_{k}{suffix}.csv')
        df = pd.read_csv(filepath)
        df['K'] = k
        df['covariate_set'] = cov_type
        all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
print(f"Total models loaded: {len(combined)}")
print(f"Successful models: {combined['success'].sum()}")

# %% Basic statistics
print("\n" + "="*60)
print("BINDER LOSS STATISTICS")
print("="*60)
print(combined['binder_loss'].describe())

# %% Models achieving target: K=2 or 3 and n_clusters=2 or 3
print("\n" + "="*60)
print("TARGET MODELS: K in {2,3} AND n_clusters in {2,3}")
print("="*60)
target_models = combined[
    (combined['K'].isin([2, 3])) & 
    (combined['n_clusters'].isin([2, 3]))
]
print(f"Found {len(target_models)} models matching criteria")

if len(target_models) > 0:
    target_sorted = target_models.sort_values('binder_loss')
    print("\nTop 10 by lowest binder loss:")
    cols = ['model_id', 'K', 'covariate_set', 'n_clusters', 'binder_loss', 
            'cluster_beta', 'spatial', 'cluster_sigma', 'offset', 'cluster_alpha']
    print(target_sorted[cols].head(10).to_string(index=False))

# %% Best models per K
print("\n" + "="*60)
print("BEST MODEL (lowest binder loss) PER K")
print("="*60)
for k in range(2, 6):
    subset = combined[combined['K'] == k]
    if len(subset) == 0:
        print(f"\nK={k}: No models found")
        continue
    best = subset.loc[subset['binder_loss'].idxmin()]
    print(f"\nK={k}: model_id={int(best['model_id'])}, "
          f"cov={best['covariate_set']}, "
          f"n_clusters={int(best['n_clusters'])}, "
          f"binder_loss={best['binder_loss']:.2f}")
    print(f"  Features: beta={best['cluster_beta']}, spatial={best['spatial']}, "
          f"sigma={best['cluster_sigma']}, offset={best['offset']}, alpha={best['cluster_alpha']}")

# %% Feature importance (marginal effect)
print("\n" + "="*60)
print("FEATURE IMPORTANCE (mean binder loss difference: True - False)")
print("="*60)
features = ['cluster_beta', 'spatial', 'cluster_sigma', 'offset', 'cluster_alpha']
for feat in features:
    with_feat = combined[combined[feat] == True]['binder_loss'].mean()
    without_feat = combined[combined[feat] == False]['binder_loss'].mean()
    delta = with_feat - without_feat
    direction = "REDUCES" if delta < 0 else "INCREASES"
    print(f"{feat:20s}: Delta = {delta:+8.2f}  ({direction} loss)")

# %% Covariate set comparison
print("\n" + "="*60)
print("COVARIATE SET COMPARISON")
print("="*60)
for cov in ['full', 'cov']:
    subset = combined[combined['covariate_set'] == cov]
    label = "10 covariates" if cov == 'full' else "5 ECEC covariates"
    print(f"{label}: mean binder_loss = {subset['binder_loss'].mean():.2f}")

# %% Cluster count distribution
print("\n" + "="*60)
print("CLUSTER COUNT DISTRIBUTION")
print("="*60)
cluster_dist = combined.groupby(['K', 'n_clusters']).size().unstack(fill_value=0)
print(cluster_dist)

# %% Models with n_clusters == 2 or 3 (desirable)
print("\n" + "="*60)
print("MODELS WITH DESIRABLE CLUSTER COUNT (2 or 3)")
print("="*60)
desirable = combined[combined['n_clusters'].isin([2, 3])]
print(f"Total: {len(desirable)} / {len(combined)} models ({100*len(desirable)/len(combined):.1f}%)")

print("\nBreakdown by K (K=2 to 5):")
for k in range(2, 6):
    subset = desirable[desirable['K'] == k]
    print(f"  K={k}: {len(subset)} models")

print("\nTop 5 overall (lowest binder loss, n_clusters in {2,3}):")
top5 = desirable.nsmallest(5, 'binder_loss')
cols = ['model_id', 'K', 'covariate_set', 'n_clusters', 'binder_loss']
print(top5[cols].to_string(index=False))

# %% Save combined data
output_path = os.path.join(base_dir, 'combined_results.csv')
combined.to_csv(output_path, index=False)
print(f"\n[OK] Combined data saved to: {output_path}")

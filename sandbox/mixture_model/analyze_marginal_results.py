"""
Analysis script for Marginal Mixture Models (K=2, 3).
Loads summary CSVs from 'stan_models_marginal/', combines them, and performs exploratory analysis.
"""

import pandas as pd
import numpy as np
import os
import sys

def main():
    # Base directory (where this script is located)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    # Directory containing the marginal model results
    results_dir = os.path.join(base_dir, 'stan_models_marginal')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run 'all_marginal_models.py' first.")
        return

    print(f"Analysis script started.")
    print(f"Looking for results in: {results_dir}")

    # Load and concatenate data for K=2 and K=3 (and others if present)
    all_data = []
    # We prioritize 2 and 3 as requested, but we can look for 2..5 loosely
    k_range = [2, 3, 4, 5] 
    
    for k in k_range:
        filepath = os.path.join(results_dir, f'summary_K{k}_marginal.csv')
        if os.path.exists(filepath):
            print(f"Loading K={k} summary from: {filepath}")
            df = pd.read_csv(filepath)
            df['K'] = k
            all_data.append(df)
        else:
            if k in [2, 3]:
                print(f"Warning: Summary file for K={k} not found ({filepath})")

    if not all_data:
        print("\nNo summary files found! Exiting.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal models loaded: {len(combined)}")
    
    # Filter for success
    if 'success' in combined.columns:
        n_success = combined['success'].sum()
        print(f"Successful models: {n_success}")
        combined = combined[combined['success'] == True].copy()
    else:
        print("Warning: 'success' column not found, assuming all are successful.")

    if len(combined) == 0:
        print("No successful models to analyze.")
        return

    # %% Basic statistics
    print("\n" + "="*60)
    print("BINDER LOSS STATISTICS")
    print("="*60)
    if 'binder_loss' in combined.columns:
        print(combined['binder_loss'].describe())
    else:
        print("Column 'binder_loss' not found.")

    # %% Best models per K
    print("\n" + "="*60)
    print("BEST MODEL (lowest binder loss) PER K")
    print("="*60)
    
    feature_cols = ['cluster_mean', 'spatial', 'cluster_sigma', 'offset']
    
    for k in sorted(combined['K'].unique()):
        subset = combined[combined['K'] == k]
        if len(subset) == 0:
            continue
            
        if 'binder_loss' in subset.columns:
            best_idx = subset['binder_loss'].idxmin()
            best = subset.loc[best_idx]
            
            print(f"\nK={k}: model_id={best.get('model_id', '?')}")
            print(f"  Binder Loss: {best['binder_loss']:.4f}")
            print(f"  Clusters found: {best.get('n_clusters', '?')}")
            
            # Print features
            feat_str = []
            for feat in feature_cols:
                if feat in best:
                    val = "Y" if best[feat] else "N"
                    feat_str.append(f"{feat}={val}")
            print(f"  Features: {', '.join(feat_str)}")

    # %% Feature importance (marginal effect)
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Mean Binder Loss Difference: With - Without)")
    print("="*60)
    
    for feat in feature_cols:
        if feat not in combined.columns:
            continue
            
        with_feat = combined[combined[feat] == True]['binder_loss'].mean()
        without_feat = combined[combined[feat] == False]['binder_loss'].mean()
        
        if pd.isna(with_feat) or pd.isna(without_feat):
            print(f"{feat:20s}: Insufficient data")
            continue
            
        delta = with_feat - without_feat
        direction = "REDUCES" if delta < 0 else "INCREASES"
        print(f"{feat:20s}: Delta = {delta:+8.4f}  ({direction} loss)")

    # %% Cluster count distribution
    print("\n" + "="*60)
    print("CLUSTER COUNT DISTRIBUTION")
    print("="*60)
    if 'n_clusters' in combined.columns:
        cluster_dist = combined.groupby(['K', 'n_clusters']).size().unstack(fill_value=0)
        print(cluster_dist)
    else:
        print("Column 'n_clusters' not found.")

    # %% Save combined data
    output_path = os.path.join(results_dir, 'combined_results_marginal.csv')
    combined.to_csv(output_path, index=False)
    print(f"\n[OK] Combined results saved to: {output_path}")

if __name__ == "__main__":
    main()

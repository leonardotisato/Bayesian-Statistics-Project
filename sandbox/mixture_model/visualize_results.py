"""
Visualization script for 32 Model Combinations Results

Load results from results_all.pkl and create:
1. Posterior Similarity Matrix (PSM) heatmaps
2. Geographic maps showing cluster assignments
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import pickle
import os

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def order_by_hclust(psm, method="average"):
    """Order PSM by hierarchical clustering for visualization."""
    psm = np.asarray(psm)
    dist = 1.0 - psm
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    return leaves_list(Z)


def config_to_string(config):
    """Convert config to readable string."""
    parts = []
    abbrev = {
        'cluster_specific_beta': 'β',
        'spatial_effect': 'φ', 
        'cluster_specific_sigma': 'σ',
        'offset': 'off',
        'cluster_specific_alpha': 'α'
    }
    for key, val in config.items():
        parts.append(f"{abbrev[key]}={'✓' if val else '✗'}")
    return " ".join(parts)


def config_to_short_string(config):
    """Convert config to short string for filenames."""
    bits = ""
    for key in ['cluster_specific_beta', 'spatial_effect', 'cluster_specific_sigma', 'offset', 'cluster_specific_alpha']:
        bits += "1" if config[key] else "0"
    return bits


# =============================================================================
# PSM PLOTTING
# =============================================================================

def plot_psm(result, save_path=None, figsize=(10, 10)):
    """Plot PSM heatmap for a single model."""
    if not result['success']:
        print(f"Model {result['config_id']} failed, skipping PSM plot")
        return
    
    psm = result['psm']
    order = order_by_hclust(psm)
    psm_sorted = psm[np.ix_(order, order)]
    
    plt.figure(figsize=figsize)
    sns.heatmap(psm_sorted, center=0.5, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.title(f"PSM - Model {result['config_id']}\n{config_to_string(result['config'])}\nBinder Loss: {result['binder_loss']:.2f}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_all_psm_grid(results, save_path=None, ncols=8):
    """Plot all PSM heatmaps in a grid."""
    successful = [r for r in results if r['success']]
    n = len(successful)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.flatten() if n > 1 else [axes]
    
    for idx, result in enumerate(successful):
        ax = axes[idx]
        psm = result['psm']
        order = order_by_hclust(psm)
        psm_sorted = psm[np.ix_(order, order)]
        
        im = ax.imshow(psm_sorted, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title(f"M{result['config_id']}\nL={result['binder_loss']:.0f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for idx in range(len(successful), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Posterior Similarity Matrices - All Models", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# MAP PLOTTING
# =============================================================================

def plot_map(result, gdf, save_path=None, figsize=(12, 10)):
    """Plot geographic map with cluster assignments for a single model."""
    if not result['success']:
        print(f"Model {result['config_id']} failed, skipping map plot")
        return
    
    labels = result['labels']
    gdf_plot = gdf.copy()
    gdf_plot['cluster'] = labels
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf_plot.plot(column='cluster', ax=ax, legend=True, 
                  cmap='Set2', edgecolor='black', linewidth=0.3,
                  legend_kwds={'title': 'Cluster', 'loc': 'upper right'})
    ax.set_title(f"Model {result['config_id']}: {config_to_string(result['config'])}\nBinder Loss: {result['binder_loss']:.2f}, Clusters: {result['n_clusters']}")
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_all_maps_grid(results, gdf, save_path=None, ncols=8):
    """Plot all maps in a grid."""
    successful = [r for r in results if r['success']]
    n = len(successful)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.flatten() if n > 1 else [axes]
    
    for idx, result in enumerate(successful):
        ax = axes[idx]
        labels = result['labels']
        gdf_plot = gdf.copy()
        gdf_plot['cluster'] = labels
        
        gdf_plot.plot(column='cluster', ax=ax, cmap='Set2', 
                      edgecolor='black', linewidth=0.1)
        ax.set_title(f"M{result['config_id']}\nL={result['binder_loss']:.0f}, C={result['n_clusters']}", fontsize=7)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(successful), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Cluster Maps - All Models", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison_best_models(results, gdf, n_best=6, save_path=None):
    """Plot maps for the n best models (lowest Binder loss)."""
    successful = sorted([r for r in results if r['success']], 
                        key=lambda x: x['binder_loss'])[:n_best]
    
    ncols = min(3, n_best)
    nrows = (n_best + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten() if n_best > 1 else [axes]
    
    for idx, result in enumerate(successful):
        ax = axes[idx]
        labels = result['labels']
        gdf_plot = gdf.copy()
        gdf_plot['cluster'] = labels
        
        gdf_plot.plot(column='cluster', ax=ax, legend=True, cmap='Set2',
                      edgecolor='black', linewidth=0.3)
        ax.set_title(f"Model {result['config_id']}: {config_to_string(result['config'])}\nBinder Loss: {result['binder_loss']:.2f}, Clusters: {result['n_clusters']}", fontsize=9)
        ax.axis('off')
    
    for idx in range(len(successful), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Top {n_best} Models (Lowest Binder Loss)", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, 'stan_models_32', 'results_all.pkl')
    geojson_path = os.path.join(base_dir, '../data/updated_data.geojson')
    output_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} model results")
    
    # Load geodata
    print("Loading geodata...")
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.reset_index(drop=True)
    
    # Print summary
    successful = [r for r in results if r['success']]
    print(f"\nSuccessful models: {len(successful)}/{len(results)}")
    
    if len(successful) == 0:
        print("No successful models to plot!")
        return
    
    # Sort by Binder loss
    successful_sorted = sorted(successful, key=lambda x: x['binder_loss'])
    print("\nTop 5 models by Binder loss:")
    for r in successful_sorted[:5]:
        print(f"  Model {r['config_id']}: {config_to_string(r['config'])} - Loss: {r['binder_loss']:.2f}, Clusters: {r['n_clusters']}")
    
    # Plot options
    print("\n" + "="*50)
    print("PLOTTING OPTIONS")
    print("="*50)
    print("1. Plot PSM grid (all models)")
    print("2. Plot map grid (all models)")
    print("3. Plot top 6 best models comparison")
    print("4. Plot single model PSM (enter model ID)")
    print("5. Plot single model map (enter model ID)")
    print("6. Plot all and save to files")
    print("0. Exit")
    
    while True:
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            plot_all_psm_grid(results)
        elif choice == '2':
            plot_all_maps_grid(results, gdf)
        elif choice == '3':
            plot_comparison_best_models(results, gdf, n_best=6)
        elif choice == '4':
            model_id = int(input("Enter model ID (1-32): "))
            result = next((r for r in results if r['config_id'] == model_id), None)
            if result:
                plot_psm(result)
            else:
                print(f"Model {model_id} not found")
        elif choice == '5':
            model_id = int(input("Enter model ID (1-32): "))
            result = next((r for r in results if r['config_id'] == model_id), None)
            if result:
                plot_map(result, gdf)
            else:
                print(f"Model {model_id} not found")
        elif choice == '6':
            print("\nSaving all plots...")
            plot_all_psm_grid(results, save_path=os.path.join(output_dir, 'all_psm_grid.png'))
            plot_all_maps_grid(results, gdf, save_path=os.path.join(output_dir, 'all_maps_grid.png'))
            plot_comparison_best_models(results, gdf, n_best=6, 
                                        save_path=os.path.join(output_dir, 'top6_comparison.png'))
            
            # Save individual plots for each successful model
            for result in successful:
                model_id = result['config_id']
                config_str = config_to_short_string(result['config'])
                plot_psm(result, save_path=os.path.join(output_dir, f'psm_model_{model_id}_{config_str}.png'))
                plot_map(result, gdf, save_path=os.path.join(output_dir, f'map_model_{model_id}_{config_str}.png'))
                plt.close('all')
            
            print(f"\nAll plots saved to: {output_dir}")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()

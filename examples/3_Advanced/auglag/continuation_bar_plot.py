import numpy as np
import os
import glob
import matplotlib.pyplot as plt
""" This script plots the data for Figure 1 (bar plot figure) of the Augmented Lagrangian Paper """

def find_latest_opt_data(base_dir, run_number):
    """Finds the opt_data.npz file within a specific run directory."""
    run_dir = os.path.join(base_dir, f'run_{run_number}_no_continuation')
    if not os.path.isdir(run_dir):
        return None
    # Use glob to find the subdirectory, since its name is long and complex
    search_pattern = os.path.join(run_dir, 'qa_standard_ncoils5_*', 'opt_data_{i}.npz')
    files = glob.glob(search_pattern)
    if files:
        return files[0]
    return None

# --- Main plotting script ---

# Base directory where all the run folders are located
BASE_DIR = './output_paper/'
NUM_RUNS = 20
REFERENCE_RUN_INDEX = 20 # Use the first run (run_1) as the reference

# A dictionary to hold the final values for each metric from all runs
all_data = {
    'BdotN': [],
    'links': [],
    'lengths': [],
    'cs_distances': [],
    'cc_distances': [],
    'curvatures': [],
    'msc_curvatures': [],
    'forces': []
}
metric_keys = list(all_data.keys())

# --- Data Retrieval ---
print("--- Retrieving data from optimization runs ---")
for i in range(1, NUM_RUNS + 1):
    file_path = find_latest_opt_data(BASE_DIR, i)
    
    if file_path and os.path.exists(file_path):
        try:
            data = np.load(file_path)
            for key in metric_keys:
                if key in data:
                    all_data[key].append(data[key][-1])
                else:
                    all_data[key].append(np.nan)
        except Exception as e:
            print(f"Error processing file for run {i} ({file_path}): {e}")
            for key in metric_keys:
                all_data[key].append(np.nan)
    else:
        print(f"Warning: opt_data.npz not found for run_{i}")
        for key in metric_keys:
            all_data[key].append(np.nan)

OUT_DIR1 = '/home/gipe/auglag/simsopt/examples/3_Advanced/auglag/output_paper/qa__dynamic_ncoils5_curvature5_msc5_force0.01_flux1e-16_length30_cc0.1_cs0.15_start_order_19/'
file_path = OUT_DIR1 + 'opt_data.npz'
    
if file_path and os.path.exists(file_path):
    try:
        data = np.load(file_path)
        for key in metric_keys:
            if key in data:
                all_data[key].append(data[key][-1])
            else:
                all_data[key].append(np.nan)
    except Exception as e:
        print(f"Error processing file for run {i} ({file_path}): {e}")
        for key in metric_keys:
            all_data[key].append(np.nan)

# Extract the reference run data
reference_run_data = {key: all_data[key][REFERENCE_RUN_INDEX] for key in metric_keys}


# --- Plotting with Matplotlib ---
print("\n--- Generating Combined Static Plot for All Metrics ---")

plot_details = {
    'BdotN': {'ylabel': r'$\langle|\mathbf{B}\cdot\mathbf{n}|\rangle/\langle B \rangle$ (log scale)', 'title': 'Squared Flux'},
    'links': {'ylabel': 'Linking Number', 'title': 'Linking Number Comparison'},
    'lengths': {'ylabel': 'Total Length [m]', 'title': 'Coil Length Comparison', 'threshold': 30},
    'cs_distances': {'ylabel': 'Coil-Surface Dist. [m]', 'title': 'Coil-Surface Distance Comparison', 'threshold': 0.15},
    'cc_distances': {'ylabel': 'Coil-Coil Dist. [m]', 'title': 'Coil-Coil Distance Comparison', 'threshold': 0.1},
    'curvatures': {'ylabel': 'Max Curvature [1/m] ', 'title': 'Max Curvature Comparison', 'threshold': 5},
    'msc_curvatures': {'ylabel': 'Mean Squared Curvature [1/m]', 'title': 'MSC Comparison', 'threshold': 5},
    'forces': {'ylabel': 'Max Force [N/m] (log scale)', 'title': 'Max Force Comparison', 'threshold': 1e4}
}

# Create a figure with subplots stacked vertically
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(22, 25))
axes = axes.flatten()

for i, key in enumerate(metric_keys):
    ax = axes[i]

    metric_values = np.array(all_data[key])
    reference_value = reference_run_data[key]
    
    # Define labels for the bars
    run_labels = [f'{j}' for j in range(1, NUM_RUNS + 1)]
    x_labels = run_labels + [f'AL']
    
    # Define values for the bars
    y_values = list(metric_values)

    # Define colors to distinguish the reference bar
    bar_colors = ['#1f77b4'] * NUM_RUNS + ['#ff7f0e']

    # Add bar chart to the corresponding subplot
    ax.bar(x_labels, y_values, color=bar_colors)

    details = plot_details.get(key, {})
#    ax.set_title(details.get('title', key), fontsize=16, weight='bold')
    ax.set_ylabel(details.get('ylabel', key), fontsize=30)
    ax.tick_params(axis='x', labelrotation=0, labelsize=20)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis to log scale for the 'forces' plot
    if key == 'forces': # or 'BdotN':
        ax.set_yscale('log')

    # Add horizontal threshold line if specified
    if 'threshold' in details:
        ax.axhline(y=details['threshold'], color='r', linestyle='--', linewidth=3, label=f'Threshold ({details["threshold"]})')
        ax.legend(fontsize=30)

    if key == 'BdotN':
        ax.set_yscale('log')
        # Add text labels on top of bars          
plt.tight_layout()


# Save the static figure to a PNG file
output_filename = os.path.join(BASE_DIR, 'all_runs_static_comparison_no_continuation.png')
plt.savefig(output_filename, dpi=600)
plt.close(fig)
    
print(f"\nSaved static plot to {output_filename}")
print("\n--- All plots generated. ---")


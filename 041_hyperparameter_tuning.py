import os
import pandas as pd
import pickle
from multiprocessing import Queue, Process, cpu_count
import concurrent.futures
from sklearn.model_selection import ParameterSampler
import math
from randomized_search_cv import randomized_search_cv

# Define the directories containing the .pkl files
data_directory = 'feature_selection'

# Initialize an empty dictionary to store the dataframes
dfs = {}

# Load all .pkl files starting with 'train' from the subfolder 'feature_selection'
for root, _, files in os.walk(data_directory):
    for file in files:
        if file.endswith('.pkl') and file.startswith('train'):
            df_name = os.path.splitext(file)[0]
            dfs[df_name] = pd.read_pickle(os.path.join(root, file))

print(f"Loaded {len(dfs)} dataframes.")

# Define the parameter grid
param_grid = {
    'dataframe_name': list(dfs.keys()),
    'sample_size': [500],
    'nr_clus': list(range(1, 21)),
    'clustering_method': ["fcm", "fcm_binary", "fst-pso", "gk", "gmm"],
    'm': [1.5, 2.0, 2.5],  # For fcm, fcm_binary, fst-pso, gk
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # For gmm
    'mf_shape': ["gauss", "gauss2", "invgauss", "sigmoid", "invsigmoid", "trimf", "trapmf"],
    'consequent_method': ["normalized_means", "global_LSE", "local_LSE"],
    'merge_threshold': [1.0],
    'random_state': [2024]
}

# Sample 10 hyperparameter combinations from the grid
param_sampler = list(ParameterSampler(param_grid, n_iter=50, random_state=81))

print(f"Selected {len(param_sampler)} hyperparameter combinations.")

# Print the number of CPU cores
num_cores = cpu_count()
print("############################################")
print(f"Found {num_cores} CPU cores.")
print("############################################")

# Worker function with timeout for each task
def worker(task_queue, results_queue, dfs):
    while not task_queue.empty():
        try:
            params = task_queue.get_nowait()
            df_name = params['dataframe_name']
            df = dfs[df_name]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(randomized_search_cv, df, params)
                try:
                    accuracy, auc = future.result(timeout=1800)
                    results_queue.put({'dataframe': df_name, 'params': params, 'accuracy': accuracy, 'auc': auc})
                except concurrent.futures.TimeoutError:
                    results_queue.put({'dataframe': df_name, 'params': params, 'accuracy': None, 'auc': None, 'error': 'Timeout'})
                    
        except Exception as e:
            results_queue.put({'dataframe': df_name, 'params': params, 'accuracy': None, 'auc': None, 'error': str(e)})
            continue

# Function to run the parallelized search
def run_parallel_search(dfs, param_sampler):
    task_queue = Queue()
    results_queue = Queue()

    for params in param_sampler:
        task_queue.put(params)

    processes = []
    for i in range(num_cores):
        p = Process(target=worker, args=(task_queue, results_queue, dfs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    results_df = pd.DataFrame(results)
    results_df_filtered = results_df.dropna(subset=['auc'])
    results_df_sorted = results_df_filtered.sort_values(by='auc', ascending=False)

    return results_df, results_df_sorted

if __name__ == "__main__":
    results_df, results_df_sorted = run_parallel_search(dfs, param_sampler)

    # Define the output folder
    output_folder = 'model_training'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save results
    results_csv = os.path.join(output_folder, 'results_hyperparameter_tuning.csv')
    results_csv_sorted = os.path.join(output_folder, 'results_hyperparameter_tuning_without_errors.csv')
    best_params_pkl = os.path.join(output_folder, 'best_hyperparameters.pkl')

    results_df.to_csv(results_csv, index=False)
    results_df_sorted.to_csv(results_csv_sorted, index=False)
    results_df_sorted.to_pickle(best_params_pkl)

    print(f"Saved results to {output_folder}.")
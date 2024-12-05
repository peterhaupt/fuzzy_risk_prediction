import pandas as pd

# Import the experiment runner
from run_experiment import run_experiment

def randomized_search_cv(df, params):
    """
    Function to run a single experiment based on the given dataframe and parameter set.

    Parameters:
    - df: The dataframe containing the dataset.
    - params: Dictionary with parameters for the experiment.

    Returns:
    - accuracy: The accuracy of the model on the test set.
    - auc: The AUC score of the model on the test set.
    """
    try:
        # Extract the clustering method for the current iteration
        clustering_method = params.get('clustering_method', 'fcm')
        
        # Set default values for m and covariance_type
        m_value = params.get('m', 2)  # Default fuzziness coefficient for fuzzy methods
        covariance_type = params.get('covariance_type', 'full')  # Default for GMM

        # Run the experiment using the imported run_experiment function (from your existing code)
        accuracy, auc = run_experiment(df,
                                       sample_size=params.get('sample_size', None),
                                       nr_clus=params.get('nr_clus', 10),
                                       clustering_method=clustering_method,
                                       mf_shape=params.get('mf_shape', 'gauss'),
                                       consequent_method=params.get('consequent_method', 'local_LSE'),
                                       merge_threshold=params.get('merge_threshold', 1.0),
                                       random_state=params.get('random_state', 81),
                                       m=m_value if clustering_method in ['fcm', 'fcm_binary', 'fst-pso', 'gk'] else None,
                                       covariance_type=covariance_type if clustering_method == 'gmm' else None)

        return accuracy, auc

    except Exception as e:
        print(f"Experiment failed with error: {e}")
        return None, None
import pandas as pd
import numpy as np
from pyfume import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Define a function to run the experiment with customizable parameters
def run_experiment(df, sample_size=500, nr_clus=10, clustering_method='fcm', mf_shape='gauss', consequent_method="normalized_means", merge_threshold=1.0, random_state=81):
    """
    Run an experiment with the given parameters.

    Parameters:
    - df: The dataframe containing the dataset with 'target' as the label column.
    - sample_size: If provided, the dataframe will be sampled to this size.
    - nr_clus: Number of clusters for the clustering and the zero order TS model.
    - clustering_method: The clustering method to use ('fcm' or other).
    - mf_shape: Membership function shape for the antecedent estimator ('gauss', 'triangular', etc.).
    - consequent_method: Method for consequent estimation ('local_LSE' or 'suglms').
    - merge_threshold: Threshold for merging membership functions.
    - random_state: Random state for reproducibility.

    Returns:
    - accuracy: The accuracy of the model on the test set.
    - auc: The AUC score of the model on the test set.
    """

    try:
        # Sample the dataframe if sample_size is provided
        if sample_size:
            df = df.sample(n=sample_size, random_state=random_state)

        # Split the data into features and target
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Extract variable names except the target
        variable_names = df.columns[:-1]

        # Perform an 80/20 train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Convert x_train, x_test, y_train, and y_test to NumPy arrays
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        # Cluster the training data (in input-output space)
        cl = Clustering.Clusterer(x_train=x_train, y_train=y_train, nr_clus=nr_clus)
        cluster_centers, partition_matrix, _ = cl.cluster(method=clustering_method)

        # Estimate the membership functions of the system
        ae = AntecedentEstimator(x_train=x_train, partition_matrix=partition_matrix)
        antecedent_parameters = ae.determineMF(mf_shape=mf_shape, merge_threshold=merge_threshold)

        # Calculate the firing strength of each rule for each data instance
        fsc = FireStrengthCalculator(antecedent_parameters=antecedent_parameters, nr_clus=nr_clus, variable_names=variable_names)
        firing_strengths = fsc.calculate_fire_strength(data=x_train)

        # Estimate the parameters of the consequent functions
        ce = ConsequentEstimator(x_train=x_train, y_train=y_train, firing_strengths=firing_strengths)
        consequent_parameters = ce.zero_order(method=consequent_method)

        # Build a zero-order Takagi-Sugeno model using Simpful
        simpbuilder = SugenoFISBuilder(
            antecedent_sets=antecedent_parameters, 
            consequent_parameters=consequent_parameters, 
            variable_names=variable_names, 
            model_order='zero',
            extreme_values=None, 
            save_simpful_code=False
        )
        model = simpbuilder.get_model()

        # Create a tester object to evaluate the model on the test data
        test = SugenoFISTester(model=model, test_data=x_test, variable_names=variable_names, golden_standard=y_test)

        # Predict probabilities for the test data
        y_pred_proba = test.predict()

        # Extract first element of the tuple y_pred_proba which is the label (second element is the error)
        y_pred_proba = y_pred_proba[0]

        # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate accuracy and AUC
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        return accuracy, auc

    except Exception as e:
        print(f"Error occurred during experiment: {str(e)}")
        return None, None
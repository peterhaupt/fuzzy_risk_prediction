import pandas as pd
import numpy as np
from run_experiment_RFE import run_experiment
import concurrent.futures
import copy
import os

def evaluate_feature_removal(df, selected_features, feature, target_column):
    """
    Evaluate the AUC score after removing a specific feature.
    
    Parameters:
    - df: The dataframe containing the dataset.
    - selected_features: List of currently selected feature names.
    - feature: The feature to remove for this evaluation.
    - target_column: The name of the target column.
    
    Returns:
    - feature: The feature that was removed.
    - auc: The AUC score after removing the feature.
    """
    # Create a new feature list without the current feature
    features_to_test = [f for f in selected_features if f != feature]
    
    # Create a new dataframe with only the selected features and the target
    df_subset = df[features_to_test + [target_column]]
    
    # Run the experiment and get the AUC score
    _, auc = run_experiment(df_subset)
    
    return feature, auc

def feature_selection_process(df, initial_features, target_column, min_features=10, max_workers=None, pkl_filename="selected_features.pkl"):
    """
    Perform feature elimination based on AUC scores, removing one feature at a time.
    It will also save the selected features after each round to a pickle file.
    
    Parameters:
    - df: The dataframe containing the dataset.
    - initial_features: List of feature names to start with.
    - target_column: The name of the target column.
    - min_features: The minimum number of features to retain.
    - max_workers: Number of workers (cores) to use for parallel processing.
    - pkl_filename: Filename to save/load the selected features to/from a pickle file.
    
    Returns:
    - selected_features: The final list of most important features.
    - auc_values: The AUC values after each round.
    """
    
    # Check if the pickle file exists and load previously selected features
    if os.path.exists(pkl_filename):
        print(f"Resuming from saved state: Loading features from {pkl_filename}")
        selected_features = pd.read_pickle(pkl_filename).tolist()
    else:
        selected_features = initial_features.copy()

    auc_values = []  # To store the highest AUC after each round

    # Loop until we have the desired number of features
    while len(selected_features) > min_features:
        feature_auc_dict = {}

        # Use concurrent.futures for parallel feature evaluation
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(evaluate_feature_removal, df, selected_features, feature, target_column)
                for feature in selected_features
            ]
            
            # Collect results as they are completed
            for future in concurrent.futures.as_completed(futures):
                feature, auc = future.result()
                feature_auc_dict[feature] = auc
        
        # Find the feature whose removal gives the highest AUC
        feature_to_remove = max(feature_auc_dict, key=feature_auc_dict.get)
        highest_auc = feature_auc_dict[feature_to_remove]  # Get the highest AUC for this round
        auc_values.append(highest_auc)  # Store the highest AUC
        
        # Remove that feature from the selected features list
        selected_features.remove(feature_to_remove)
        
        # Print the removed feature and the current list of features
        print(f"Removed feature: {feature_to_remove}, Highest AUC: {highest_auc}")
        print(f"Remaining features: {len(selected_features)}")

        # Save the current state of selected features to the pickle file
        pd.Series(selected_features).to_pickle(pkl_filename)
        print(f"Saved current selected features to {pkl_filename}")
    
    return selected_features, auc_values

def select_features_from_datasets(train_df, test_df, selected_features, target_column='target', special_columns=None):
    # Select features + target column for train dataset
    train_df_selected = train_df[selected_features.tolist() + [target_column]].copy()
    
    # Select features + target column for test dataset
    test_df_selected = test_df[selected_features.tolist() + [target_column]].copy()
    
    # If special columns exist, add them to the test dataset
    if special_columns is not None:
        special_columns_in_test = [col for col in special_columns if col in test_df.columns]
        test_df_selected = pd.concat([test_df_selected, test_df[special_columns_in_test]], axis=1)
    
    return train_df_selected, test_df_selected

if __name__ == "__main__":
    # Load dataset
    df = pd.read_pickle("feature_selection/train_df_all_features_20.pkl")

    # Set initial parameters
    initial_features = df.drop(columns=['target']).columns.tolist()  # List of initial features
    target_column = 'target'
    min_features = 10  # Desired number of features
    pkl_filename = "feature_selection/selected_features_RFE.pkl"  # Filename to save progress

    # Run the feature selection process using all available CPU cores
    selected_features, auc_values = feature_selection_process(df, initial_features, target_column, min_features, max_workers=None, pkl_filename=pkl_filename)

    # Print the final list of selected features
    print("Final selected features:", selected_features)
    
    # Print the highest AUC values after each round
    print("Highest AUC values after each round:", auc_values)

    # Store the selected features in a CSV file
    pd.DataFrame(selected_features, columns=["Selected Features"]).to_csv("feature_selection/selected_features.csv", index=False)

    # Store the AUC values in a CSV file
    pd.DataFrame(auc_values, columns=["Highest AUC"]).to_csv("feature_selection/highest_auc_values.csv", index=False)

    ########
    # create dataset with the selected feature and store it in a pickle file

    # Load the test and train datasets
    train_df_all = pd.read_pickle("data/all_late_depression_train_encoded_500.pkl")
    test_df_all = pd.read_pickle("data/all_late_depression_test_encoded_500.pkl")

    # List of special columns only available in the test dataset
    special_columns = ['eid', 'p130894', 'p130895', 'p53_i0']

    # convert list of selected features to pandas dataframe
    RFE_selected_features = pd.Series(selected_features)

    # For the all dataset (drop_first=False, with _features_10_RFE suffix)
    train_df_all_features_10_RFE, test_df_all_features_10_RFE = select_features_from_datasets(
        train_df_all, test_df_all, RFE_selected_features, target_column='target', special_columns=special_columns)

    # Print shapes to confirm the new datasets
    print("train_df_all_features_10_RFE shape (drop_first=False):", train_df_all_features_10_RFE.shape)
    print("test_df_all_features_10_RFE shape (drop_first=False):", test_df_all_features_10_RFE.shape)

    # Save the new datasets with the suffix '_features_10_RFE'
    train_df_all_features_10_RFE.to_pickle("feature_selection/train_df_all_features_10_RFE.pkl")
    test_df_all_features_10_RFE.to_pickle("feature_selection/test_df_all_features_10_RFE.pkl")
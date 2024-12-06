# A Fuzzy Risk Prediction Model for Late-Life Depression

This repository contains the code of the master thesis of Peter Haupt without the confidential data from the UK Biobank. The code should be used in the following order.

## Clone repository

Clone this repository with the following command in the terminal.

`git clone https://github.com/peterhaupt/fuzzy_risk_prediction`

## Sample Data

The folders data, feature_selection, figures, model_evaluation, and model_training already contain sample data. If you want to run the whole modelling process from scratch, run this script to delete locally all the sample data.

`bash 000_delete_sample_data.sh`

## Create Virtual Environment and Install Necessary Packages

**It is highly recommended to run the code in a virtual environment. The code has been tested on the following Python versions 3.9.20, 3.11.9, and 3.11.10. This code and especially pyFUME is not running on Python 3.12 or higher.**

**Please ensure that no previous version of pyFUME is installed in the virtual environment. The most robust solution is to run this code in a newly created virtual environment. This is due to the fact that pyFUME also requires specific older versions of Numpy and Pandas.**

The required packages should only be installed with the requirements.txt file. This means to run the following command from the within the cloned folder of the repository in the terminal:

`pip install -r requirements.txt`

## Create Synthetical Train and Test Dataset

In the first step synthetical data is created. This is required because the UK Biobank data is confidential.

Run the Python script `010_generate_synthetical_data.py`.

## Data Analysis and Cleaning

This folder contains all code that is required to perform the methods that are described in the section data cleaning of the chapter experimental design of the thesis.

Run the Jupyter notebook `020_data_analysis_and_cleaning.ipynb`.

## Feature Selection

This folder contains all code that is required to perform the feature selection steps that are described in the section feature selection of the chapter experimental design of the thesis. The feature selection methods mutual information and logistic regression with LASSO are calculated in the Jupyter notebook `031_feature_selection.ipynb`and the recursive feature elimination (RFE) is performed with the Python script `032_RFE.py`. RFE is computationally expensive and takes a few minutes to run. Please run them in the following order.

1. `031_feature_selection.ipynb`
2. `032_RFE.py`

## Model Training
This folder contains all code that is required to perform the model training.

First, hyperparameter tuning is performed to identify the best performing hyperparameters for the fuzzy model. Please run this Python script for the hyperparameter tuning. This might take around ten minutes for 50 hyperparameter combinations and a sample size of 500.

`041_hyperparameter_tuning.py`

The implementation of FST-PSO clustering relies on an external package, which does not provide functionality to fix randomness using a random seed.

Second, the threshold for the binary classification is tuned. Please run the following Jupyter notebook to tune the decision threshold.

`042_threshold_tuning.ipynb`

Third, a cross-validation is performed to evaluate the stability of the model performance. Run the following Jupyter notebook to perform the cross-validation.

`043_cross_validation.ipynb`

Fourth, the final model is trained on the whole training dataset. Run the following Jupyter notebook to train the final model.

`044_training_final_model.ipynb`

## Model Evaluation

The final model is evaluated in three main steps. First, the model is evaluated on the unseen test data. Run the following Jupyter notebook to evaluate the model on the unseen test data.

`051_model_evaluation.ipynb`

Second, the performance on the unseen test data is compared to a baseline logistic regression model. Run the following Jupyter notebook to evaluate the performance of a logistic regression model on the same data.

`052_baseline_logistic_regression.ipynb`

Third, the performance on the unseen test data is also compared to a baseline random forest model. Run the following Jupyter notebook to evaluate the performance of a random forest model on the same data.

`053_baseline_random_forest.ipynb`

## Create Figures

The figures of the master thesis can be as well created with a Jupyter notebook. Run the following Jupyter notebook to create the figures.

`060_create_figures.ipynb`
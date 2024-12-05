# A Fuzzy Risk Prediction Model for Late-Life Depression

This repository contains the code of the master thesis of Peter Haupt without the confidential data from the UK Biobank. The code should be used in the following order.

## Clone repository

Clone this repository with the following command in the terminal.

`git clone https://github.com/peterhaupt/fuzzy_risk_prediction`

## Create Virtual Environment and Install Necessary Packages

It is highly recommended to run the code in a virtual environment. The code has been tested on the following Python versions 3.9.20, 3.11.9, and 3.13.0.

**Please ensure that no previous version of pyFUME is installed in the virtual environment. The most robust solution is to run this code in a newly created virtual environment. This due to the fact that pyFUME requires also specific older versions of Numpy and Pandas.**

The required packages should only be installed with the requirements.txt file. This means to run the following command from the within the cloned folder of the repository in the terminal:

`pip install -r requirements.txt`

## Create Synthetical Train and Test Dataset

In the first step synthetical data is created. This is required because the UK Biobank data is confidential.

Run the Python script `01_generate_synthetical_data.py`.

## Data Analysis and Cleaning

This folder contains all code that is required to perform the methods that are described in the section data cleaning of the chapter experimental design of the thesis.

Run the Jupyter notebook `02_data_analysis_and_cleaning.ipynb`.

## Feature Selection

This folder contains all code that is required to perform the feature selection steps that are described in the section feature selection of the chapter experimental design of the thesis. The feature selection methods mutual information and logistic regression with LASSO are calculated in the Jupyter notebook `031_feature_selection.ipynb`and the recursive feature elimination (RFE) is performed with the Python script `032_RFE.py`. Please run them in the following order.

1. `031_feature_selection.ipynb`
2. `032_RFE.py`

## Model Training
This folder contains all code that is required to perform the model training.

First, hyperparameter tuning is performed to identify the best performing hyperparameters for the fuzzy model. Please run this Python script for the hyperparameter tuning. This might take around ten minutes for 50 hyperparameter combinations and a sample size of 500.

`041_hyperparameter_tuning.py`

The implementation of FST-PSO clustering relies on an external package, which does not provide functionality to fix randomness using a random seed.

Second, the threshold for the binary classification is tuned. Please run the following Jupyter notebook to tune the decision threshold.





Third, a cross-validation is performed to evaluate the stability of the model performance.





## Sample Data

The folders XXX contain sample data / processed information like selected features. If you want to run everything from scratch simply delete all subfolders.




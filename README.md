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



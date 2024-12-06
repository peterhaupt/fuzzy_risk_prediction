import pandas as pd
import numpy as np
import pickle
from faker import Faker
from datetime import timedelta
import os

# Load column_list from pkl file
with open("column_list.pkl", "rb") as file:
    column_list = pickle.load(file)

# Initialize Faker for generating synthetic data
fake = Faker()

# Define a function to generate random dates within a range
def random_date(start, end, rows):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    delta = (end_date - start_date).days
    return [start_date + timedelta(days=np.random.randint(0, delta)) for _ in range(rows)]

# Define a function to create correlated data
def generate_correlated_data(target, correlation, rows):
    noise = np.random.normal(0, 1, rows)
    return target * correlation + noise * np.sqrt(1 - correlation**2)

# Define a function to generate synthetic data based on column types
def generate_synthetic_data(column_list, rows=500):
    data = {}

    # Create "target" first as binary
    data["target"] = np.random.choice([0, 1], rows)

    # Remove "target" from column_list after creating it
    column_list = [(col, dtype) for col, dtype in column_list if col != "target"]

    for col, dtype in column_list:
        if col == "eid":
            data[col] = [f"ID_{i}" for i in range(1, rows + 1)]
        elif col == "birth_date":
            data[col] = [fake.date_of_birth(minimum_age=40, maximum_age=84) for _ in range(rows)]
        elif col == "age_at_baseline":
            data[col] = np.random.randint(40, 71, rows)
        elif col == "p130894":
            data[col] = random_date("2006-07-01", "2022-10-31", rows)
        elif col == "p53_i0":
            data[col] = random_date("2006-03-14", "2010-09-30", rows)
        elif dtype == "int64":
            # Generate correlated data for every fifth int64 column
            if len(data) % 5 == 0:
                data[col] = generate_correlated_data(data["target"], 0.1, rows).astype(int)
            else:
                data[col] = np.random.randint(1, 101, rows)
        elif dtype == "float64":
            # Generate correlated data for every fifth float64 column
            if len(data) % 5 == 0:
                data[col] = generate_correlated_data(data["target"], 0.1, rows)
            else:
                data[col] = np.random.random(rows) * 100
        elif dtype == "O":
            # Set different categories for every fiftieth categorical column
            if len(data) % 50 == 0:
                data[col] = [
                    "Category_B" if t == 1 and np.random.rand() > 0.95 else
                    "Category_E" if t == 0 and np.random.rand() > 0.95 else
                    np.random.choice(["Category_A", "Category_C", "Category_D"])
                    for t in data["target"]
                ]
            else:
                # Add more diversity and randomness to the non-predictive categories
                data[col] = np.random.choice(
                    ["Category_A", "Category_B", "Category_C", "Category_D", "Category_E"], 
                    rows, 
                    p=[0.2, 0.2, 0.2, 0.2, 0.2]  # Equal probability; adjust as needed for skew
                )
        else:
            data[col] = [None] * rows

    # Reorder "target" column to the end
    df = pd.DataFrame(data)
    cols = [col for col in df.columns if col != "target"] + ["target"]
    return df[cols]

# Generate two synthetic dataframes
train_df = generate_synthetic_data(column_list)
test_df = generate_synthetic_data(column_list)

# Create the "data" subfolder if it doesn't already exist
os.makedirs("data", exist_ok=True)

# Save the dataframes as pkl files, overwriting if they already exist
train_df.to_pickle("data/train.pkl")
test_df.to_pickle("data/test.pkl")

print("Synthetic dataframes created and saved as 'train.pkl' and 'test.pkl'.")
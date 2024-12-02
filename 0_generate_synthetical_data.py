import pandas as pd
import numpy as np
import pickle
from faker import Faker

# Load column_list from pkl file
with open("data/column_list.pkl", "rb") as file:
    column_list = pickle.load(file)

# Initialize Faker for generating synthetic data
fake = Faker()

# Define a function to generate synthetic data based on column types
def generate_synthetic_data(column_list, rows=500):
    data = {}
    for col, dtype in column_list:
        if col == "eid":
            data[col] = [f"ID_{i}" for i in range(1, rows + 1)]
        elif col == "birth_date":
            data[col] = [fake.date_of_birth(minimum_age=40, maximum_age=84) for _ in range(rows)]
        elif col == "age_at_baseline":
            data[col] = np.random.randint(40, 71, rows)
        elif col == "target":
            data[col] = np.random.choice([0, 1], rows)
        elif dtype == "int64":
            data[col] = np.random.randint(1, 101, rows)
        elif dtype == "float64":
            data[col] = np.random.random(rows) * 100
        elif dtype == "O":
            data[col] = np.random.choice(["Category_A", "Category_B", "Category_C", "Category_D", "Category_E"], rows)
        else:
            data[col] = [None] * rows
    return pd.DataFrame(data)

# Generate two synthetic dataframes
train_df = generate_synthetic_data(column_list)
test_df = generate_synthetic_data(column_list)

# Save the dataframes as pkl files
train_df.to_pickle("data/train.pkl")
test_df.to_pickle("data/test.pkl")

print("Synthetic dataframes created and saved as 'train.pkl' and 'test.pkl'.")
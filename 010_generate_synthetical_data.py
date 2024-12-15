import pandas as pd
import numpy as np
import pickle
from faker import Faker
from datetime import timedelta
import os
import argparse

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
def generate_synthetic_data(column_list, rows=500, \
                            percent_correlated_numerical=0.2,\
                            percent_correlated_categorical=0.05, \
                            strength_numerical_correlation=0.2,\
                            noise_correlated_categorical_columns=0.8):
    data = {}

    # Create "target" first as binary
    data["target"] = np.random.choice([0, 1], rows)

    # Remove "target" from column_list after creating it
    column_list = [(col, dtype) for col, dtype in column_list if col != "target"]

    fake = Faker()

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
            if len(data) % int(1 / percent_correlated_numerical) == 0:
                data[col] = generate_correlated_data(data["target"], strength_numerical_correlation, rows).astype(int)
            else:
                data[col] = np.random.randint(1, 101, rows)
        elif dtype == "float64":
            # Generate correlated data for every fifth float64 column
            if len(data) % int(1/percent_correlated_numerical) == 0:
                data[col] = generate_correlated_data(data["target"], strength_numerical_correlation, rows)
            else:
                data[col] = np.random.random(rows) * 100
        elif dtype == "O":
            # Set different categories for every fiftieth categorical column
            if len(data) % int(1/percent_correlated_categorical) == 0:
                data[col] = [
                    "Category_B" if t == 1 and np.random.rand() > noise_correlated_categorical_columns else
                    "Category_E" if t == 0 and np.random.rand() > noise_correlated_categorical_columns else
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

if __name__ == "__main__":
    # Parse command-line arguments for sample size
    parser = argparse.ArgumentParser(description="Generate synthetic dataframes.")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of rows in each generated dataframe.")
    parser.add_argument("--percent_correlated_numerical", type=float, default=0.20, help="Percentage of correlated numerical columns.")
    parser.add_argument("--percent_correlated_categorical", type=float, default=0.05, help="Percentage of correlated categorical columns.")
    parser.add_argument("--strength_numerical_correlation", type=float, default=0.20, help="Strength of numerical correlation.")
    parser.add_argument("--noise_correlated_categorical_columns", type=float, default=0.8, help="Noise for correlated categorical columns.")
    args = parser.parse_args()

    # Load column_list from pkl file
    with open("column_list.pkl", "rb") as file:
        column_list = pickle.load(file)

    # Generate two synthetic dataframes
    train_df = generate_synthetic_data(column_list, rows=args.sample_size, \
                                       percent_correlated_numerical = args.percent_correlated_numerical,\
                                        percent_correlated_categorical=args.percent_correlated_categorical, \
                                        strength_numerical_correlation=args.strength_numerical_correlation, \
                                            noise_correlated_categorical_columns=args.noise_correlated_categorical_columns)
    test_df = generate_synthetic_data(column_list, rows=args.sample_size, \
                                       percent_correlated_numerical = args.percent_correlated_numerical,\
                                        percent_correlated_categorical=args.percent_correlated_categorical, \
                                        strength_numerical_correlation=args.strength_numerical_correlation, \
                                            noise_correlated_categorical_columns=args.noise_correlated_categorical_columns)
    # Create the "data" subfolder if it doesn't already exist
    os.makedirs("data", exist_ok=True)

    # Save the dataframes as pkl files, overwriting if they already exist
    train_df.to_pickle("data/train.pkl")
    test_df.to_pickle("data/test.pkl")

    # Save the sample_size as a pkl file in the "data" subfolder
    with open("data/sample_size.pkl", "wb") as size_file:
        pickle.dump(args.sample_size, size_file)

    print(f"Synthetic dataframes with {args.sample_size} rows each created and saved as 'train.pkl' and 'test.pkl'.\n"
      f"Arguments used: percent_correlated_numerical={args.percent_correlated_numerical}, "
      f"percent_correlated_categorical={args.percent_correlated_categorical}, "
      f"strength_numerical_correlation={args.strength_numerical_correlation}, "
      f"noise_correlated_categorical_columns={args.noise_correlated_categorical_columns}.")
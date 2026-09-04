import pandas as pd
import numpy as np
import os

# Get the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

data = {
    "date": pd.date_range(start="2026-01-25", periods=12),
    "unit_type": ["electricity", "electricity", "gas", "gas", "water", "electricity", "gas", "gas", "water", "electricity", "gas", "water"],
    "usage": [12, 45, 30, 75, 5, 60, 33, 90, 21, 15, np.nan, 40]
}

# create a dataset csv with custom data with date and units and usage

dataset = pd.DataFrame(data)
csv_path = os.path.join(SCRIPT_DIR, "utility_usage.csv")
dataset.to_csv(csv_path, index=False)

# Load the csv
dataset = pd.read_csv(csv_path, parse_dates=["date"])
print(dataset.head())
print("\nShape of the dataset: ", dataset.shape)
print(dataset.info())
print(dataset.describe())

# indexing the dataset
print("\nRows 0-2\n", dataset.loc[0:2])
print("\nCol 1-2 (iloc):\n", dataset.iloc[:, 1:3])

# PREPROCESSING: handling the missing values
print("\nMissing values:\n", dataset.isnull().sum()) # isnull() checks the rows with null values # sum() provides all rows together
dataset["unit_type"] = dataset["unit_type"].fillna("unknown") # mark the null valued rows with unknown
dataset["usage"] = dataset["usage"].fillna(dataset["usage"].mean()) # Replace it with the mean values of the dataset for the usage column
print("\nAfter cleaning:\n", dataset.isnull().sum()) # print the dataset after handlign the missing values

# group by aggregation
avg_type = dataset.groupby("unit_type")["usage"].mean()
print("\nAverage usage per type:\n", avg_type)

# Feature Engineering
dataset["day_of_week"] = dataset["date"].dt.day_name()
dataset["is_weekend"] = dataset["date"].dt.dayofweek >= 5
dataset["high_usage"] = (dataset["usage"] > dataset["usage"].mean()).astype(int)
print("\nEngineered Features:\n", dataset.head())

# Encode categorical column like State , countries into binary values 
dataset_encoded = pd.get_dummies(dataset, columns=["unit_type"], prefix="type")
print("\nEncoded columns:\n", dataset_encoded.columns.tolist())

# normalized numeric column
dataset["usage_normalized"] = (dataset["usage"] - dataset["usage"].min()) / (dataset["usage"].max() - dataset["usage"].min())
print("\nNormalized range:\n", dataset[["usage", "usage_normalized"]].head())

# sorting and filtering the data

top_usage_days = dataset.sort_values("usage", ascending=False).head(3) # third column
print("\nTop 3 usage days: \n", top_usage_days[["date", "unit_type", "usage"]])

cleaned_csv_path = os.path.join(SCRIPT_DIR, "utility_usage_cleaned.csv")
dataset_encoded.to_csv(cleaned_csv_path, index=False)
print("\nCleaned dataset saved -> utility_usage_cleaned.csv")
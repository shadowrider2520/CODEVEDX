import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

FILE = "utility_usage.csv"

def init_file():
    if not os.path.exists(FILE):
        df = pd.DataFrame(columns=["date", "unit_type", "usage"])
        df.to_csv(FILE, index=False)

def add_record():
    date = input("Enter date [y-m-d]:")
    unit_type = input("enter unit type:(Electricity,gas/water):")
    try: # Error handling for handling the clean data passing into the dataset
        usage = float(input("enter the usage:"))
    except ValueError:
        print("Invalid format of data value.")
        return
    dataset = pd.read_csv(FILE)
    new_row = {"date":date,"unit_type":unit_type,"usage":usage} # indexing the datas according their columns in the dataset
    dataset = pd.concat([dataset,pd.DataFrame([new_row])],ignore_index=True) # adds the new row to the dataset independant of duplicate indexed values in any row
    dataset.to_csv(FILE,index=False)
    print("Records added")

def view_records():
    try:
        df = pd.read_csv(FILE)
        print(df if not df.empty else "No records found.")
    except FileNotFoundError:
        print("File not found. Run option to initialize first.")

def update_record():
    try:
        view_records()
        idx = int(input("Enter row index to update: "))
        usage = float(input("Enter new usage: "))
        df = pd.read_csv(FILE)
        df.loc[idx, "usage"] = usage
        df.to_csv(FILE, index=False)
        print("Record updated.")
    except (ValueError, IndexError, KeyError):
        print("Invalid index or input.")

def summary():
    try:
        df = pd.read_csv(FILE)
        if df.empty:
            print("No data to summarize.")
            return
        print(df.groupby("unit_type")["usage"].agg(["mean", "max", "min"]))
    except Exception as e:
        print(f"Error generating summary: {e}")

def predict_next_usage():
    try:
        dataset = pd.read_csv(FILE)
        dataset = dataset.dropna().reset_index(drop=True)
        if len(dataset) < 5:
            print("Need at least 5 records to predict.")
            return

        dataset["date"] = pd.to_datetime(dataset["date"]).map(pd.Timestamp.toordinal)

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        print("X shape:", X.shape, "y shape:", y.shape)  # debug check

        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough'
        )
        X = np.array(ct.fit_transform(X))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        results = pd.DataFrame({
            "Predicted usage":np.round(y_pred,2),
            "Actual usage":np.round(y_test,2)
        })
        print("Prediction using multiple linear regression\n")
        print(results.to_string(index=False))

    except Exception as e:
        print(f"Prediction error: {e}")

def main():
    init_file()
    while True:
        print("\n--- Utility Usage Prediction Tool ---")
        print("1. Add Record")
        print("2. View Records")
        print("3. Update Record")
        print("4. Summary")
        print("5. Predict Next Usage (ML)")
        print("6. Exit")
        choice = input("Enter choice: ")

        try:
            if choice == "1":
                add_record()
            elif choice == "2":
                view_records()
            elif choice == "3":
                update_record()
            elif choice == "4":
                summary()
            elif choice == "5":
                predict_next_usage()
            elif choice == "6":
                print("Exiting...")
                break
            else:
                print("Invalid choice, try again.")
        except Exception as e:
            print(f"Menu error: {e}")

if __name__ == "__main__":
    main()
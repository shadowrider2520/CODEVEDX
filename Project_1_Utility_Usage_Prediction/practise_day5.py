import pandas as pd
import os
FILE = "utility_usage.csv"
def init_file():
    if not os.path.exists(FILE): # check for the csv file to be existed , else initialize a new one
        dataset = pd.DataFrame(columns=["date","unit_type","usage"])
        dataset.to_csv(FILE,index=False)

# adding new record

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
    dataset = pd.read_csv(FILE)
    if dataset.empty:
        print("No records found")
    else:
        print(dataset)

def update_record():
    view_records() # fetches all records
    try:
        idx = int(input("Enter the row index[to be updated]"))
        usage = float(input("Enter new usage:"))
        dataset = pd.read_csv(FILE)
        dataset.loc[idx,"usage"] = usage # "locating the specific row in usage column using the loc function"
        dataset.to_csv(FILE,index = False) 
        print("REcord updated:")
    except (ValueError,IndexError,KeyError):
        print("Invalid input or index")

def summary():
    dataset = pd.read_csv(FILE)
    if dataset.empty:
        print("no data to summarize:")
        return
    # Summarize the whole dataset and grouping the similar elements
    print(dataset.groupby("unit_type")["usage"].agg(["mean","max","min"])) 

def main():
    init_file()
    while True:
                    print("\n--- Utility Usage Prediction Tool ---")
                    print("1. Add Record")
                    print("2. View Records")
                    print("3. Update Record")
                    print("4. Summary")
                    print("5. Exit")
                    choice = input("Enter choice: ")
            
                    if choice == "1":
                        add_record()
                    elif choice == "2":
                        view_records()
                    elif choice == "3":
                        update_record()
                    elif choice == "4":
                        summary()
                    elif choice == "5":
                        print("Exiting...")
                        break
                    else:
                        print("Invalid choice, try again.")

if __name__ == "__main__":
     main()
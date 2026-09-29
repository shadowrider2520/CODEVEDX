import pandas as pd

catalog = ["Product_A","Product_B","Product_C","Product_D"] # Iterms available

def collect_user_preferences():
    print("Rate each product from 0 to 5:")
    ratings = {}
    for product in catalog:
        while True:
            try:
                score = int(input(f"{product}:")) # ask the user to rate this product
                if 0<=score<=5:
                    ratings[product] = score
                    break
                else:
                    print('Please enter a number between 0 and 5 !')
            except ValueError:
                print("Invalid input:Please enter a number!")
    return ratings

def add_user_to_dataset(dataset,user,ratings):
    new_row = pd.DataFrame([ratings],index=[user])
    updated_dataset = pd.concat([dataset,new_row])
    return updated_dataset

existing_data = pd.DataFrame({
    "Product_A": [5, 4, 0, 1],
    "Product_B": [3, 0, 4, 5],
    "Product_C": [4, 5, 0, 0],
    "Product_D": [0, 1, 5, 4],
}, index=["Gomathi", "Mithun", "Navina", "Darshan"])
print("Existing Dataset: \n",existing_data)
new_ratings = {"Product_A":2,"Product_B":5,"Product_C":0,"Product_D":4}
existing_data = add_user_to_dataset(existing_data,"Suruli",new_ratings)
print("\nDataset after adding new user:\n",existing_data)

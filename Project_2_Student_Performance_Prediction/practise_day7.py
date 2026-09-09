import pandas as pd
import numpy as np

data = {
    "student_id": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [85,90,np.nan,60,75,95,np.nan,40,40,88],
    "marks": [78, 85, 65, np.nan, 70, 92, 55, 30, 30, 80],
    "study_hours": [3, 4, 2, 1, "3", 5, 2, 0.5, 0.5, 4],
}
dataset = pd.DataFrame(data)
dataset.to_csv("student_raw.csv",index=False)
pd.read_csv("student_raw.csv")
print("Initial shape:",dataset.shape)
print(dataset.info())
dataset.drop_duplicates(subset="student_id")   # removes the duplicate student id enrollments on student_id column
print("\nAfter Removing duplicates:",dataset.shape)
# fixing data types by converting them to int/ numeric
dataset["study_hours"] = pd.to_numeric(dataset["study_hours"],errors="coerce")
# handling the missing values 
X = dataset.iloc[:,[1,3]].values # Matrix of Features [ for Training the model ] [attendance,study_hours]
y = dataset.iloc[:,2].values # dependant variable [ What to predict ? ] [marks]

from sklearn.impute import SimpleImputer
imputer_x = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer_y = SimpleImputer(missing_values=np.nan,strategy="mean")
X = imputer_x.fit_transform(X)
y = imputer_y.fit_transform(y.reshape(-1,1)).flatten() # reshape(-1,1) transforms it to a 2d matrix , flatten() reverts it 

from sklearn.model_selection import train_test_split
X_train,X_test,y_test,y_train = train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Feature scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
dataset.to_csv("student_cleaned.csv",index=False)
print("\nSaved -> student_cleaned.csv")
print("X_train (Scaled Features):\n",X_train)
print("\nTrain (Target marks):\n",y_train)


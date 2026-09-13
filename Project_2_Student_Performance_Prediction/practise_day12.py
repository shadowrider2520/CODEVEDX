import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("student_no_missing.csv")

X = dataset[['attendance','study_hours']].values
y = dataset['marks'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluating the model
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
r2 = r2_score(y_test,y_pred) # using a inbuilt r2_score function to evaulate the model amd provide a score for its performance
meanaerror = mean_absolute_error(y_test,y_pred)
meansqerror = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"R sq Score: {r2} (closer to 1 means better fit / excellent prediction)")
print(f"MeanArithmeticError:{meanaerror}(Average prediction error in the marks)")
print(f"RootMeanSquareError:{meansqerror}")

# plotting the actual vs predicted marks graph
plt.figure(figsize=(6,5))
plt.scatter(y_test,y_pred,color='red') # plotting the scatter plot of all predicted vs real points
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'b') # perfect prediction line
plt.xlabel('Actual marks')
plt.ylabel('predicted Marks')
plt.title("Actual vs predicted Marks - closer to the red line[better]")
plt.savefig('ActualVSPredicted.png') # saves the graph as image
plt.close()

# plotting the attendance vs marks with multi regression trend
plt.figure(figsize=(6,5))
plt.scatter(dataset['attendance'],dataset['marks'],color='purple')
plt.xlabel('Attendance')
plt.ylabel('Marks')
plt.title('Attendance vs Marks')
plt.savefig('attendanceVSmarks.png')
plt.close()

# plotting the Study hours vs marks with multi regression methods
plt.figure(figsize=(6,5))
plt.scatter(dataset['study_hours'],dataset['marks'],color='orange')
plt.xlabel('Study hours')
plt.ylabel('Marks')
plt.title("StudyHours vs Marks across students")
plt.savefig('studyHours_vs_marks.png')
plt.close()

# plotting the Residuals [ errors ]
residuals = y_test - y_pred
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, color="crimson")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Predicted Marks")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot — should scatter randomly around 0")
plt.savefig("residual_plot.png")
plt.close()

print('\nThe Final visualizations have been made using the student_no_missing data which represents the accuracy of the multi linear regressiion model:actual_vs_predicted.png,attendance_vs_marks.png, study_hours_vs_marks.png, residual_plot.png')
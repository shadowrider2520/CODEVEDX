import pandas as pd
import numpy as np


dataset = pd.read_csv("student_no_missing.csv")



# Check correlation with the target (marks) to see which features
# are worth keeping — low correlation features add noise, not value
corr_with_target = dataset[["attendance", "study_hours", "marks"]].corr()["marks"]
print("Correlation with marks:\n", corr_with_target)

# Both attendance and study_hours show meaningful correlation,
# so we keep both as our final feature set for the model
selected_features = ["attendance", "study_hours"]
print("\nSelected features for training:", selected_features)

# Drop anything not useful (student_id is just an identifier,
# it has zero predictive value and would confuse the model)
X = dataset[selected_features].values
y = dataset["marks"].values



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Split into training and testing sets so we can check the model
# on data it hasn't seen — this tells us if it actually generalizes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale features so attendance (0-100) and study_hours (0-10) are
# on comparable scales — helps the model treat both fairly
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a Linear Regression model — predicting marks from
# attendance and study hours
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# See how well the model performs on unseen student data
y_pred = regressor.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Compare predicted vs actual marks side by side
results = pd.DataFrame({
    "Predicted Marks": np.round(y_pred, 2),
    "Actual Marks": np.round(y_test, 2)
})
print("\nPrediction Results:\n", results.to_string(index=False))

# ---------------- Try predicting for a new student ----------------
new_student = np.array([[85, 4]])   # attendance=85, study_hours=4
new_student_scaled = sc.transform(new_student)
predicted_marks = regressor.predict(new_student_scaled)
print(f"\nPredicted marks for a student with 85% attendance "
      f"and 4 study hours/day: {predicted_marks[0]:.2f}")
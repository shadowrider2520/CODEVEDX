import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("student_cleaned.csv")
# ---------------- DAY 8: Handling Missing Values ----------------
print("Missing values before cleanup:\n", dataset.isnull().sum())

dataset["attendance"] = dataset["attendance"].fillna(dataset["attendance"].mean())
dataset["marks"] = dataset["marks"].fillna(dataset["marks"].median())
dataset["study_hours"] = dataset["study_hours"].fillna(dataset["study_hours"].mode()[0])

print("\nMissing values after cleanup:\n", dataset.isnull().sum())

dataset.to_csv("student_no_missing.csv", index=False)
print("\nSaved -> student_no_missing.csv")

# ---------------- DAY 9: Exploratory Data Analysis ----------------
print("\nSummary stats:\n", dataset.describe())

corr = dataset[["attendance", "marks", "study_hours"]].corr()
print("\nCorrelation matrix:\n", corr)

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("How Attendance, Marks & Study Hours Relate")
plt.savefig("correlation_heatmap.png")
plt.close()

for col in ["attendance", "marks", "study_hours"]:
    plt.figure()
    sns.histplot(dataset[col], kde=True)
    plt.title(f"How {col} is Distributed Across Students")
    plt.savefig(f"{col}_distribution.png")
    plt.close()

plt.figure()
sns.scatterplot(x="study_hours", y="marks", data=dataset)
plt.title("Study Hours vs Marks — Is There a Trend?")
plt.savefig("study_vs_marks.png")
plt.close()

print("\nEDA complete — plots saved.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("student_cleaned.csv")
# ---------------- DAY 8: Handling Missing Values ----------------
# First, let's see how bad the damage is
print("Missing values before cleanup:\n", df.isnull().sum())
# Attendance is usually fairly consistent across students,
# so filling gaps with the average makes sense here
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())
# Marks can vary a lot (some students score very high/low),
# so median is safer than mean — it won't get skewed by outliers
df["marks"] = df["marks"].fillna(df["marks"].median())

# Study hours tend to cluster around common values (like 2hrs, 3hrs),
# so the most frequent value (mode) is a reasonable fill
df["study_hours"] = df["study_hours"].fillna(df["study_hours"].mode()[0])

# If a row is missing the student_id itself, we can't identify
# who it belongs to — safest to just drop that row
df = df.dropna(subset=["student_id"])

# Double-check everything is clean now
print("\nMissing values after cleanup:\n", df.isnull().sum())

# Save the cleaned version so EDA below works with good data
df.to_csv("student_no_missing.csv", index=False)
print("\nCleaned dataset saved -> student_no_missing.csv")

# ---------------- DAY 9: Exploratory Data Analysis ----------------

# Quick overview — ranges, averages, spread of each column
print("\nSummary stats:\n", df.describe())

# Correlation tells us how strongly features move together.
# A high positive value between study_hours and marks would mean
# "more study hours -> higher marks" tends to hold true
corr = df[["attendance", "marks", "study_hours"]].corr()
print("\nCorrelation matrix:\n", corr)

# Heatmap makes the correlation numbers easy to read at a glance
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("How Attendance, Marks & Study Hours Relate")
plt.savefig("correlation_heatmap.png")
plt.close()


# are most students scoring similarly, or is there a big range?
for col in ["attendance", "marks", "study_hours"]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"How {col} is Distributed Across Students")
    plt.savefig(f"{col}_distribution.png")
    plt.close()

plt.figure()
sns.scatterplot(x="study_hours", y="marks", data=df)
plt.title("Study Hours vs Marks — Is There a Trend?")
plt.savefig("study_vs_marks.png")
plt.close()

print("\nEDA complete — plots saved for the report: "
      "correlation_heatmap.png, feature distributions, study_vs_marks.png")
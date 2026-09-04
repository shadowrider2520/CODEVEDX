import numpy as np
daily_usage = [12,45,30,78,5,60,33,90,21,15] # units/day
unit_type = "electricity" 
threshold = 50   # units/day considered high

usage_arr = np.array(daily_usage) # convert readings to Numpy array
print(f"Unit type: {unit_type}")
print("Readings: ",usage_arr)
print("Shape:",usage_arr.shape," | DataType: ",usage_arr.dtype)

mean_usage = np.mean(usage_arr)
max_usage = np.max(usage_arr)
min_usage = np.min(usage_arr)
std_usage = np.std(usage_arr)
print(f"Mean: {mean_usage} | Max usage : {max_usage} | min usage: {min_usage} | Std: {std_usage}")

# Classifying the each day using the Conditonal statements

high_days,normal_days,low_days = [],[],[]

for day in range(len(usage_arr)):
    reading = usage_arr[day]
    if reading>threshold:
        high_days.append((day+1,reading))
    elif reading>=threshold*0.4:
        normal_days.append((day+1,reading))
    else:
        low_days.append((day+1,reading))
print("\nHigh usage days: ",high_days)
print("\nNormal usage days: ",normal_days)
print("\nLow usage days: ",low_days)

# Flage the anamolized Days using vectors
anomaly_mask = usage_arr > (mean_usage+std_usage) # when does the anomaly occur
anomaly_days = np.where(anomaly_mask)[0]+1 # Using numpy to detect the anomalies based on the anomaly condition # returns as array
print("\nAnomaly days (Unusually high): ",anomaly_days.tolist())

# The anomalies reshaped into a weekly Matrix [ 12 weeks x 5 Days ] from multi feature anomaly detection for ML models
usage_matrix = usage_arr.reshape(2,5)
print("\nWeekly Matrix\n",usage_matrix)
print("Week 1 total: ",usage_matrix[0].sum())
print("Week 2 total: ",usage_matrix[1].sum())

# As in ML , to prevent models from making wrong uncategorized preassumptions , Normalizaton is essential [ Conversion to Binary values ]
normalized = (usage_arr-min_usage)/(max_usage-min_usage)
print("\nNormalized readings: ",np.round(normalized,2))

# Alert system with prediction
predicted_next_day = mean_usage #initalization
if predicted_next_day > threshold:
    print(f'\nAlert: Predicted usage: {predicted_next_day} exceeds {threshold}')
else:
    print(f"\nPredicted usage {predicted_next_day} is within the Limits")




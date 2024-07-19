import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file
file_path = "/Users/yuhan/Downloads/proDataLabel.csv"
data = pd.read_csv(file_path)

# Extract the label column
labels = data['Label']

# Remove the label column, standardize only the numerical columns
features = data.drop(columns=['Label'])

# Use StandardScaler for Z-score normalization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame and add the label column
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
scaled_data['Label'] = labels

# Output the standardized data
print(scaled_data.head())

# Save the standardized data back to a CSV file if needed
output_path = "/Users/yuhan/Downloads/standardized_data.csv"
scaled_data.to_csv(output_path, index=False)

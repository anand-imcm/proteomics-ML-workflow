import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for preprocessing of data')
    parser.add_argument('-i','--csv',type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-p','--prefix',type=str, help='Output prefix')
    return parser.parse_args()

def data_scaler(inp_csv, out_csv):
    # Read the CSV file
    data = pd.read_csv(inp_csv)

    # Check if 'SampleID' column exists
    if 'SampleID' not in data.columns:
        raise ValueError("Input CSV must contain a 'SampleID' column.")

    # Extract the 'SampleID' and 'Label' columns
    sample_ids = data['SampleID']
    if 'Label' not in data.columns:
        raise ValueError("Input CSV must contain a 'Label' column.")
    labels = data['Label']

    # Remove the 'SampleID' and 'Label' columns to get only numerical features
    features = data.drop(columns=['SampleID', 'Label'])

    # Use StandardScaler for Z-score normalization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Convert the scaled features back to a DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Reattach 'SampleID' and 'Label' columns
    scaled_data = pd.concat([sample_ids, scaled_features_df, labels], axis=1)

    # Optionally, reorder columns to have 'SampleID' first, followed by scaled features, then 'Label'
    # This step ensures that the order is preserved as desired
    columns_order = ['SampleID'] + list(features.columns) + ['Label']
    scaled_data = scaled_data[columns_order]

    # Output the standardized data (optional)
    print(scaled_data.head())

    # Save the standardized data back to a CSV file
    scaled_data.to_csv(out_csv, index=False)

if __name__ == "__main__":
    args = parse_arguments()
    csv_basename = Path(args.csv).stem
    print(f"Processing file: {csv_basename}.csv")
    out_file = f"{csv_basename}_scaled.csv"  # Changed to include '_scaled' for clarity
    if args.prefix:
        out_file = f"{args.prefix}.csv"
    
    data_scaler(args.csv, out_file)
    print(f"Output generated: {out_file}")

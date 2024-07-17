import argparse
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
    scaled_data.to_csv(out_csv, index=False)

if __name__ == "__main__":
    args = parse_arguments()
    csv_basename = os.path.basename(args.csv)
    out_file = f"{csv_basename}.csv"
    if args.prefix:
        out_file = f"{args.prefix}.csv"
    
    data_scaler(args.csv, out_file)
    print ("Output generated: {out_file}")

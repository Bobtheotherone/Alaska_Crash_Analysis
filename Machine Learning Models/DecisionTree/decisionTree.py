"""
Implementation of the DecisionTree ML model
Hard coded for now with the severity mapping until
We get the LLM working to map it for us
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import argparse


severity_mapping = {
    'No Apparent Injury': 0,
    'Possible Injury': 1,
    'Suspected Minor Injury': 1,
    'Suspected Serious Injury': 2,
    'Fatal Injury (Killed)': 2
}


if __name__ == "__main__":
    #get user file (temp for now for testing)
    parser = argparse.ArgumentParser(description="Get the cleaned dataset")
    parser.add_argument("-f", "--file", required=False, help="Path to the input .csv file.")
    args = parser.parse_args()

    if args.file:
        input_file = Path(args.file)
    else:
        #prompt user for file path if not provided
        raw = input("Enter the path to the input file: ").strip().strip('"') #use raw for input cleaning
        raw = raw.replace("'", "").replace('"', '') #remove single quotes or double quotes if user added them
        input_file = Path(raw)

    if not input_file.exists():
        raise SystemExit(f"Error: The file {input_file} does not exist.") #failsafe if file not found
    if input_file.is_dir(): #directory, not file entered
        raise SystemExit(f"Error: The path {input_file} is a directory, not a file.")
    
    #load file into df
    df = pd.read_csv(input_file)
    print(f"\nLoaded: {input_file.name}   Shape: {df.shape[0]:,} × {df.shape[1]}")

    #dynamically get column names
    column_names = df.columns.tolist()
    print(f"\nDetected {len(column_names)} columns: ")
    print(column_names)

    #in datasets, column 0 is usually a unique ID, so just drop column 1
    if df.shape[1] >= 1:
        first_col_label = df.columns[0]
        df = df.drop(columns=[first_col_label])

    #reset column names
    column_names = df.columns.tolist()



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

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#Put root directory into sys so we can get DataCleaning scripts
from DataCleaning.unknown_discovery import discover_unknown_placeholders
from DataCleaning.config import UNKNOWN_STRINGS

severity_mapping = {
    'No Apparent Injury': 0,
    'Possible Injury': 1,
    'Suspected Minor Injury': 1,
    'Suspected Serious Injury': 2,
    'Fatal Injury (Killed)': 2
}

def group_severity(Crash_Severity):
    if Crash_Severity in ["No Apparent Injury", "Suspected Minor Injury"]:
        return "Low Severity"
    elif Crash_Severity in ["Possible Injury", "Suspected Serious Injury"]:
        return "Moderate Severity"
    elif Crash_Severity == "Fatal Injury (Killed)":
        return "High Severity"
    else:
        return None  # Handle other categories if needed

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

    #discover unknowns dynamically and set them to pandas NaN
    unknown_values = discover_unknown_placeholders(df, UNKNOWN_STRINGS)
    df = df.replace(list(unknown_values), np.nan)

    #FIX make this severity column dynamic based on reading values in columns
    severity_col = "Crash Severity"

    #apply severity grouping
    df['grouped_severity'] = df['Crash Severity'].apply(group_severity)


    #define our target using our severity mapping
    y = df[severity_col].map(severity_mapping)

    #keep rows where it mapped properly
    mask = y.notna()
    X = df.loc[mask].drop(columns=[severity_col], errors="ignore")
    y = y.loc[mask].astype(int)

    print("y distribution:", y.value_counts().sort_index().to_dict())
    print("X shape:", X.shape)

    #find numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    
    print(f"Numeric cols: {len(numeric_cols)}")
    print(f"Categorical cols: {len(categorical_cols)}")


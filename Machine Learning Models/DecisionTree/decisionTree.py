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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


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

"""
#previous group implementation
def group_severity(Crash_Severity):
    if Crash_Severity in ["No Apparent Injury", "Suspected Minor Injury"]:
        return "Low Severity"
    elif Crash_Severity in ["Possible Injury", "Suspected Serious Injury"]:
        return "Moderate Severity"
    elif Crash_Severity == "Fatal Injury (Killed)":
        return "High Severity"
    else:
        return None  # Handle other categories if needed
"""

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

    #discover unknowns dynamically and set them to pandas NaN
    unknown_values = discover_unknown_placeholders(df, UNKNOWN_STRINGS)
    df = df.replace(list(unknown_values), np.nan)

    #FIX make this severity column dynamic based on reading values in columns
    #also make the severity mapping dynamic somehow?
    severity_col = "Crash Severity"

    #apply severity grouping
    #df['grouped_severity'] = df['Crash Severity'].apply(group_severity)


    #define our target using our severity mapping
    y = df[severity_col].map(severity_mapping)

    #keep rows where it mapped properly
    mask = y.notna()
    X = df.loc[mask].drop(columns=[severity_col], errors="ignore")
    y = y.loc[mask].astype(int)
    """
    lines below up to X = X.drop are to remove data leakage columns
    Data leakage columns are columns that give away the answer
    to the model, such as number of fatalities or injuries
    """
    #FIX make leakage columns dynamic based on dataset
    leak_cols = {
    "Number of Fatalities",
    "Number of Serious Injuries",
    "Number of Minor Injuries",
    "Number of Serious Injuries with Fatalities",
    "Form Type",    #temp fix
    }
    X = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")

    print("y distribution:", y.value_counts().sort_index().to_dict())
    print("X shape:", X.shape)

    #find numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    #fixme blocklist
    blocklist = {"Intersecting Street", "Reporting Agency", "City"}
    categorical_cols = [c for c in categorical_cols if c not in blocklist]

    print(f"Numeric cols: {len(numeric_cols)}")
    print(f"Categorical cols: {len(categorical_cols)}")


    #data preprocessing including the following
    #replace numeric empty values with median value
    #replace categorical empty values with most common value
    #one hot encode categorical columns
    #min frequency 0.01 means any rare category is grouped into an infrequent encoding
    #good practice since extremely rare categories often aren't beneficial to the model

    preprocess = ColumnTransformer(
        transformers=[
            #numeric transformer
            ("num", SimpleImputer(strategy="median"), numeric_cols),

            #categorical columns
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("to_str", FunctionTransformer(lambda X: X.astype(str))), #ensure all categorical data is string type
                ("ohe", OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.01,
                    dtype=np.float32,
                    sparse_output=False
                    )),
            ]), categorical_cols),
        ],
        remainder="drop", #drop any column not in our data (backup)
    )

    
    # Check if stratification is safe (each class needs at least 2 samples)
    vc = y.value_counts()
    can_stratify = (y.nunique() >= 2) and (vc.min() >= 2)
    if y.nunique() < 2:
        raise SystemExit("Target has fewer than 2 classes after mapping; cannot train a classifier.")

    """
    Stratification ensures that the ratio of 0/1/2 in severity
    stays the same across the train and test cases, otherwise you
    could end up with a massive skew in what classifications of
    data are in your train and test, leading to less than ideal outcomes.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None
    )

    #model will pay more attention to serious crashes (reduce false negatives)
    class_weights = "balanced"    #temp fix

    #create another sklearn pipeline to preprocess and create decision tree
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", DecisionTreeClassifier(
            random_state=42,
            criterion="entropy",    #test others
            max_depth=10,           #test others
            min_samples_split=2,    #test others
            min_samples_leaf=1,     #test others
            class_weight=class_weights
        ))
    ])
    #hyperparameters might be able to be dynamic like their lines 119-135
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print()
    print("Test Metrics (Decision Tree)")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-macro:  {f1:.4f}")
    print()
    print('\nConfusion Matrix:\n', conf_matrix)
    print('\nClassification Report:\n', class_report)
    print()
    #extract feature importances
    #get feature names after preprocessing
    ct = model.named_steps["preprocess"]
    #numeric feature names remain the same
    num_cols_used = ct.transformers_[0][2]
    num_feature_names=list(num_cols_used)

    #categorical feature names come from one hot encoder
    ohe = ct.named_transformers_["cat"].named_steps["ohe"]
    cat_cols_used = ct.transformers_[1][2]
    cat_feature_names = ohe.get_feature_names_out(cat_cols_used)

    #put feature names together
    feat_names = num_feature_names + list(cat_feature_names)


    tree = model.named_steps["model"]
    importances = tree.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
       .sort_values("Importance", ascending=False)
    )

    print("\nTop 15 features by importance:")
    print(importance_df.head(15).to_string(index=False))
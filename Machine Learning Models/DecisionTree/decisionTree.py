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


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#Put root directory into sys so we can get DataCleaning scripts
#and severity mapping utils

from DataCleaning.unknown_discovery import discover_unknown_placeholders
from DataCleaning.config import UNKNOWN_STRINGS

from severity_mapping_utils import find_severity_mapping

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
def find_high_cardinality(df, categorical_cols, threshold_ratio=0.1, absolute_cap=100):
    #find high cardinality categorical columns
    n_rows = len(df)
    hcc = []
    for c in categorical_cols:
        k = df[c].nunique(dropna=True)
        ratio = k / n_rows
        if k > absolute_cap or ratio > threshold_ratio:
            hcc.append(c)
    return hcc

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

    #discover unknowns dynamically and set them to pandas NaN
    unknown_values = discover_unknown_placeholders(df, UNKNOWN_STRINGS)
    df = df.replace(list(unknown_values), np.nan)

    #find severity column by prompting user 
    severity_col_found = False
    for col in column_names:
        if "severity" in col.lower() or "severe" in col.lower():
            severity_col = col
            print(f"\nIs'{severity_col}' the severity column? (y/n): ", end="")
            user_input = input().strip().lower()
            if user_input == 'y':
                print(f"Using '{severity_col}' as severity column.")
                severity_col_found = True
                break
            else:
                continue
    
    if not severity_col_found:
        print("\nCould not automatically determine severity column, please enter the column name: ")
    while not severity_col_found:
        severity_col = input().strip()
        if severity_col not in column_names:
            print(f"Error: '{severity_col}' is not a valid column name. Please try again.")
        else:
            print(f"Using '{severity_col}' as severity column.")
            severity_col_found = True

    #dynamically find severity mapping\
    severity_mapping = find_severity_mapping(df, severity_col)

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
    "Reporting Agency"  #temp fix
    }
    X = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")

    print("y distribution:", y.value_counts().sort_index().to_dict())
    print("X shape:", X.shape)

    #find numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    high_cardinality_cols = find_high_cardinality(X, categorical_cols)
    print(f"High cardinality categorical columns (to be removed): {high_cardinality_cols}")
    categorical_cols = [c for c in categorical_cols if c not in high_cardinality_cols]

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
    class_weights = "balanced"

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
    #dynamic hyperparameter tuning can be toggled
    #dynamic hyperparameters increases runtime significantly due to multiple fits
    #but can also massively increase model performance
    use_hyperparam_search = True

    if use_hyperparam_search:
        param_dist = {
            "model__max_depth": [5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__criterion": ["gini", "entropy", "log_loss"]
        }

        #randomly searches using hyperparameters and cross-validation through f1 score
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=40,             # number of random combinations to try
            scoring="f1_macro",    # scoring
            cv=3,                  # 3-fold cross-validation on the training set
            n_jobs=-1,             # use all cores if available
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        print("\nBest hyperparameters found:")
        print(search.best_params_)
        model = search.best_estimator_

    else:
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
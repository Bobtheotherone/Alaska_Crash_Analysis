import pandas as pd

def find_leakage_columns(X,y, use_near_perfect_check=True,
                         min_accuracy=0.9, max_unique=50):
    """
    Function which controls finding / selecting data leakage columns
    1) Name-Based Suggestions based on rudimentary keyword scan
    2) Near-perfect predictor check using (X, y)
    3) Prompt user for more columns to drop

    Returns a set of column names for data leakage
    """

    # 1) Name based scan
    name_suggestions = suggest_by_name(X.columns)

    # 2) Near-perfect predictors
    near_perfect_suggestions = []
    if use_near_perfect_check:
        near_perfect_suggestions = find_near_perfect_predictors(
            X,y,
            min_accuracy=min_accuracy,
            max_unique=max_unique,
        ) 

    # 3) User selections
    leak_cols = interactive_select_leakage(
        X,
        name_suggestions=name_suggestions,
        near_perfect_suggestions=near_perfect_suggestions)
    
    return leak_cols


def suggest_by_name(columns):
    """
    Finds suggested leakage columns via keywords
    """
    keywords = [
        "fatal", "fatalities", "death", "dead", "killed",
        "injury", "injuries", "injured",
        "severity", "severe", "serious",
        "k_count", "killed_cnt", "inj_cnt",
    ]

    suggestions = set()
    lower_map = {c: c.lower() for c in columns}

    for col, lower_name in lower_map.items():
        if any(kw in lower_name for kw in keywords):
            suggestions.add(col)

    return suggestions

def find_near_perfect_predictors(X, y, min_accuracy=0.98, max_unique=50):
    """
    Get columns in X
    check they have <= max_unique unique values
    compute how well that column predicts y via a majority mapping
    returns list of col, accuracy where accuracy >= min_accuracy
    """
    suspicious = []

    for col in X.columns:
        series = X[col]

        #skip high cardinality columns
        if series.nunique(dropna=True) > max_unique:
            continue
        
        df_col = pd.DataFrame({"feature": series, "target": y})
        df_col = df_col.dropna(subset=["feature", "target"])
        if df_col.empty:
            continue

        #map features to the majority target class
        mapping = (
            df_col.groupby("feature")["target"]
            .agg(lambda s: s.value_counts().idxmax())
        )

        y_hat = df_col["feature"].map(mapping)
        acc = (y_hat == df_col["target"]).mean()
        if acc >= min_accuracy:
            suspicious.append((col,acc))

    return suspicious

def interactive_select_leakage(
        X,
        name_suggestions,
        near_perfect_suggestions,
):
    """
    Combine our suggestions made and prompt user for which columns
    to actually mark as leakage columns
    """
    all_cols = list(X.columns)
    leak_cols = set()

    print("\n|--- Data Leakage Detection ---|")
    print(
        "Data leakage columns are features that reveal the outcome directly\n"
        "(e.g., number of fatalities, number of injuries, another severity code).\n"
        "These are usually not known at prediction time and will artificially\n"
        "inflate your model performance if left in.\n"
        "This can lead to innacurate prediction metrics, so we ask that\n"
        "you enter potential leakage columns to improve your results."
        "We have done some scanning and have some recommendations, but these\n"
        "recommendations are not comprehensive."
    )
    # 1) Show all columns
    print("\nAll feature columns:")
    for c in all_cols:
        print(f"  - {c}")

    # 2) Name-based suggestions
    if name_suggestions:
        print("\nColumns suspected by NAME (may indicate leakage):")
        for c in sorted(name_suggestions):
            print(f"  * {c}")
    else:
        print("\nNo columns suspected by name keywords.")

    # 3) Near-perfect predictors
    if near_perfect_suggestions:
        print("\nColumns that almost perfectly predict severity by themselves:")
        for col, acc in near_perfect_suggestions:
            print(f"  * {col} (single-column accuracy ≈ {acc:.3f})")
        print("These are VERY likely to be leakage.")
    else:
        print("\nNo near-perfect single-column predictors found.")

    # 4) Ask user to pick leakage columns
    print("\nEnter a comma-separated list of columns to DROP as leakage.")
    print("Example: Number of Injuries, Number of Fatalities, Form Type")
    print("You can include any columns from above suggestions or others.")
    print("Press Enter to skip if you don't want to drop anything.")
    user_input = input("Leakage columns to drop: ").strip()

    if user_input:
        parts = [p.strip() for p in user_input.split(",")]
        for p in parts:
            if p in X.columns:
                leak_cols.add(p)
            else:
                print(f"  (Warning: '{p}' is not a valid column name, ignoring.)")

    return leak_cols
    

def warn_suspicious_importances(feature_names, importances,
                                importance_threshold=0.2,
                                dominance_ratio=2.0,
                                top_n=10):
    """
    Post-training helper: look at feature importances and warn
    about any feature that is either:
      - Above importance_threshold, or
      - More than dominance_ratio times the second-highest importance.

    Args:
        feature_names: list of feature names (same order as importances)
        importances: 1D array-like of importances
        importance_threshold: minimum importance to flag
        dominance_ratio: how many times larger than second-max to be 'dominant'
        top_n: how many top features to print for context

    Returns:
        List of suspicious feature names.
    """
    if not feature_names or len(feature_names) != len(importances):
        print("warn_suspicious_importances: feature_names and importances size mismatch.")
        return []

    # Pair up and sort
    pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    if len(pairs) == 0:
        return []

    max_name, max_imp = pairs[0]
    second_imp = pairs[1][1] if len(pairs) > 1 else 0.0

    suspicious = []

    if second_imp == 0:
        dominance = float("inf") if max_imp > 0 else 0.0
    else:
        dominance = max_imp / second_imp

    if (max_imp >= importance_threshold) or (dominance >= dominance_ratio):
        suspicious.append(max_name)

    if suspicious:
        print("\nWARNING: The following feature(s) have unusually high importance and may indicate leakage:")
        for name in suspicious:
            print(f"  - {name}")
        print(
            "Inspect whether these columns encode information that would not be\n"
            "available at prediction time (e.g., injury counts, final severity codes)."
        )
    else:
        print("\nNo obviously dominant features found based on the current thresholds.")

    return suspicious
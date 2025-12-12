import pandas as pd
def find_severity_mapping(df, severity_col):
    """
    Will dynamically find the severity mapping by first scanning
    for keywords, then if issues are found, prompt the user to
    manually map the severity levels.
    0 = low severity
    1 = moderate severity
    2 = high severity
    Returns a dictionary mapping original severity values to 0, 1, 2
    """
    sev_vals = df[severity_col].dropna()
    unique_sev_vals= sev_vals.unique().tolist()
    severity_mapping = {}

    numeric_sev = pd.to_numeric(sev_vals, errors="coerce") #check if it converts to numeric
    num_valid = numeric_sev.notna().sum()
    frac_valid = num_valid / len(sev_vals) if len(sev_vals) > 0 else 0.0

    #check for numeric severity first
    if frac_valid >= 0.9: #if 90% of our values can be mapped to numeric values
        mapping = map_numeric_severity(unique_sev_vals)
        if mapping:

            return confirm_or_edit_mapping(mapping, sev_vals)
    
    #non numeric severity mapping
    text_mapping = map_text_severity(unique_sev_vals)
    if text_mapping:

        return confirm_or_edit_mapping(text_mapping, sev_vals)
    
    #if we reach here, we couldn't map automatically
    print("\nCould not automatically determine severity mapping.")
    print("Please enter the severity mapping manually.")
    return manual_severity_mapping(unique_sev_vals)


def map_numeric_severity(unique_sev_values):
    """
    Maps numeric severity values to 0, 1, 2 based on thresholds
    Returns a dictionary mapping original severity values to 0, 1, 2
    """
    value_to_num = {}
    numeric_codes = []

    for val in unique_sev_values:
        num = pd.to_numeric([val], errors="coerce")[0]
        if not pd.isna(num):
            value_to_num[val] = num
            numeric_codes.append(num)
    
    if not numeric_codes:
        return {}
    
    unique_codes = sorted(set(numeric_codes))
    n_codes = len(unique_codes)
    severity_mapping = {}

    if n_codes == 1:
        #only 1 severity level, just map to medium
        only_code = unique_codes[0]
        for raw_val, num in value_to_num.items():
            if num == only_code:
                severity_mapping[raw_val] = 1
    elif n_codes == 2:
        #two levels, map lower to 0, higher to 2
        low_code = unique_codes[0]
        high_code = unique_codes[1]
        for raw_val, num in value_to_num.items():
            if num == low_code:
                severity_mapping[raw_val] = 0
            elif num == high_code:
                severity_mapping[raw_val] = 2
    else:
        #three or more levels, map lowest to 0, highest to 2, rest to middle
        low_code = unique_codes[0]
        high_code = unique_codes[-1]
        middle_codes = set(unique_codes[1:-1])

        for raw_val, num in value_to_num.items():
            if num == low_code:
                severity_mapping[raw_val] = 0
            elif num == high_code:
                severity_mapping[raw_val] = 2
            elif num in middle_codes:
                severity_mapping[raw_val] = 1

    return severity_mapping

def map_text_severity(unique_sev_values):
    """
    Maps text severity values to 0, 1, 2 based on strings
    Not guaranteed to work on all datasets
    Returns a dictionary mapping original severity values to 0, 1, 2
    """
    low_keywords = [
        "no injury", "no apparent", "no evident", "no visible",
        "non-injury", "property damage only", "pdo", "noninjury"
    ]
    mid_keywords = [
        "possible", "minor", "non-incapacitating",
        "complaint of pain", "c injury", "b injury"
    ]
    high_keywords = [
        "serious", "severe", "major", "incapacitating",
        "hospitalized", "critical", "life-threatening", "a injury"
    ]
    fatal_keywords = [
        "fatal", "killed", "death", "dead", "k injury"
    ]
    severity_mapping = {}
    for raw_val in unique_sev_values:
        val_lower = str(raw_val).lower()
        sev = None
        if any(kw in val_lower for kw in fatal_keywords):
            sev = 2
        elif any(kw in val_lower for kw in high_keywords):
            sev = 2
        elif any(kw in val_lower for kw in mid_keywords):
            sev = 1
        elif any(kw in val_lower for kw in low_keywords):
            sev = 0

        if sev is not None:
            severity_mapping[raw_val] = sev

    return severity_mapping

def confirm_or_edit_mapping(proposed_mapping, sev_vals):
    """
    Show the user the proposed severity mapping
    and allow them to confirm or edit it.
    """
    print("\nProposed severity mapping:")
    value_counts = sev_vals.value_counts(dropna=False)
    label_names = {0: "Low", 1: "Moderate", 2: "High"}

    for raw_val, sev in proposed_mapping.items():
        count = value_counts.get(raw_val, 0)
        label = label_names.get(sev, "Unknown")
        print(f"  {repr(raw_val)}  ->  {sev} ({label})   [Count: {count}]")
    response = input("\nUse this mapping? (y/n): ").strip().lower()
    if response == 'y':
        return proposed_mapping
    
    print("\nResorting to manual mapping.")
    unique_vals = value_counts.index.tolist()

    return manual_severity_mapping(unique_vals)


def manual_severity_mapping(unique_sev_vals):
    """
    Have the user manually input the severity mapping
    for each unique severity value.
    """
    print("\nManual severity mapping.")
    print("For each value, enter:")
    print("  0 = low severity")
    print("  1 = moderate severity")
    print("  2 = high severity")
    print("  'skip' (or just Enter) to leave unmapped (these rows will be dropped).")
    severity_mapping = {}

    for raw_val in unique_sev_vals:
        while True:
            user_input = input(f"Mapping for {repr(raw_val)}: ").strip().lower()
            if user_input == 'skip' or user_input == '':
                break
            elif user_input in {'0', '1', '2'}:
                severity_mapping[raw_val] = int(user_input)
                break
            else:
                print("Invalid input. Please enter 0, 1, 2, or 'skip'.")
    
    return severity_mapping
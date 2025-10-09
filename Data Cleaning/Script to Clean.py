#make a script to clean the dataset

import pandas as pd

file_path1='Crash Level 13-17 (1).xlsx'
#file_path2='Crash Level 09-12 (1).xlsx'        #dataset 2
df1 = pd.read_excel(file_path1)
#df2 = pd.read_excel(file_path2)                #dataset 2

UNKNOWN_STRINGS = {
    "no data",
    "missing value",
    "null value",
    "missing",
    "na", "n/a", "n.a.",
    "none",
    "null",
    "nan",
    "unknown",
    "unspecified",
    "not specified",
    "not applicable",
    "tbd", "tba", "to be determined",
    "-", "--",
    "(blank)", "blank",
    "(null)",
    "?", 
    "prefer not to say",
    "refused",
    # adding some specific to dataset 1
    "unknown census area",  #census area
    "unknown house district",       #house/election district
    "unknown election district",    #house/election district
    "unknown station",                #maintenance station
    "unknown category",           #maintenance category
    "unknown maintenance responsibility",       #maintenance responsibility
    "missing functional class",               #functional class
    "unspecified area",                   # urban/rural
    #end of dataset 1 specific additions
}

def percent_unknowns_per_column(df, unknown_strings):
    tokens = {s.strip().lower() for s in unknown_strings} #tokenize and lowercase column inputs
    results = []
    for col in df.columns:
        series = df[col]            #each row in the column
        total_count = len(series)   #total rows in the column
        if total_count == 0:
            percent_unknown = 0.0
        else:
            mask_null = series.isna() #to avoid pandas 'nan' string issue
            series_norm = series[~mask_null].astype(str).str.strip().str.lower() #while not pandas null, convert to string, strip string, lowercase string
            mask_token = series_norm.isin(tokens)   #check if in our unknown tokens
            unknown_count = int(mask_null.sum()) + int(mask_token.sum()) #sum pandas nans and our known unknowns
            percent_unknown = (unknown_count / total_count) * 100
        results.append((col, percent_unknown))

    return results

def yes_no(df, unknown_strings):
    #scans each column for percentage of yes/no values
    tokens = {s.strip().lower() for s in unknown_strings} #tokenize and lowercase column inputs
    YES = {"yes", "y", "true", "t"}
    NO = {"no", "n", "false", "f"}

    out = {}
    for col in df.columns:
        series = df[col]            #each row in the column

        mask_not_null = (~series.isna()) #filter out pandas nulls
        if not mask_not_null.any():
            continue

        series_norm = series[mask_not_null].astype(str).str.strip().str.lower() #while not pandas null, convert to string, strip string, lowercase string
        series_norm = series_norm[~series_norm.isin(tokens)] #filter out known unknowns

        is_yes = series_norm.isin(YES)
        is_no = series_norm.isin(NO)

        yes_count = int(is_yes.sum())
        no_count = int(is_no.sum())
        total_count = yes_count + no_count      #total count of yes/no for percentage calc

        if total_count > 0:
            yes_pct = round(100.0 * yes_count / total_count, 2)
            no_pct  = round(100.0 * no_count  / total_count, 2)
            out[col] = (yes_pct, no_pct, yes_count, no_count, total_count)

    return out

res = percent_unknowns_per_column(df1, UNKNOWN_STRINGS)
yn = yes_no(df1, UNKNOWN_STRINGS)

#just print for now until we decide functionality
res.sort(key=lambda x: x[1], reverse=True)
print("\n% Unknown by column (NaN + known placeholders):")
for col, pct in res:
    extra = ""
    if col in yn:
        y_pct, n_pct, y_cnt, n_cnt, total = yn[col]
        extra = f"   (Yes/No: {y_pct:.2f}%/{n_pct:.2f}% of {total} known)"
    print(f"{pct:6.2f}%  -  {col}{extra}")
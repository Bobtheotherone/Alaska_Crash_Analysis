from typing import Set #sets are faster lookup than python lists
import pandas as pd
from pathlib import Path
import re

GENERIC_UNKNOWN_SUBSTRINGS: Set[str] = {
    "unknown",
    "missing",
    "unspecified",
    "not specified",
    "not applicable",
    "n/a",
    "na",
    "null",
    "blank",
    "tbd",
    "tba",
    "to be determined",
    "refused",
    "prefer not to say",
    "no data",
    "no value",
}

#scans the dataset for new unknown placeholder values
def discover_unknown_placeholders(
        df,
        base_unknowns,
        substrings=GENERIC_UNKNOWN_SUBSTRINGS,
        min_freq=2,
        max_token_len=80
    ):

    base_set = {s.strip().lower() for s in base_unknowns}
    generic_parts = [p.lower() for p in substrings if p] #generic unknown words
    discovered = {} #newly discovered unknowns are added here

    #avoids detecting na as a substring rather than a seperate word eg Tanana would detect it and flag it as unknown
    patterns = [re.compile(rf"\b{re.escape(part)}\b") for part in generic_parts]

    for col in df.columns:
        #normalizes non-null values
        s = df[col].dropna().astype(str).str.strip().str.lower()
        for v in s: #for each cell value (row x column)
            if not v or v in base_set or len(v) > max_token_len: #cap token length for processing concerns
                continue
            if any(p.search(v) for p in patterns): #if we find a substring of generic unknown
                discovered[v] = discovered.get(v,0) + 1 #add string to unknowns and count it

    freq = {tok for tok, count in discovered.items() if count >= min_freq} #ensure at least 2 occurrences of this value
    return base_set | freq #combine known unknowns and new unknowns



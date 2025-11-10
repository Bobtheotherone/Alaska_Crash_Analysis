import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 100_000

# ---- Severity (as STRINGS matching your severity_mapping) ----

# First choose numeric 0/1/2 to control proportions
severity_probs = [0.65, 0.28, 0.07]  # ~ real distribution
severity_numeric = np.random.choice([0, 1, 2], size=n_rows, p=severity_probs)

# Map 0/1/2 to your string labels
severity_strings = []
for s in severity_numeric:
    if s == 0:
        severity_strings.append("No Apparent Injury")
    elif s == 1:
        severity_strings.append(
            np.random.choice(["Possible Injury", "Suspected Minor Injury"])
        )
    else:  # 2
        severity_strings.append(
            np.random.choice(["Suspected Serious Injury", "Fatal Injury (Killed)"])
        )

# ---- Core numeric columns ----
miles = np.random.uniform(0, 400, n_rows)              # Milepoint
aadt = np.random.randint(100, 40000, n_rows)           # AADT
latitudes = np.random.uniform(58.0, 71.0, n_rows)
longitudes = np.random.uniform(-170.0, -130.0, n_rows)
num_units = np.random.randint(1, 5, n_rows)            # Number of Motorized Units
year = np.random.choice(range(2015, 2025), size=n_rows)
month = np.random.randint(1, 13, n_rows)
day = np.random.randint(1, 29, n_rows)
hour = np.random.randint(0, 24, n_rows)

# ---- Categorical features (realistic-ish domains) ----
weather = np.random.choice(
    ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Wind", "Other"],
    n_rows,
    p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02],
)

road_surface = np.random.choice(
    ["Dry", "Wet", "Ice/Frost", "Snow", "Other"],
    n_rows,
    p=[0.55, 0.25, 0.1, 0.08, 0.02],
)

lighting = np.random.choice(
    ["Daylight", "Dark - Lighted", "Dark - Unlighted", "Dawn", "Dusk"],
    n_rows,
    p=[0.6, 0.15, 0.15, 0.05, 0.05],
)

alcohol = np.random.choice(["Yes", "No"], n_rows, p=[0.1, 0.9])
drugs = np.random.choice(["Yes", "No"], n_rows, p=[0.05, 0.95])

region = np.random.choice(
    ["Northern", "Interior", "Southwest", "Anchorage", "Southeast"], n_rows
)
junction = np.random.choice(
    ["None", "T-Intersection", "Crossroad", "Ramp", "Driveway"], n_rows
)
crash_type = np.random.choice(
    ["Rear End", "Angle", "Head-On", "Sideswipe", "Single Vehicle", "Motorcycle"],
    n_rows,
)
manner = np.random.choice(
    ["Front-To-Rear", "Front-To-Front", "Side", "Other"], n_rows
)
relation = np.random.choice(
    ["On Roadway", "Off Roadway", "Shoulder", "Median"], n_rows
)

county = np.random.choice(
    ["Anchorage", "Fairbanks", "Mat-Su", "Kenai", "Juneau"], n_rows
)
city = np.random.choice(
    ["Anchorage", "Fairbanks", "Wasilla", "Kenai", "Juneau", "Nome"], n_rows
)
ahs_system = np.random.choice(["Yes", "No"], n_rows)
nhs_system = np.random.choice(["Yes", "No"], n_rows)

# ---- Columns intended to be DROPPED by your cleaner ----

# 1) High-cardinality ID-like column (nunique ~ n_rows)
row_id = np.array([f"ID_{i:06d}" for i in range(n_rows)])

# 2) All-same column (nunique == 1)
all_same_col = np.array(["SAME_VALUE"] * n_rows)

# 3) Almost-constant column: ~99.7% "A", small minority "B"
almost_constant = np.where(
    np.random.rand(n_rows) < 0.997, "A", "B"
)

# 4) Mostly-unknown column with common placeholder tokens
unknown_tokens = ["UNKNOWN", "Unknown", "Not Stated", "N/A", " "]
high_unknown_col = np.where(
    np.random.rand(n_rows) < 0.9,
    np.random.choice(unknown_tokens, n_rows),
    np.random.choice(["foo", "bar", "baz"], n_rows),
)

# 5) Random junk text with lots of unique-ish values
random_notes = np.array([
    f"note_{np.random.randint(0, 10_000)}"
    for _ in range(n_rows)
])

# ---- Build DataFrame ----
df = pd.DataFrame({
    # "Real" columns similar to your cleaned file
    "Form Type": np.random.choice(["Long", "Short"], n_rows),
    "Reporting Agency": np.random.choice(["State Troopers", "Local PD", "Other"], n_rows),
    "Milepoint": miles,
    "Latitude": latitudes,
    "Longitude": longitudes,
    "AADT": aadt,
    "Year": year,
    "Month": month,
    "Day of Month": day,
    "Day of the Week": np.random.choice(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n_rows
    ),
    "Time of Day": hour,
    "At Intersection": np.random.choice(["Yes", "No"], n_rows),
    "Junction": junction,
    "Number of Motorized Units": num_units,
    "Crash Severity": severity_strings,  # <- uses your mapping
    "Number of Serious Injuries": np.random.randint(0, 3, n_rows),
    "Number of Minor Injuries": np.random.randint(0, 5, n_rows),
    "Number of Fatalities": np.random.randint(0, 2, n_rows),
    "First Harmful Event": np.random.choice(
        ["Collision", "Overturn", "Pedestrian", "Animal"], n_rows
    ),
    "Manner of Collision": manner,
    "Crash Type": crash_type,
    "Relation to Trafficway": relation,
    "Weather": weather,
    "Road Surface": road_surface,
    "Lighting": lighting,
    "Alcohol Suspected": alcohol,
    "Drugs Suspected": drugs,
    "Region": region,
    "County-Borough": county,
    "City": city,
    "NHS System": nhs_system,
    "AHS System": ahs_system,
    "Pavement": np.random.choice(
        ["Asphalt", "Concrete", "Gravel", "Other"], n_rows
    ),
    "Within Intersection Named Zone": np.random.choice(["Yes", "No"], n_rows),
    "Within Safety Corridor Named Zone": np.random.choice(["Yes", "No"], n_rows),
    "Within School Zone Named Zone": np.random.choice(["Yes", "No"], n_rows),

    # Dirty / likely-to-be-dropped columns
    "Row ID": row_id,                       # high cardinality
    "All Same Column": all_same_col,        # nunique == 1
    "Almost Constant Column": almost_constant,  # near-constant
    "High Unknown Column": high_unknown_col,    # unknown placeholders
    "Random Notes": random_notes,           # lots of uniques
})

# Optionally inject some unknown-like tokens into a real column too:
mask_unknown_weather = np.random.rand(n_rows) < 0.05
df.loc[mask_unknown_weather, "Weather"] = np.random.choice(unknown_tokens, mask_unknown_weather.sum())

df.to_csv("synthetic_crashes.csv", index=False)
print("✅ synthetic_crashes.csv created with shape:", df.shape)

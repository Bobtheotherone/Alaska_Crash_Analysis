import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 100_000

# ----------------------------------
# 1. Base feature generation
# ----------------------------------

# Core numeric columns
milepoint = np.random.uniform(0, 400, n_rows)
aadt = np.random.randint(100, 40_000, n_rows)
latitudes = np.random.uniform(58.0, 71.0, n_rows)
longitudes = np.random.uniform(-170.0, -130.0, n_rows)
num_units = np.random.randint(1, 5, n_rows)            # 1–4 vehicles
year = np.random.choice(range(2015, 2025), size=n_rows)
month = np.random.randint(1, 13, n_rows)
day = np.random.randint(1, 29, n_rows)
hour = np.random.randint(0, 24, n_rows)

weather_vals = np.random.choice(
    ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Wind", "Other"],
    n_rows,
    p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02],
)

road_surface_vals = np.random.choice(
    ["Dry", "Wet", "Ice/Frost", "Snow", "Other"],
    n_rows,
    p=[0.55, 0.25, 0.1, 0.08, 0.02],
)

lighting_vals = np.random.choice(
    ["Daylight", "Dark - Lighted", "Dark - Unlighted", "Dawn", "Dusk"],
    n_rows,
    p=[0.6, 0.15, 0.15, 0.05, 0.05],
)

alcohol_vals = np.random.choice(["Yes", "No"], n_rows, p=[0.1, 0.9])
drugs_vals = np.random.choice(["Yes", "No"], n_rows, p=[0.05, 0.95])

region_vals = np.random.choice(
    ["Northern", "Interior", "Southwest", "Anchorage", "Southeast"], n_rows
)
junction_vals = np.random.choice(
    ["None", "T-Intersection", "Crossroad", "Ramp", "Driveway"], n_rows
)
crash_type_vals = np.random.choice(
    ["Rear End", "Angle", "Head-On", "Sideswipe", "Single Vehicle", "Motorcycle"],
    n_rows,
)
manner_vals = np.random.choice(
    ["Front-To-Rear", "Front-To-Front", "Side", "Other"], n_rows
)
relation_vals = np.random.choice(
    ["On Roadway", "Off Roadway", "Shoulder", "Median"], n_rows
)

county_vals = np.random.choice(
    ["Anchorage", "Fairbanks", "Mat-Su", "Kenai", "Juneau"], n_rows
)
city_vals = np.random.choice(
    ["Anchorage", "Fairbanks", "Wasilla", "Kenai", "Juneau", "Nome"], n_rows
)
ahs_system_vals = np.random.choice(["Yes", "No"], n_rows)
nhs_system_vals = np.random.choice(["Yes", "No"], n_rows)

# ----------------------------------
# 2. Risk score from features
#    (positive/negative contributions)
# ----------------------------------

risk = np.zeros(n_rows, dtype=float)

# Weather effects
weather_effect = {
    "Clear": -0.6,
    "Cloudy": -0.2,
    "Rain": 0.2,
    "Snow": 0.6,
    "Fog": 0.8,
    "Wind": 0.1,
    "Other": 0.0,
}
risk += np.vectorize(weather_effect.get)(weather_vals)

# Road surface effects
road_effect = {
    "Dry": -0.4,
    "Wet": 0.2,
    "Ice/Frost": 1.0,
    "Snow": 0.7,
    "Other": 0.3,
}
risk += np.vectorize(road_effect.get)(road_surface_vals)

# Lighting effects
lighting_effect = {
    "Daylight": -0.4,
    "Dark - Lighted": 0.3,
    "Dark - Unlighted": 0.8,
    "Dawn": 0.1,
    "Dusk": 0.2,
}
risk += np.vectorize(lighting_effect.get)(lighting_vals)

# Crash type effects
crash_effect = {
    "Rear End": 0.2,
    "Angle": 0.5,
    "Head-On": 1.0,        # strong positive severity contribution
    "Sideswipe": 0.3,
    "Single Vehicle": 0.1,
    "Motorcycle": 0.9,     # strong positive
}
risk += np.vectorize(crash_effect.get)(crash_type_vals)

# Manner of collision
manner_effect = {
    "Front-To-Rear": 0.3,
    "Front-To-Front": 0.9,  # bad
    "Side": 0.4,
    "Other": 0.1,
}
risk += np.vectorize(manner_effect.get)(manner_vals)

# Relation to trafficway
relation_effect = {
    "On Roadway": 0.3,
    "Off Roadway": 0.5,   # into ditch/trees etc.
    "Shoulder": 0.1,
    "Median": 0.2,
}
risk += np.vectorize(relation_effect.get)(relation_vals)

# Alcohol / drugs
risk += np.where(alcohol_vals == "Yes", 1.0, 0.0)
risk += np.where(drugs_vals == "Yes", 0.7, 0.0)

# Number of vehicles (more vehicles -> more chance for bad)
risk += (num_units - 1) * 0.15

# AADT: heavier traffic slightly increases risk
risk += (aadt / 40_000.0) * 0.3  # scaled into ~[0, 0.3]

# Small noise term
risk += np.random.normal(loc=0.0, scale=0.4, size=n_rows)

# Optional: clamp extremely low/high values to a reasonable range
risk = np.clip(risk, -2.0, 5.0)

# ----------------------------------
# 3. Convert risk to severity 0/1/2
#    (thresholds you can tweak)
# ----------------------------------

# thresholding: low risk -> 0, mid -> 1, high -> 2
severity_numeric = np.empty(n_rows, dtype=int)
severity_numeric[risk < 0.5] = 0
severity_numeric[(risk >= 0.5) & (risk < 1.7)] = 1
severity_numeric[risk >= 1.7] = 2

# Map 0/1/2 to your original string labels
severity_strings = []
for s in severity_numeric:
    if s == 0:
        severity_strings.append("No Apparent Injury")
    elif s == 1:
        severity_strings.append(
            np.random.choice(["Possible Injury", "Suspected Minor Injury"])
        )
    else:
        severity_strings.append(
            np.random.choice(["Suspected Serious Injury", "Fatal Injury (Killed)"])
        )

# For fun check approximate distribution
unique, counts = np.unique(severity_numeric, return_counts=True)
print("Severity distribution (0/1/2):", dict(zip(unique, counts)))

# ----------------------------------
# 4. "Dirty" columns for cleaning tests
# ----------------------------------

# High-cardinality ID-like column
row_id = np.array([f"ID_{i:06d}" for i in range(n_rows)])

# All-same column
all_same_col = np.array(["SAME_VALUE"] * n_rows)

# Almost-constant column
almost_constant = np.where(
    np.random.rand(n_rows) < 0.997, "A", "B"
)

# Mostly-unknown column
unknown_tokens = ["UNKNOWN", "Unknown", "Not Stated", "N/A", " "]
high_unknown_col = np.where(
    np.random.rand(n_rows) < 0.9,
    np.random.choice(unknown_tokens, n_rows),
    np.random.choice(["foo", "bar", "baz"], n_rows),
)

# Random junk notes (high cardinality)
random_notes = np.array([
    f"note_{np.random.randint(0, 10_000)}" for _ in range(n_rows)
])

# ----------------------------------
# 5. Build DataFrame
# ----------------------------------

df = pd.DataFrame({
    "Form Type": np.random.choice(["Long", "Short"], n_rows),
    "Reporting Agency": np.random.choice(["State Troopers", "Local PD", "Other"], n_rows),
    "Milepoint": milepoint,
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
    "Junction": junction_vals,
    "Number of Motorized Units": num_units,
    "Crash Severity": severity_strings,  # <- your mapping-compatible labels
    "Number of Serious Injuries": np.random.randint(0, 3, n_rows),
    "Number of Minor Injuries": np.random.randint(0, 5, n_rows),
    "Number of Fatalities": np.random.randint(0, 2, n_rows),
    "First Harmful Event": np.random.choice(
        ["Collision", "Overturn", "Pedestrian", "Animal"], n_rows
    ),
    "Manner of Collision": manner_vals,
    "Crash Type": crash_type_vals,
    "Relation to Trafficway": relation_vals,
    "Weather": weather_vals,
    "Road Surface": road_surface_vals,
    "Lighting": lighting_vals,
    "Alcohol Suspected": alcohol_vals,
    "Drugs Suspected": drugs_vals,
    "Region": region_vals,
    "County-Borough": county_vals,
    "City": city_vals,
    "NHS System": nhs_system_vals,
    "AHS System": ahs_system_vals,
    "Pavement": np.random.choice(
        ["Asphalt", "Concrete", "Gravel", "Other"], n_rows
    ),
    "Within Intersection Named Zone": np.random.choice(["Yes", "No"], n_rows),
    "Within Safety Corridor Named Zone": np.random.choice(["Yes", "No"], n_rows),
    "Within School Zone Named Zone": np.random.choice(["Yes", "No"], n_rows),

    # dirty / likely-to-be-dropped
    "Row ID": row_id,
    "All Same Column": all_same_col,
    "Almost Constant Column": almost_constant,
    "High Unknown Column": high_unknown_col,
    "Random Notes": random_notes,
})

# Inject some unknown tokens into a real feature too
mask_unknown_weather = np.random.rand(n_rows) < 0.05
df.loc[mask_unknown_weather, "Weather"] = np.random.choice(
    unknown_tokens, mask_unknown_weather.sum()
)

# ----------------------------------
# 6. Save to CSV
# ----------------------------------

out_path = "synthetic_crashes.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved synthetic dataset to: {out_path}")
print("Shape:", df.shape)

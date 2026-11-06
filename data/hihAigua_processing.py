import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

array_path = "data/hihAigua/hihAigua_array.npy"
features_path = "data/hihAigua/hihAigua_features.txt"

output_path = "data/hihaigua_daily_seqlen206_v2.h5"

TRAIN_SIZE = 0.70
VAL_SIZE = 0.20
TEST_SIZE = 0.10

SEED = 42

# Empty -> keep all features
FEATURE_FILTER = [
    "hotWater_volume",
    "water_volume",
    "heatCoolingLoadMeter_returnTemperature",
    "heatCoolingLoadMeter_flowTemperature",
    "ambientSensor_temperature"
]

data = np.load(array_path)

with open(features_path) as f:
    feature_names = [line.strip() for line in f if line.strip()]

assert data.shape[2] == len(feature_names)

if len(FEATURE_FILTER) == 0:
    selected_features = feature_names
    selected_indices = np.arange(len(feature_names))
else:
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    selected_indices = [feature_to_idx[f] for f in FEATURE_FILTER]
    selected_features = FEATURE_FILTER

data = data[:, :, selected_indices]
for feature in ["hotWater_volume", "water_volume"]:
    if feature in selected_features:
        idx = selected_features.index(feature)
        data[:, :, idx] = np.maximum(data[:, :, idx], 0)

temp_features = [
    "heatCoolingLoadMeter_returnTemperature", 
    "heatCoolingLoadMeter_flowTemperature", 
    "ambientSensor_temperature"
]

MAX_CONSECUTIVE_INTERP_GAP = 10  # Max zero run length to linearly interpolate

for feature in temp_features:
    if feature in selected_features:
        idx = selected_features.index(feature)
        
        # Enforce non-negative constraint
        data[:, :, idx] = np.maximum(data[:, :, idx], 0)
        
        # Compute overall non-zero median feature value for complete outages
        non_zero_vals = data[:, :, idx][data[:, :, idx] > 0]
        global_median = np.median(non_zero_vals) if len(non_zero_vals) > 0 else 0.0

        # Process each sample sequence individually
        for i in range(data.shape[0]):
            seq = data[i, :, idx]
            
            # Skip sequences that are entirely non-zero
            if not np.any(seq == 0):
                continue

            # Convert sequence to Pandas Series
            s = pd.Series(seq)
            
            # Mark exact 0s as NaN
            s[s == 0] = np.nan
            
            # Linearly interpolate consecutive zeros up to MAX_CONSECUTIVE_INTERP_GAP
            s_interp = s.interpolate(
                method="linear", 
                limit=MAX_CONSECUTIVE_INTERP_GAP, 
                limit_direction="both"
            )
            
            # If entire sequence was 0 (or gaps exceed limit), fill remaining NaNs with global median
            s_interp = s_interp.fillna(global_median)
            
            # Write back to array
            data[i, :, idx] = s_interp.values

indices = np.arange(len(data))

train_idx, temp_idx = train_test_split(
    indices,
    train_size=TRAIN_SIZE,
    shuffle=True,
    random_state=SEED,
)

val_fraction = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

val_idx, test_idx = train_test_split(
    temp_idx,
    train_size=val_fraction,
    shuffle=True,
    random_state=SEED,
)

string_dt = h5py.string_dtype("utf-8")
print(data.shape)
with h5py.File(output_path, "w") as f:

    f.create_dataset(
        "features",
        data=np.asarray(selected_features, dtype=object),
        dtype=string_dt,
    )

    f.create_dataset("train", data=data[train_idx], compression="gzip")
    f.create_dataset("val", data=data[val_idx], compression="gzip")
    f.create_dataset("test", data=data[test_idx], compression="gzip")

print(f"Saved dataset to {output_path}")
print(f"Train: {len(train_idx)}")
print(f"Val:   {len(val_idx)}")
print(f"Test:  {len(test_idx)}")
import os
import numpy as np
import pandas as pd
from pathlib import Path


# set up
data_folder = Path(__file__).resolve().parent.parent / "data"
inp = data_folder / "restaurant_info.csv"

p_busy, p_long, p_must = 0.50, 0.55, 0.60
seed = 42

# reading CSV
df = pd.read_csv(inp)

# Def keys
candidate_keys = ["restaurant_id", "place_id", "id", "slug", "name"]
key = next((k for k in candidate_keys if k in df.columns), None)
if key is None:
    df = df.reset_index().rename(columns={"index": "row_id"})
    key = "row_id"

# Unique keys + random feature
keys_df = df[[key]].drop_duplicates().copy()
n = len(keys_df)
rng = np.random.default_rng(seed)
features = keys_df.assign(
    crowdedness=(rng.random(n) < p_busy).astype(int),      # 1=busy, 0=not
    length_of_stay=(rng.random(n) < p_long).astype(int),   # 1=long, 0=short
    food_quality=(rng.random(n) < p_must).astype(int),     # 1=must-visit, 0=average
)

# right join:
merged = pd.merge(features, df, on=key, how="right")

# adding new columns to the end
new_cols = ["crowdedness", "length_of_stay", "food_quality"]
for c in new_cols:
    merged[c] = merged[c].fillna(0).astype(int).clip(0, 1)
other_cols = [c for c in merged.columns if c not in new_cols]
merged = merged[other_cols + new_cols]

# data file
os.makedirs(os.path.dirname(inp), exist_ok=True)

# save
merged.to_csv(inp, index=False)

print(f"✅ Saved to: {os.path.abspath(inp)}")
print(f"Rows: {len(merged):,} | Key: {key} | "
      f"crowdedness=1: {merged['crowdedness'].mean():.2%}, "
      f"length_of_stay=1: {merged['length_of_stay'].mean():.2%}, "
      f"food_quality=1: {merged['food_quality'].mean():.2%}")

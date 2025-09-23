import numpy as np
import pandas as pd
from pathlib import Path

# set up
inp = Path.home() / "/Users/nickchuprinskiy/Desktop/MAR1-master-3/restaurant_info.csv"  # file path
out = Path("data/restaurant_info.csv")
p_busy, p_long, p_must = 0.50, 0.55, 0.60
seed = 42

# reading initial file
df = pd.read_csv(inp)

# detecting keys
candidate_keys = ["restaurant_id", "place_id", "id", "slug", "name"]
key = next((k for k in candidate_keys if k in df.columns), None)
if key is None:
    df = df.reset_index().rename(columns={"index": "row_id"})
    key = "row_id"

# unique keys
keys_df = df[[key]].drop_duplicates().copy()
n = len(keys_df)

# generating binary features
rng = np.random.default_rng(seed)
features = keys_df.assign(
    busy=(rng.random(n) < p_busy).astype(int),
    long_stay=(rng.random(n) < p_long).astype(int),
    must_visit=(rng.random(n) < p_must).astype(int),
)

#prep for merge
merged = pd.merge(features, df, on=key, how="right")

# placing columns to right
new_cols = ["busy", "long_stay", "must_visit"]
other_cols = [c for c in merged.columns if c not in new_cols]
merged = merged[other_cols + new_cols]
for c in new_cols:
    merged[c] = merged[c].fillna(0).astype(int).clip(0, 1)

#adding
out.parent.mkdir(parents=True, exist_ok=True)

#saving final table
merged.to_csv(out, index=False)

print(f"✅ Saved to: {out.resolve()}")
print(f"Rows: {len(merged):,} | Key: {key} | "
      f"busy=1: {merged['crowdedness'].mean():.2%}, "
      f"long_stay=1: {merged['length_of_stay'].mean():.2%}, "
      f"must_visit=1: {merged['food_quality'].mean():.2%}")

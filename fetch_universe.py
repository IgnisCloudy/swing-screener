"""Run once from an Indian connection, then commit data/nse_universe.csv."""
import os
from scan_job_v3 import get_universe
df = get_universe()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nse_universe.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} symbols to {out}")

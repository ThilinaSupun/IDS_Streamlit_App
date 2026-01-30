import pandas as pd

# Read large CSV safely in chunks
chunks = pd.read_csv(
    "Farm-Flows.csv",
    chunksize=100_000,
    low_memory=False
)

# Take the first chunk (you can increase later)
sample_df = next(chunks)

# Save smaller CSV
sample_df.to_csv("Farm-Flows-sample.csv", index=False)

print("✅ Sample file created")
print("Shape:", sample_df.shape)

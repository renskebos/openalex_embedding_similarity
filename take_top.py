import polars as pl

# Load the parquet files
parquet_file_target1 = '/Users/renskebos/Documents/AI_Master/openalex_embedding_similarity/cos_sim_target1.parquet'
parquet_file_target2 = '/Users/renskebos/Documents/AI_Master/openalex_embedding_similarity/cos_sim_target2.parquet'

# Read parquet files
df1 = pl.read_parquet(parquet_file_target1, use_pyarrow=True)
df2 = pl.read_parquet(parquet_file_target2, use_pyarrow=True)

# Take the top 10,000 papers based on similarity
top_10000_target1 = df1.sort("similarity", descending=True).head(10_000)
top_10000_target2 = df2.sort("similarity", descending=True).head(10_000)

# Combine and remove duplicates based on 'id'
combined_df = pl.concat([top_10000_target1, top_10000_target2]).unique(subset=['id'])

# Extract only the work IDs
work_ids_df = combined_df.select("id")

# Save to CSV
output_csv = '/Users/renskebos/Documents/AI_Master/openalex_embedding_similarity/work_ids.csv'
work_ids_df.write_csv(output_csv)

print(f"Work IDs opgeslagen in: {output_csv}")





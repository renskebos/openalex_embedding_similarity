import polars as pl
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

BATCH_SIZE = 100  # Process files in batches

def process_parquet_files(parquet_dir: str, target_embeddings: list[np.ndarray], target_ids: list[str], output_path: str, num_embeddings: int):
    """Process parquet files and compute cosine similarities."""
    parquet_files = list(Path(parquet_dir).rglob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}.")
        return

    accumulated_results = []

    for idx, parquet_file in enumerate(tqdm(parquet_files, desc="Processing parquet files")):
        try:
            df = pl.read_parquet(parquet_file, use_pyarrow=True).head(num_embeddings)

            if "id" not in df.columns or "embedding" not in df.columns:
                continue  # Skip files missing required columns

            ids = df["id"].to_list()
            embeddings = np.vstack(df["embedding"].to_list())

            for target_id, target_embedding in zip(target_ids, target_embeddings):
                similarities = compute_cosine_similarity_batch(target_embedding, embeddings)
                result_df = pl.DataFrame({
                    "target_id": [target_id] * len(ids),
                    "id": ids,
                    "similarity": similarities.tolist()
                })
                accumulated_results.append(result_df)

            if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == len(parquet_files):
                save_results(accumulated_results, output_path)
                accumulated_results.clear()

        except Exception:
            continue  # Skip files with errors

def compute_cosine_similarity_batch(target_embedding: np.ndarray, embeddings: np.ndarray):
    """Compute cosine similarity between a target embedding and a batch of embeddings."""
    target_norm = target_embedding / np.linalg.norm(target_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(embeddings_norm, target_norm)

def load_target_embeddings(target_parquet_paths: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Load target embeddings and their IDs from parquet files."""
    target_embeddings, target_ids = [], []

    for path in target_parquet_paths:
        df = pl.read_parquet(path, use_pyarrow=True)
        target_ids.append(df["id"][0])
        target_embeddings.append(np.array(df["embedding"][0]))

    return target_embeddings, target_ids

def save_results(results: list[pl.DataFrame], output_path: str):
    """Save results to a Parquet file."""
    if not results:
        return

    combined_df = pl.concat(results, how="vertical")

    if os.path.exists(output_path):
        existing_df = pl.read_parquet(output_path)
        combined_df = pl.concat([existing_df, combined_df], how="vertical")

    combined_df.write_parquet(output_path)

def main(target_parquet_paths: list[str], parquet_dir: str, output_path: str, num_embeddings: int = 10000):
    """Main function to process Parquet files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_embeddings, target_ids = load_target_embeddings(target_parquet_paths)
    process_parquet_files(parquet_dir, target_embeddings, target_ids, output_path, num_embeddings)

def run():
    """Run the script with predefined paths."""
    target_parquet_path = [
        "/Users/renskebos/Documents/AI_Master/openalex_embedding_similarity/target_embedding.parquet"
    ]
    parquet_dir = "/Users/renskebos/Documents/AI_Master/literature_overview/extracted_files/jinaai_jina-embeddings-v3/classification"
    output_path = "/Users/renskebos/Documents/AI_Master/literature_overview/similarities_output_combined.parquet"

    main(target_parquet_path, parquet_dir, output_path, num_embeddings=10000)

if __name__ == "__main__":
    run()
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def load_embeddings(file_path: Path) -> pd.DataFrame:
    """Load embeddings from a parquet file after validating it."""
    try:
        if not file_path.suffix == ".parquet" or file_path.stat().st_size < 100:
            raise ValueError(f"File {file_path} is not a valid or sufficiently large parquet file.")
        df = pd.read_parquet(file_path)
        required_columns = {"id", "embedding"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(df.columns)} in {file_path}")
        return df
    except Exception as e:
        logging.warning(f"Skipping file {file_path}: {e}")
        return pd.DataFrame()

def find_target_embeddings(data_dir: str, target_ids: list[str], output_parquet: str):
    """
    Find and save embeddings corresponding to the given target IDs.

    Args:
        data_dir (str): Directory containing parquet files with embeddings.
        target_ids (list[str]): List of target IDs to find.
        output_parquet (str): Path to save the found target embeddings as a parquet file.
    """
    data_path = Path(data_dir)
    parquet_files = list(data_path.rglob("*.parquet"))
    if not parquet_files:
        logging.error(f"No parquet files found in {data_dir}.")
        return

    found_embeddings = []
    found_ids_set = set()

    for file_path in tqdm(parquet_files, desc="Searching for target embeddings"):
        df = load_embeddings(file_path)
        if df.empty:
            continue
        target_rows = df[df["id"].isin(target_ids)]
        if not target_rows.empty:
            # Ensure embeddings are saved as proper vectors (lists of floats)
            target_rows = target_rows.copy()
            target_rows["embedding"] = target_rows["embedding"].apply(lambda x: x if isinstance(x, list) else list(np.array(x)))
            found_embeddings.append(target_rows)
            newly_found_ids = set(target_rows['id'].tolist()) - found_ids_set
            found_ids_set.update(newly_found_ids)
            logging.info(f"Found target IDs {list(newly_found_ids)} in {file_path}")
            # Stop if all target IDs are found
            if found_ids_set.issuperset(target_ids):
                break

    if found_embeddings:
        final_df = pd.concat(found_embeddings, ignore_index=True)
        # Save to a parquet file using Pandas to_parquet function
        final_df.to_parquet(output_parquet, index=False)
        logging.info(f"Saved target embeddings to {output_parquet}")
    else:
        logging.warning("No target embeddings found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and save target embeddings from parquet files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing embedding parquet files.")
    parser.add_argument("--output_parquet", type=str, required=True, help="Path to save the found target embeddings as a parquet file.")

    args = parser.parse_args()

    # Example target IDs
    target_ids = ["W4396509800", "W4398255947"]

    find_target_embeddings(
        data_dir=args.data_dir,
        target_ids=target_ids,
        output_parquet=args.output_parquet,
    )

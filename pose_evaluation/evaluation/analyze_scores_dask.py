import argparse
import argparse
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from dask.distributed import LocalCluster
from pathlib import Path
#         client.close()
#         cluster.close()
from pathlib import Path

# # https://docs.dask.org/en/latest/dataframe-hive.html#reading-parquet-data-with-hive-partitioning
# # https://docs.dask.org/en/stable/deploying.html?utm_source=tds&utm_medium=pyarrow-in-pandas-and-dask
# if __name__ == "__main__":
#     from dask.distributed import LocalCluster
#     cluster = LocalCluster()  # Fully-featured local Dask cluster
#     client = cluster.get_client()
#     parser = argparse.ArgumentParser(description="Analyze")
#     parser.add_argument("dataset_dir", type=Path, help="Pyarrow Dataset of scores")
#     args = parser.parse_args()
#     # Assuming your data is already in a Dask DataFrame 'ddf'
#     # as indicated by dd.read_parquet
#     try:
#         ddf = dd.read_parquet(args.dataset_dir)
#     except FileNotFoundError:
#         print(f"Error: Directory not found: {args.dataset_dir}")
#         exit(1)
#     except Exception as e:
#         print(f"Error reading parquet files: {e}")
#         exit(1)
#     # Filter the Dask DataFrame where "GLOSS_A" is equal to "GLOSS_B"
#     # filtered_ddf = ddf[ddf["GLOSS_A"] == ddf["GLOSS_B"]]
#     # # Group by "METRIC" and calculate the mean of "SCORE" on the filtered data
#     # mean_scores = filtered_ddf.groupby("METRIC", observed=True)["SCORE"].mean()
#     mean_scores = ddf.groupby("METRIC", observed=True)["SCORE"].mean()
#     print(cluster.dashboard_link)
#     # Compute the result to get a Pandas Series
#     try:
#         result = mean_scores.compute()
#         # Print the resulting mean scores for each metric
#         print(result)
#         print(len(result))
#     except Exception as e:
#         print(f"Error during computation: {e}")
#     finally:

if __name__ == "__main__":
    cluster = LocalCluster()
    client = cluster.get_client()
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument("dataset_dir", type=Path, help="Pyarrow Dataset of scores")
    args = parser.parse_args()

    try:
        ddf = dd.read_parquet(args.dataset_dir)
        metric_counts = ddf.groupby("METRIC", observed=True)["SCORE"].count().compute(scheduler="sync")
        print(metric_counts)
    except Exception as e:
        print(f"Error during computation: {e}")
    finally:
        client.close()
        cluster.close()

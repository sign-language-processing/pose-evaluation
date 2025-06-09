"""WIP script to try and add relations to dataset_dfs, depends on fixing gloss/vocabulary matching problems"""

from pathlib import Path

import pandas as pd

from pose_evaluation.evaluation.combine_semantic_and_lookalike import create_gloss_tuple
from pose_evaluation.evaluation.load_splits_and_run_metrics import combine_dataset_dfs

# /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/combine_semantic_and_lookalike.py

if __name__ == "__main__":
    in_dir = Path("/opt/home/cleong/projects/pose-evaluation/gloss_dfs_256_glosses")
    out_dir = Path("/opt/home/cleong/projects/pose-evaluation/gloss_dfs_256_glosses_with_similar_items_added")
    out_dir.mkdir(exist_ok=True)
    dataset_dfs = list(Path("/opt/home/cleong/projects/pose-evaluation/dataset_dfs").glob("*.csv"))
    relations_csv = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns/semantically_and_visually_similar_sign_pairs.csv"
    )
    relations_df = pd.read_csv(relations_csv)
    relations_df["gloss_tuple"] = relations_df.apply(create_gloss_tuple, col1="GLOSS_A", col2="GLOSS_B", axis=1)
    # Create a mapping from gloss_tuple to relation
    gloss_to_relation = dict(zip(relations_df["gloss_tuple"], relations_df["relation"]))

    relations_df_vocab = list(
        set(relations_df["GLOSS_A"].unique().tolist() + relations_df["GLOSS_B"].unique().tolist())
    )

    # prints things like
    # HOMEWORK      PAPER   Semantic     (HOMEWORK, PAPER)
    # FRIENDLY        SAD  Lookalike       (FRIENDLY, SAD)
    # LION        TIGER  BOTH       (LION, TIGER)
    print(relations_df)

    df = combine_dataset_dfs(dataset_dfs, splits=["test"])
    filtered_df = df[df["GLOSS"].isin(relations_df_vocab)]
    print(filtered_df)
    # exit()

    gloss_out_csvs = list(in_dir.glob("*_out.csv"))
    for gloss_out_csv in gloss_out_csvs:
        print("*" * 55)
        gloss_out_df = pd.read_csv(gloss_out_csv)
        gloss = gloss_out_csv.stem.split("_")[0]
        gloss_out_df["GLOSS_A"] = gloss
        gloss_out_df["gloss_tuple"] = gloss_out_df.apply(create_gloss_tuple, col1="GLOSS_A", col2="GLOSS", axis=1)

        print(
            f"Loaded {len(gloss_out_df)} from {gloss_out_csv.name}, gloss={gloss}, out-glosses count: {len(gloss_out_df['GLOSS'].unique())}"
        )

        # right here, I want to check if gloss_tuple is in relations_df["gloss_tuple"]
        # if so, relations_df["relation"] has the relation. Else set gloss_out_df["relation"] to "Neither"
        gloss_out_df["relation"] = gloss_out_df["gloss_tuple"].map(gloss_to_relation).fillna("Neither")

        if gloss in relations_df_vocab:
            print(f"{gloss} IS IN THE VOCAB")
            matching_rows = relations_df[(relations_df["GLOSS_A"] == gloss) | (relations_df["GLOSS_B"] == gloss)]
            print(matching_rows)

        # TODO: add a few (10? that are known to be similar to this)

        for rel in gloss_out_df["relation"].unique():
            rel_df = gloss_out_df[gloss_out_df["relation"] == rel]
            print(f"There are {len(rel_df)} for {gloss}, {rel}")

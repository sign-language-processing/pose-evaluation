import pandas as pd
from io import StringIO
from pathlib import Path

"""Trying to combine Sem-Lex, ASL Citizen, ASL Knowledge Graph and "Known Lookalikes" from 
"""


def create_gloss_tuple(row, col1: str, col2: str):
    gloss_1 = row[col1].upper()
    gloss_2 = row[col2].upper()
    return tuple(sorted([gloss_1, gloss_2], reverse=False))


def merge_and_check_unmatched(df, df_name: str, asl_lex_2_df, left_on="ASL-LEX Code", right_on="Code"):
    print("$" * 88)
    df[left_on] = df[left_on].str.upper()
    asl_lex_2_df[right_on] = asl_lex_2_df[right_on].str.upper()
    print(f"\nMerging ASL LEX with {df_name}, left on {left_on}, right on {right_on}")
    asl_lex_cols = list(set([right_on, "EntryID"]))
    print(f"\nColumns:")
    print(df.columns)  # includes "label"
    merged_df = df.merge(
        asl_lex_2_df[asl_lex_cols],
        left_on=left_on,
        right_on=right_on,
        how="outer",  # use outer join to see unmatched on both sides
        indicator=True,  # adds a column called '_merge'
    )
    print("\nValue Counts:")
    print(merged_df["_merge"].value_counts())

    print(f"{df_name} ONLY")
    unmatched_left = merged_df[merged_df["_merge"] == "left_only"]
    print("\nUnmatched (Left)")
    print(unmatched_left)
    print(f"\nUnmatched (Left): {left_on}")
    print(unmatched_left[left_on])

    print(f"\nASL-LEX ONLY")
    unmatched_right = merged_df[merged_df["_merge"] == "right_only"]
    print(unmatched_right)
    print(f"\nUnmatched (ASL-LEX): {right_on}")
    print(unmatched_right[right_on])


def find_weird(df: pd.DataFrame, cols: list, source: str):
    print("\n")
    # print("*" * 99)
    vocab = []

    for col in cols:
        vocab.extend(df[col].unique().tolist())
    vocab = list(set(vocab))

    print(f"## {source} has {len(vocab)} vocab items")
    weird_chars = set()
    for s in vocab:
        non_alnum_chars = [c for c in s if not c.isalnum() and c != "_"]
        non_alnum_chars = set(non_alnum_chars)
        if non_alnum_chars:
            print(f"Non-alphanumeric characters for '{s}':", non_alnum_chars)
            weird_chars.update(non_alnum_chars)
    print(f"Complete set: {weird_chars}")


if __name__ == "__main__":
    aslkg_csv = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASLKG/edges_v2_noweights.tsv")
    known_lookalikes_csv = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns/deduped_sorted_similar_gloss_pairs.csv"
    )

    asl_citizen_test_csv = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASL_Citizen/splits/test.csv"
    )
    sem_lex_csv = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/Sem-Lex/semlex_metadata.csv"
    )
    asl_lex_2_csv = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASLLEX/signdata.csv")
    out = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/local_data/SimilarSigns/semantically_and_visually_similar_sign_pairs.csv"
    )
    # Adding the ASL Knowledge Graph
    asl_knowledge_graph_df = pd.read_csv(aslkg_csv, delimiter="\t")
    # # get the "response" relation
    sem_related_df = asl_knowledge_graph_df[asl_knowledge_graph_df["relation"] == "response"]
    sem_related_df = asl_knowledge_graph_df

    sem_related_df = sem_related_df[
        sem_related_df["subject"].str.contains("asllex:") & sem_related_df["object"].str.contains("asllex:")
    ]
    print(sem_related_df[["subject", "object"]])

    # ASL-Lex Vocabulary for compatibility
    # Remove underscores unless they are immediately before digits at the end
    sem_related_df["subject"] = (
        sem_related_df["subject"]
        .str.replace("asllex:", "", regex=False)
        .str.replace("#", "", regex=False)
        # .str.upper()
        # .str.replace(r"_(?!\d+$)", "", regex=True)
    )

    sem_related_df["object"] = (
        sem_related_df["object"]
        .str.replace("asllex:", "", regex=False)
        .str.replace("#", "", regex=False)
        # .str.upper()
        # .str.replace(r"_(?!\d+$)", "", regex=True)  # oh_i_see -> ohisee, but FISHING_2 left unchanged
    )

    sem_related_df = sem_related_df[["subject", "object"]]
    sem_related_df.drop_duplicates()
    sem_related_df["gloss_tuple"] = sem_related_df.apply(create_gloss_tuple, col1="subject", col2="object", axis=1)
    sem_related_df["relation"] = "Semantic"
    sem_related_df = sem_related_df.rename(columns={"subject": "GLOSS_A", "object": "GLOSS_B"})
    find_weird(sem_related_df, cols=["GLOSS_A", "GLOSS_B"], source=f"ASLKG ({aslkg_csv.name})")

    # ASL Citizen Test Set
    asl_citizen_test_df = pd.read_csv(asl_citizen_test_csv)
    find_weird(
        asl_citizen_test_df,
        cols=["Gloss"],
        source=f"ASL Citizen Test Set ({asl_citizen_test_csv.name})",
    )

    # Sem-Lex
    sem_lex_df = pd.read_csv(sem_lex_csv)
    find_weird(sem_lex_df, cols=["label"], source=f"Sem-Lex ({sem_lex_csv.name})")

    # Sem-Lex (without freetext)
    sem_lex_df = sem_lex_df[sem_lex_df["label_type"] != "freetext"]
    find_weird(
        sem_lex_df,
        cols=["label"],
        source=f"Sem-Lex (no freetext) ({sem_lex_csv.name})",
    )

    sem_lex_df = sem_lex_df[sem_lex_df["label_type"] == "asllex"]
    find_weird(
        sem_lex_df,
        cols=["label"],
        source=f"Sem-Lex (ASL-LEX labels) ({sem_lex_csv.name})",
    )

    # asl_lex_2_csv
    # Step 1: Read and decode the raw bytes
    with open(asl_lex_2_csv, "rb") as f:
        raw = f.read()

    # Step 2: Decode with replacement for bad bytes
    decoded = raw.decode("utf-8", errors="replace")

    asl_lex_2_df = pd.read_csv(StringIO(decoded))
    # asl_lex_2_df = pd.read_csv(asl_lex_2_csv)
    find_weird(asl_lex_2_df, cols=["EntryID"], source=f"ASL Lex 2.0 ({asl_lex_2_csv})")

    print("*" * 77)

    ##########
    # Fix ASL Citizen
    # Merge the two dataframes on the matching columns
    merge_and_check_unmatched(asl_citizen_test_df, "ASL-Citizen", asl_lex_2_df)

    merge_and_check_unmatched(sem_lex_df, "Sem-Lex (ASL-LEX labels)", asl_lex_2_df, left_on="label", right_on="EntryID")
    merge_and_check_unmatched(sem_lex_df, "Sem-Lex (ASL-LEX labels)", asl_lex_2_df, left_on="label", right_on="EntryID")

    # sem_related_df
    merge_and_check_unmatched(sem_related_df, "ASLKG", asl_lex_2_df, left_on="GLOSS_A", right_on="EntryID")

    exit()

    lookalikes_df = pd.read_csv(known_lookalikes_csv)
    lookalikes_df["gloss_tuple"] = lookalikes_df.apply(create_gloss_tuple, col1="GLOSS_A", col2="GLOSS_B", axis=1)
    lookalikes_df["relation"] = "Lookalike"
    print(lookalikes_df)

    # Combine the two dataframes
    combined_df = pd.concat([sem_related_df, lookalikes_df], ignore_index=True)

    # Group by gloss_tuple and determine relation
    relation_df = (
        combined_df.groupby("gloss_tuple")["relation"]
        .agg(lambda x: "Both" if len(set(x)) > 1 else list(x)[0])
        .reset_index()
    )

    # Drop duplicate gloss_tuples to avoid multiple rows per pair
    gloss_info_df = combined_df.drop_duplicates(subset="gloss_tuple")[["gloss_tuple", "GLOSS_A", "GLOSS_B"]]

    # Merge the relation info back
    final_df = gloss_info_df.merge(relation_df, on="gloss_tuple")
    final_df["GLOSS_A"] = final_df["gloss_tuple"].apply(lambda x: x[0])
    final_df["GLOSS_B"] = final_df["gloss_tuple"].apply(lambda x: x[1])

    for relation in final_df["relation"].unique():
        rel_df = final_df[final_df["relation"] == relation]
        rel_vocab = rel_df["GLOSS_A"].unique().tolist() + rel_df["GLOSS_B"].unique().tolist()
        rel_vocab = list(set(rel_vocab))
        print(f"There are {len(rel_df)} with relation {relation}, with a total vocab of {len(rel_vocab)}")
        print(rel_df)
    print(f"Writing final to {out}")
    final_df[["GLOSS_A", "GLOSS_B", "relation"]].to_csv(out, index=False)

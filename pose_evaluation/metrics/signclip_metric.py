from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_format import Pose
from pathlib import Path
from typing import Literal
import numpy as np
import itertools
from tqdm import tqdm
from scipy.spatial.distance import cosine
import math
import pandas as pd

class SignCLIPEmbeddingDistanceMetric(PoseMetric):
    def __init__(self,   
                 model_id="baseline_temporal",
                 kind: Literal["cosine", "l2"] = "cosine",
                 higher_is_better: bool = False):
        super().__init__(name=f"SignCLIPDistanceMetric {kind}", higher_is_better=False)

        self.kind = kind

    def load_precalculated_embedding(self, saved_embedding_path:Path) -> np.ndarray:

        embedding = np.load(saved_embedding_path) # typically (1, 768)
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0] # new shape:(768, )
        return embedding
    
    def embed_pose(self, pose:Pose)->np.ndarray:
        # blocked by the fact that embedding with SignCLIP is nontrivial. 
        # See https://github.com/sign-language-processing/pose-evaluation/issues/1
        raise NotImplementedError
    
    def get_embedding(self, input: Path|np.ndarray|Pose)->np.ndarray:
        if isinstance(input, np.ndarray):
            # often (1, 768)
            if input.ndim == 2 and input.shape[0] == 1:
                input = input[0] # new shape:(768, )
        elif isinstance(input, Path):
            input = self.load_precalculated_embedding(input)
        elif isinstance(input, Pose):
            input = self.embed_pose(pose=input)

        return input


    def score(self, hypothesis: Path|np.ndarray|Pose, reference: Path|np.ndarray|Pose) -> float:
        hypothesis = self.get_embedding(hypothesis)
        reference = self.get_embedding(reference)            
        
        return cosine(hypothesis, reference)

        
    



if __name__ =="__main__":
    metric = SignCLIPEmbeddingDistanceMetric()

    # embeddings_path = Path.cwd()/"ASL_Citizen_curated_sample_with_embeddings_from_all_models"/"embeddings"
    embeddings_path = Path("/media/aqsa/Deep-Storage/colin/ASL_Citizen/embeddings/sem-lex") 
    embeddings_files = list(embeddings_path.glob("*.npy"))
    # embeddings= [metric.load_precalculated_embedding(npy_file)  for npy_file in embeddings_path.glob("*.npy")]

    print(f"Found {len(embeddings_files)} embeddings")

    
    # loaded = metric.load_precalculated_embedding('pose_evaluation/metrics/test_poses/241481900450897-HOUSE-using-model-sem-lex.npy')
    
    # print(f"That makes for {len(combinations)} combinations")
    i = 0
    entries =[]
    out_file = Path.cwd()/"signclip_scores.csv"
    pd.DataFrame(columns=["hyp","ref","score"]).to_csv(out_file, index=False)
    for embedding, other_embedding in tqdm(itertools.combinations(embeddings_files, 2),
                                           total=math.comb(len(embeddings_files), 2),
                                           desc=f"Calculating scores, writing to {out_file}"):
            score = metric.score(embedding, other_embedding)
            entry = {
                "hyp":embedding.stem.split("-")[0], # e.g. 0031311305138936874-FATHER-using-model-sem-lex.npy becomes 0031311305138936874
                "ref":other_embedding.stem.split("-")[0], 
                "score":score
            }
            entries.append(entry)
            i = i+1
            if i%1000 == 0:
                # print(f"Collected {len(entries)} scores. Writing to {out_file} resetting")
                df = pd.DataFrame.from_dict(entries)
                df.to_csv(out_file, mode="a", index=False, header=False)
                entries = []


            # i = i+1
            # if i % 1000 == 0:
            #     print(i)
            #     exit()
            # print(f"Score between {embedding.stem} and {other_embedding.stem}: {score}")
            
    
    

    

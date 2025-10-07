# Pose Evaluation

The lack of automatic pose evaluation metrics is a major obstacle in the development of
sign language generation models.

![Distribution of scores](assets/pose-eval-title-picture.png)

## Goals

The primary objective of this repository is to house a suite of
automatic evaluation metrics specifically tailored for sign language poses.
This includes metrics proposed by Ham2Pose[^1]
as well as custom-developed metrics unique to our approach.
We recognize the distinct challenges in evaluating single signs versus continuous signing,
and our methods reflect this differentiation.

---

## Usage


For a demonstration of how to use the package, see https://colab.research.google.com/drive/1Hd7dQ93GO1shCMbKORwpiG0MEsj7dtC3?usp=sharing

Demonstrates:
* How to reconstruct the metrics from our paper.
* How to use them to score poses, with signatures.
* How to score poses with different lengths, missing/undetected keypoints, or different keypoint formats.



## Quantitative Evaluation

### Isolated Sign Evaluation

Given an isolated sign corpus such as ASL Citizen[^2], we repeat the evaluation of Ham2Pose[^1] on our metrics, ranking distance metrics by retrieval performance.

Evaluation is conducted on a combined dataset of ASL Citizen, Sem-Lex[^3], and PopSign ASL[^4].

For each sign class, we use all available samples as targets and sample four times as many distractors, yielding a 1:4 target-to-distractor ratio.

For instance, for the sign _HOUSE_ with 40 samples (11 from ASL Citizen, 29 from Sem-Lex), we add 160 distractors and compute pairwise metrics from each target to all 199 other examples (We consistently discard scores for pose files where either the target or distractor could not be embedded with SignCLIP.).

Retrieval quality is measured using Mean Average Precision (`mAP↑`) and Precision@10 (`P@10↑`). The complete evaluation covers 5,362 unique sign classes and 82,099 pose sequences.

After several pilot runs, we finalized a subset of 169 sign classes with at most 20 samples each, ensuring consistent metric coverage. We evaluated 1200 distance-based variants and SignCLIP models with different checkpoints provided by the authors on this subset.

The overall results show that DTW-based metrics outperform padding-based baselines. Embedding-based methods, particularly SignCLIP models fine-tuned on in-domain ASL data, achieve the strongest retrieval scores.

<!-- Atwell style evaluations didn't get done. Nor did AUTSL -->

## Evaluation Metrics

For the study, we evaluated over 1200 Pose distance metrics, recording mAP and other retrieval performance characteristics.

We find that the top metric

### Contributing

Please make sure to run `black pose_evaluation` before submitting a pull request.

## Cite

If you use our toolkit in your research or projects, please consider citing the work.

```bib
@misc{pose-evaluation2025,
    title={Meaningful Pose-Based Sign Language Evaluation},
    author={Zifan Jiang, Colin Leong, Amit Moryossef, Anne Göhring, Annette Rios, Oliver Cory, Maksym Ivashechkin, Neha Tarigopula, Biao Zhang, Rico Sennrich, Sarah Ebling},
    howpublished={\url{https://github.com/sign-language-processing/pose-evaluation}},
    year={2025}
}
```

### Contributions

- Zifan, Colin, and Amit developed the evaluation metrics and tools. Zifan did correlation and human evaluations, Colin did automated meta-eval, KNN, etc.
- Colin and Amit developed the library code.
- Zifan, Anne, and Lisa conducted the qualitative and quantitative evaluations.

## References

[^1]: Rotem Shalev-Arkushin, Amit Moryossef, and Ohad Fried. 2022. [Ham2Pose: Animating Sign Language Notation into Pose Sequences](https://arxiv.org/abs/2211.13613).
[^2]:
    Aashaka Desai, Lauren Berger, Fyodor O. Minakov, Vanessa Milan, Chinmay Singh, Kriston Pumphrey, Richard E. Ladner, Hal Daumé III, Alex X. Lu, Naomi K. Caselli, and Danielle Bragg.  
    2023. [ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition](https://arxiv.org/abs/2304.05934).  
    _ArXiv_, abs/2304.05934.

[^3]:
    Lee Kezar, Elana Pontecorvo, Adele Daniels, Connor Baer, Ruth Ferster, Lauren Berger, Jesse Thomason, Zed Sevcikova Sehyr, and Naomi Caselli.  
    2023. [The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes](https://api.semanticscholar.org/CorpusID:263334197).  
    _Proceedings of the 25th International ACM SIGACCESS Conference on Computers and Accessibility_.

[^4]:
    Thad Starner, Sean Forbes, Matthew So, David Martin, Rohit Sridhar, Gururaj Deshpande, Sam S. Sepah, Sahir Shahryar, Khushi Bhardwaj, Tyler Kwok, Daksh Sehgal, Saad Hassan, Bill Neubauer, Sofia Anandi Vempala, Alec Tan, Jocelyn Heath, Unnathi Kumar, Priyanka Mosur, Tavenner Hall, Rajandeep Singh, Christopher Cui, Glenn Cameron, Sohier Dane, and Garrett Tanzer.  
    2023. [PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones](https://api.semanticscholar.org/CorpusID:268030720).  
    _Neural Information Processing Systems_.

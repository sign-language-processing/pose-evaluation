# Pose Evaluation

The lack of automatic pose evaluation metrics is a major obstacle in the development of
sign language generation models.

## Goals

The primary objective of this repository is to house a suite of
automatic evaluation metrics specifically tailored for sign language poses.
This includes metrics proposed by Ham2Pose[^1]
as well as custom-developed metrics unique to our approach.
We recognize the distinct challenges in evaluating single signs versus continuous signing,
and our methods reflect this differentiation.


---

# TODO:

- [ ] Qualitative Evaluation
- [ ] Quantitative Evaluation

## Qualitative Evaluation

To qualitatively demonstrate the efficacy of these evaluation metrics,
we implement a nearest-neighbor search for selected signs from the **TODO** corpus.
The rationale is straightforward: the closer the sign is to its nearest neighbor in the corpus,
the more effective the evaluation metric is in capturing the nuances of sign language transcription and translation.

### Distribution of Scores

Using a sample of the corpus, we compute the any-to-any scores for each metric.
Intuitively, we expect a good metric given any two random signs to produce a bad score, since most signs are unrelated.
This should be reflected in the distribution of scores, which should be skewed towards lower scores.

![Distribution of scores](assets/distribution/all.png)

### Nearest Neighbor Search

INSERT TABLE HERE

## Quantitative Evaluation

### Isolated Sign Evaluation

Given an isolated sign corpus such as AUTSL[^2], we repeat the evaluation of Ham2Pose[^1] on our metrics.

### Continuous Sign Evaluation

We evaluate each metric in the context of continuous signing with our continuous metrics alongside our segmented metrics
and correlate to human judgments.

## Evaluation Metrics

**TODO** list evaluation metrics here.

## Cite

If you use our toolkit in your research or projects, please consider citing the work.

```bib
@misc{pose-evaluation2024,
    title={Pose Evaluation: Metrics for Evaluating Sign Langauge Generation Models},
    author={Zifan Jiang, Colin Leong, Amit Moryossef},
    howpublished={\url{https://github.com/sign-language-processing/pose-evaluation}},
    year={2024}
}
```

#### Contributions:
- Zifan, Colin, and Amit developed the evaluation metrics and tools.
- Zifan, Anne, and Lisa conducted the qualitative and quantitative evaluations.

## References

[^1]: Rotem Shalev-Arkushin, Amit Moryossef, and Ohad Fried.
2022. [Ham2Pose: Animating Sign Language Notation into Pose Sequences](https://arxiv.org/abs/2211.13613).
[^2]: Ozge Mercanoglu Sincan and Hacer Yalim Keles.
2020. [AUTSL: A Large Scale Multi-modal Turkish Sign Language Dataset and Baseline Methods](https://arxiv.org/abs/2008.00932).

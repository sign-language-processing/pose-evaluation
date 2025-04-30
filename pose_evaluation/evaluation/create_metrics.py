import itertools
from typing import Literal, Optional, List
from pathlib import Path
import pandas as pd

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure, AggregatedPowerDistance
from pose_evaluation.metrics.nonsense_measures import Return4Measure
from pose_evaluation.metrics.dtw_metric import (
    DTWDTAIImplementationDistanceMeasure,
    DTWAggregatedScipyDistanceMeasure,
    DTWOptimizedDistanceMeasure,
    DTWAggregatedDistanceMeasure,
    DTWAggregatedPowerDistanceMeasure,
)
from pose_evaluation.metrics.pose_processors import (
    RemoveWorldLandmarksProcessor,
    HideLegsPosesProcessor,
    ReducePosesToCommonComponentsProcessor,
    TrimMeaninglessFramesPoseProcessor,
    NormalizePosesProcessor,
    GetHandsOnlyHolisticPoseProcessor,
    InterpolateAllToSetFPSPoseProcessor,
    ReduceHolisticPoseProcessor,
    ZeroPadShorterPosesProcessor,
    get_standard_pose_processors,
    FillMaskedOrInvalidValuesPoseProcessor,
    MaskInvalidValuesPoseProcessor,
    GetYoutubeASLKeypointsPoseProcessor,
    FirstFramePadShorterPosesProcessor,
    AddTOffsetsToZPoseProcessor,
)


def construct_metric(
    distance_measure: DistanceMeasure,
    default_distance=0.0,
    trim_meaningless_frames: bool = True,
    normalize: bool = True,
    sequence_alignment: Literal["zeropad", "dtw", "padwithfirstframe"] = "padwithfirstframe",
    keypoint_selection: Literal[
        "removelegsandworld", "reduceholistic", "hands", "youtubeaslkeypoints"
    ] = "removelegsandworld",
    masked_fill_value: Optional[float] = None,
    fps: Optional[int] = None,
    name: Optional[str] = None,
    z_speed: Optional[float] = None,
):
    name_pieces = []
    if name is None:
        name = ""
    pose_preprocessors = []

    if trim_meaningless_frames:
        name_pieces.append("startendtrimmed")
        pose_preprocessors.append(TrimMeaninglessFramesPoseProcessor())
    else:
        name_pieces.append("untrimmed")

    if z_speed is not None:
        name_pieces.append(f"zspeed{z_speed}")
        pose_preprocessors.append(AddTOffsetsToZPoseProcessor())

    if normalize:
        name_pieces.append("normalizedbyshoulders")
        pose_preprocessors.append(NormalizePosesProcessor())
    else:
        name_pieces.append("unnormalized")

    #########################################
    # Keypoint Selection strategy
    if keypoint_selection == "hands":
        pose_preprocessors.append(GetHandsOnlyHolisticPoseProcessor())
    elif keypoint_selection == "reduceholistic":
        pose_preprocessors.append(ReduceHolisticPoseProcessor())
    elif keypoint_selection == "youtubeaslkeypoints":
        pose_preprocessors.append(GetYoutubeASLKeypointsPoseProcessor())
    else:
        pose_preprocessors.append(RemoveWorldLandmarksProcessor())
        pose_preprocessors.append(HideLegsPosesProcessor())

    name_pieces.append(keypoint_selection)

    ######################
    # Default Distances
    name_pieces.append(f"defaultdist{default_distance}")
    distance_measure.default_distance = default_distance

    ##########################################
    # FPS Strategy
    if fps is not None:
        pose_preprocessors.append(InterpolateAllToSetFPSPoseProcessor(fps=fps))
        name_pieces.append(f"interp{fps}")
    else:
        name_pieces.append("nointerp")

    ################################################
    # Sequence Alignment
    # Only can go AFTER things that change the length like Interpolate
    # if not then it's probably dtw, so do nothing
    if sequence_alignment == "zeropad":
        pose_preprocessors.append(ZeroPadShorterPosesProcessor())
    elif sequence_alignment == "padwithfirstframe":
        pose_preprocessors.append(FirstFramePadShorterPosesProcessor())

    name_pieces.append(sequence_alignment)

    ###########################################################
    # Masked and/or Invalid Values Strategy
    if masked_fill_value is not None:
        pose_preprocessors.append(FillMaskedOrInvalidValuesPoseProcessor(masked_fill_value))
        name_pieces.append(f"fillmasked{masked_fill_value}")
    else:
        name_pieces.append("maskInvalidVals")
        pose_preprocessors.append(MaskInvalidValuesPoseProcessor())

    ###################################################################
    # Components/points Alignment
    pose_preprocessors.append(ReducePosesToCommonComponentsProcessor())
    if "Measure" in distance_measure.name:
        name = f"{distance_measure.name}".replace("Measure", "Metric")
    else:
        name = f"{distance_measure.name}Metric"

    name = "_".join(name_pieces) + "_" + name

    return DistanceMetric(name=name, distance_measure=distance_measure, pose_preprocessors=pose_preprocessors)


def get_metrics(measures: List[DistanceMeasure] = None, include_return4=True, metrics_out: Path = None):
    metrics = []

    if measures is None:
        measures = [
            DTWDTAIImplementationDistanceMeasure(name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True),
            # DTWDTAIImplementationDistanceMeasure(name="dtaiDTWAggregatedDistanceMeasureSlow", use_fast=False), # super slow
            # DTWOptimizedDistanceMeasure(),
            # DTWAggregatedPowerDistanceMeasure(),
            # DTWAggregatedScipyDistanceMeasure(),
            AggregatedPowerDistance(),
        ]
    measure_names = [measure.name for measure in measures]
    assert len(set(measure_names)) == len(measure_names)

    z_speeds = [None, 0.1, 1.0, 4.0]

    default_distances = [
        0.0,
        1.0,
        10.0,
    ]

    # round one precision@10 shows 0.0 seems to win, top 8/10 metrics
    # but 1.0 could also work.
    # Estimated effect on 'precision@10' of 'fillmasked0.0': +0.0048
    # Estimated effect on 'precision@10' of 'fillmasked1.0': +0.0115
    # fillmasked0.0 count within top 10 by precision@10: 8
    # fillmasked0.0 count within top 5 by precision@10: 5
    # fillmasked1.0 count within top 10 by precision@10: 2
    # fillmasked1.0 count within top 5 by precision@10: 0
    # but on recall...
    # Estimated effect on 'recall@10' of 'fillmasked0.0': -0.0028
    # Estimated effect on 'recall@10' of 'fillmasked1.0': +0.0055
    # fillmasked0.0 count within top 10 by recall@10: 3
    # fillmasked0.0 count within top 5 by recall@10: 0
    # fillmasked1.0 count within top 10 by recall@10: 4
    # fillmasked1.0 count within top 5 by recall@10: 4
    masked_fill_values = [
        # None, # leads to nan values
        0.0,  # top 8/10 metrics
        1.0,  #
        10.0,
    ]  # technically could also do None, but that leads to nan values slipping through
    # trim_values = [True, False]

    # round one:
    # Trimming seems to help - Untrimmed generally worse, though not a whole lot:
    # Estimated effect on 'recall@10' of 'untrimmed': -0.0103
    # Estimated effect on 'precision@10' of 'untrimmed': -0.0125
    # Estimated effect on 'mean_score_time' of 'untrimmed': +0.0087
    # So trimmed is better...but not a whole lot
    trim_values = [True, False]

    # round one:
    # Normalizing seems to be worth it.
    # Actually improves mean out/mean in if you don't normalize
    # Estimated effect on 'mean_score_time' of 'unnormalized': -0.0018
    # Estimated effect on 'precision@10' of 'unnormalized': -0.0020
    # And none of the top 10 are unnormalized
    normalize_values = [True, False]

    sequence_alignment_strategies = ["zeropad", "padwithfirstframe", "dtw"]

    # round one precision@10, recall@10 clearly that hands-only seems to work best
    # improves mean_score_time as well
    # Estimated effect on 'precision@10' of 'hands': +0.0791
    # hands count within top 100 by precision@10: 91
    # hands count within top 10 by precision@10: 10
    # hands count within top 5 by precision@10: 5
    # and it's similar for recall
    keypoint_selection_strategies = [
        "removelegsandworld",
        "reduceholistic",
        "hands",
        "youtubeaslkeypoints",
    ]
    fps_values = [
        None,
        15,
        30,
        45,
        60,
        120,
    ]
    # round one results suggest interp doesn't help, and takes longer
    # fps_values = [None]

    # Create all combinations
    metric_combinations = itertools.product(
        measures,
        z_speeds,
        default_distances,
        trim_values,
        normalize_values,
        keypoint_selection_strategies,
        fps_values,
        masked_fill_values,
        sequence_alignment_strategies,
    )

    constructed = []

    # Iterate over them
    for (
        measure,
        z_speed,
        default_distance,
        trim,
        normalize,
        strategy,
        fps,
        masked_fill_value,
        sequence_alignment,
    ) in metric_combinations:

        #############
        # DTW vs other sequence alignments
        # DTW metrics don't use a pose preprocessor, they handle it internally in the DistanceMeasure.
        # so we need to catch that.
        if "dtw" in measure.name.lower() and sequence_alignment != "dtw":
            # we don't want double sequence alignment strategies. No need to do zeropad with dtw
            continue

        if sequence_alignment == "dtw" and "dtw" not in measure.name.lower():
            # doesn't work, creates "dtw" metrics that just fail with ValueError:
            # e.g. "operands could not be broadcast together with shapes (620,1,48,3) (440,1,48,3)""
            continue

        # print(
        #     f"Measure: {measure.name}, Z Speed:{z_speed}, Default Distance: {default_distance}, Trim: {trim}, Normalize: {normalize}, Strategy: {strategy}, FPS: {fps}, Masked Fill: {masked_fill_value}, Sequence Alignment: {sequence_alignment}"
        # )

        metric = construct_metric(
            distance_measure=measure,
            z_speed=z_speed,
            default_distance=default_distance,
            trim_meaningless_frames=trim,
            normalize=normalize,
            sequence_alignment=sequence_alignment,
            keypoint_selection=strategy,
            fps=fps,
            masked_fill_value=masked_fill_value,
        )
        metrics.append(metric)
        constructed.append(
            {
                "measure_name": measure.name,
                "default_distance": default_distance,
                "trim": trim,
                "normalize": normalize,
                "keypoint_selection_strategy": strategy,
                "fps": fps or "nointerp",
                "masked_fill_value": masked_fill_value,
                "sequence_alignment": sequence_alignment,
                "metric_name": metric.name,
                "metric_signature": metric.get_signature().format(),
            }
        )

    # baseline/nonsense measures
    if include_return4:
        metrics.append(DistanceMetric(name="Return4Metric", distance_measure=Return4Measure(), pose_preprocessors=[]))

    metric_names = [metric.name for metric in metrics]
    metric_names_set = set(metric_names)
    assert len(metric_names_set) == len(metric_names)

    if metrics_out:
        df = pd.DataFrame(constructed)
        # metrics_out.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        df.to_csv(metrics_out, index=False)
        print(f"Saved metric configurations to {metrics_out}")

        for column in df.columns:
            uniques = df[column].unique()
            if len(uniques) < 100:
                print(f"{len(uniques)} values for {column}: {uniques.tolist()}")

    return metrics


if __name__ == "__main__":
    metrics = get_metrics(metrics_out="constructed.csv", include_return4=False)
    print(f"Current settings result in the construction of {len(metrics)} metrics")

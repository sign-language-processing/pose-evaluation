import itertools
from typing import Literal, Optional, List

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure, AggregatedPowerDistance
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
)


def construct_metric(
    distance_measure: DistanceMeasure,
    trim_meaningless_frames: bool = True,
    normalize: bool = True,
    sequence_alignment: Literal["zeropad", "dtw"] = "zeropad",
    keypoint_selection: Literal["removelegsandworld", "reduceholistic", "hands"] = "removelegsandworld",
    fps: Optional[int] = None,
    name: Optional[str] = None,
):
    name_pieces = []
    if name is None:
        name = ""
    pose_preprocessors = []

    if trim_meaningless_frames:
        name_pieces.append("trimmed")
        pose_preprocessors.append(TrimMeaninglessFramesPoseProcessor())

    if normalize:
        name_pieces.append("normalized")
        pose_preprocessors.append(NormalizePosesProcessor())

    if keypoint_selection == "hands":
        pose_preprocessors.append(GetHandsOnlyHolisticPoseProcessor())

    elif keypoint_selection == "reduceholistic":
        pose_preprocessors.append(ReduceHolisticPoseProcessor())
    else:
        pose_preprocessors.append(RemoveWorldLandmarksProcessor())
        pose_preprocessors.append(HideLegsPosesProcessor())

    name_pieces.append(keypoint_selection)

    if sequence_alignment == "zeropad":
        pose_preprocessors.append(ZeroPadShorterPosesProcessor())

    name_pieces.append(sequence_alignment)

    if fps is not None:
        pose_preprocessors.append(InterpolateAllToSetFPSPoseProcessor(fps=fps))
        name_pieces.append(f"interp{fps}")
    else:
        name_pieces.append("nointerp")

    pose_preprocessors.append(ReducePosesToCommonComponentsProcessor())

    if "Measure" in distance_measure.name:
        name = f"{distance_measure.name}".replace("Measure", "Metric")
    else:
        name = f"{distance_measure.name}Metric"

    name = "_".join(name_pieces) + "_" + name

    return DistanceMetric(name=name, distance_measure=distance_measure, pose_preprocessors=pose_preprocessors)


def get_metrics(measures: List[DistanceMeasure] = None):
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
    trim_values = [True, False]
    normalize_values = [True, False]
    keypoint_selection_strategies = ["removelegsandworld", "reduceholistic", "hands"]
    fps_values = [None, 15, 120]

    # Create all combinations
    metric_combinations = itertools.product(
        measures, trim_values, normalize_values, keypoint_selection_strategies, fps_values
    )

    # Iterate over them
    for measure, trim, normalize, strategy, fps in metric_combinations:
        print(f"Measure: {measure.name}, Trim: {trim}, Normalize: {normalize}, Strategy: {strategy}, FPS: {fps}")

        sequence_alignment = "zeropad"
        if "dtw" in measure.name.lower():
            sequence_alignment = "dtw"

        metric = construct_metric(
            distance_measure=measure,
            trim_meaningless_frames=trim,
            normalize=normalize,
            sequence_alignment=sequence_alignment,
            keypoint_selection=strategy,
            fps=fps,
        )
        metrics.append(metric)

    metric_names = [metric.name for metric in metrics]
    metric_names_set = set(metric_names)
    assert len(metric_names_set) == len(metric_names)
    return metrics

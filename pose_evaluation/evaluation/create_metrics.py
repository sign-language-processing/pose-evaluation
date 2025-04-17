import itertools
from typing import Literal, Optional, List

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
)


def construct_metric(
    distance_measure: DistanceMeasure,
    default_distance=0.0,
    trim_meaningless_frames: bool = True,
    normalize: bool = True,
    sequence_alignment: Literal["zeropad", "dtw"] = "zeropad",
    keypoint_selection: Literal["removelegsandworld", "reduceholistic", "hands"] = "removelegsandworld",
    masked_fill_value: Optional[float] = None,
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
    else:
        name_pieces.append("untrimmed")

    if normalize:
        name_pieces.append("normalized")
        pose_preprocessors.append(NormalizePosesProcessor())
    else:
        name_pieces.append("unnormalized")

    #########################################
    # Keypoint Selection strategy
    if keypoint_selection == "hands":
        pose_preprocessors.append(GetHandsOnlyHolisticPoseProcessor())
    elif keypoint_selection == "reduceholistic":
        pose_preprocessors.append(ReduceHolisticPoseProcessor())
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

    default_distances = [0.0, 1.0, 10.0]
    masked_fill_values = [
        # None,
        0.0,
        1.0,
        10.0,
    ]  # technically could also do None, but that leads to nan values slipping through
    trim_values = [True, False]
    normalize_values = [True, False]
    keypoint_selection_strategies = ["removelegsandworld", "reduceholistic", "hands"]
    fps_values = [None, 15, 120]

    # Create all combinations
    metric_combinations = itertools.product(
        measures,
        default_distances,
        trim_values,
        normalize_values,
        keypoint_selection_strategies,
        fps_values,
        masked_fill_values,
    )

    # Iterate over them
    for measure, default_distance, trim, normalize, strategy, fps, masked_fill_value in metric_combinations:
        print(
            f"Measure: {measure.name}, Default: {default_distance}, Trim: {trim}, Normalize: {normalize}, Strategy: {strategy}, FPS: {fps}, Masked Fill: {masked_fill_value}"
        )

        sequence_alignment = "zeropad"
        if "dtw" in measure.name.lower():
            sequence_alignment = "dtw"

        metric = construct_metric(
            distance_measure=measure,
            default_distance=default_distance,
            trim_meaningless_frames=trim,
            normalize=normalize,
            sequence_alignment=sequence_alignment,
            keypoint_selection=strategy,
            fps=fps,
            masked_fill_value=masked_fill_value,
        )
        metrics.append(metric)


    # baseline/nonsense measures
    metrics.append(DistanceMetric(name="Return4Metric", distance_measure=Return4Measure(), pose_preprocessors=[]))

    metric_names = [metric.name for metric in metrics]
    metric_names_set = set(metric_names)
    assert len(metric_names_set) == len(metric_names)


    return metrics

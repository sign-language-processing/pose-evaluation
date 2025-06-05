"""Creates metrics by composing various settings and preprocessors together with DistanceMeasures. 

They are given names based on the settings. E.g. 'trimmed_dtw' would be dynamic time warping with trimming."""
import itertools
from typing import Literal, Optional, List
from pathlib import Path
import copy
import re

import pandas as pd

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure, AggregatedPowerDistance
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
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
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import DatasetDFCol


# --- Constants & Regexes ------------------------------------------------
# Signature: default_distance:<float>
_SIGNATURE_RE = re.compile(r"default_distance:([\d.]+)")
# metric: defaultdist<float>
_DEFAULTDIST_RE = re.compile(r"defaultdist([\d.]+)")


def extract_signature_distance(signature: str) -> Optional[str]:
    """
    From a signature string, extract the float following 'default_distance:'.
    Returns None if not found.
    """
    m = _SIGNATURE_RE.search(signature)
    return float(m.group(1)) if m else None


def extract_metric_name_dist(metric_name: str) -> Optional[float]:
    """
    From a metric_name, extract the float following 'defaultdist'.
    Returns None if not found.
    """
    m = _DEFAULTDIST_RE.search(metric_name)
    return float(m.group(1)) if m else None


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
    reduce_poses_to_common_components: bool= True
):
    distance_measure = copy.deepcopy(distance_measure)
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
    distance_measure.set_default_distance(default_distance)
    assert (
        f"default_distance:{default_distance}" in distance_measure.get_signature().format()
    ), f"{distance_measure.default_distance}, {default_distance}"

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
        name_pieces.append("maskinvalidvals")
        pose_preprocessors.append(MaskInvalidValuesPoseProcessor())

    ###################################################################
    # Components/points Alignment
    if reduce_poses_to_common_components:
        pose_preprocessors.append(ReducePosesToCommonComponentsProcessor())



    if "Measure" in distance_measure.name:
        name = f"{distance_measure.name}".replace("Measure", "Metric")
    else:
        name = f"{distance_measure.name}Metric"

    name = "_".join(name_pieces) + "_" + name

    return DistanceMetric(name=name, distance_measure=distance_measure, pose_preprocessors=pose_preprocessors)


def get_embedding_metrics(df: pd.DataFrame) -> List:
    print(f"Getting embedding_metrics from df with {df.columns}")
    if DatasetDFCol.EMBEDDING_MODEL in df:
        for model in df[DatasetDFCol.EMBEDDING_MODEL].unique().tolist():
            yield EmbeddingDistanceMetric(model=f"{model}")
    else:
        raise ValueError(f"No {DatasetDFCol.EMBEDDING_MODEL}")


def get_metrics(measures: List[DistanceMeasure] = None, include_return4=True, metrics_out: Path = None, include_masked:bool=False):
    metrics = []

    if measures is None:
        measures = [
            DTWDTAIImplementationDistanceMeasure(name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True),
            # DTWDTAIImplementationDistanceMeasure(
            #     name="dtaiDTWAggregatedDistanceMeasureSlow", use_fast=False
            # # ),  # super slow
            # DTWOptimizedDistanceMeasure(),
            # DTWAggregatedPowerDistanceMeasure(),
            # DTWAggregatedScipyDistanceMeasure(),
            AggregatedPowerDistance(),
        ]
    measure_names = [measure.name for measure in measures]
    assert len(set(measure_names)) == len(measure_names)

    z_speeds = [None, 0.1, 1.0, 4.0, 100.0, 1000.0]

    default_distances = [
        0.0,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]

    
    masked_fill_values = [
        # None, # leads to nan values

        0.0,  # top 8/10 metrics
        1.0,  #
        10.0,
        100.0,
        1000.0,
    ]  # technically could also do None, but that leads to nan values slipping through

    if include_masked:
        masked_fill_values.append(None)
        

    
    trim_values = [True, False]

   
    normalize_values = [True, False]

    sequence_alignment_strategies = ["zeropad", "padwithfirstframe", "dtw"]

    
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

        n, s = metric.name, metric.get_signature().format()

        if "defaultdist0.0" in n:
            assert "default_distance:0.0" in s, f"{n}\n{s}\n{measure}"

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
        metrics.append(
            DistanceMetric(
                name="Return4Metric_defaultdist4.0", distance_measure=Return4Measure(), pose_preprocessors=[]
            )
        )

    metric_names = [metric.name for metric in metrics]
    metric_sigs = [metric.get_signature().format() for metric in metrics]
    metric_names_set = set(metric_names)
    metric_sigs_set = set(metric_sigs)
    assert len(metric_names_set) == len(metric_names)
    assert len(metric_sigs_set) == len(metric_sigs)

    for m_name, m_sig in zip(metric_names, metric_sigs):
        sig_distance = extract_signature_distance(m_sig)
        try:
            name_distance = extract_metric_name_dist(m_name)
        except IndexError as e:
            print(f"{e} on {m_name}, {m_sig}")
            raise e
        assert sig_distance == name_distance, f"defaultdist for {m_name} does not match signature {m_sig}"

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
    metrics = get_metrics(metrics_out="constructed.csv", include_return4=True)
    metric_names = [m.name for m in metrics]
    metric_sigs = [m.get_signature().format() for m in metrics]

    print(f"Current settings result in the construction of {len(metrics)} metrics")
    print(f"{len(set(metric_names))} unique metric names")
    print(f"{len(set(metric_sigs))} unique metric signatures")

    for n, s in zip(metric_names, metric_sigs):
        if "defaultdist0.0" in n:
            assert "default_distance:0.0" in s, f"{n}\n{s}"

        if "return4" in n.lower():
            print(n)

    tiny_csv_for_testing = Path("/opt/home/cleong/projects/pose-evaluation/tiny_csv_for_testing/asl-citizen.csv")
    # tiny_csv_for_testing = Path(
    #     "/opt/home/cleong/projects/pose-evaluation/tiny_csv_for_testing/asl-citizen-no-embeddings.csv"
    # )
    df = pd.read_csv(tiny_csv_for_testing)
    try:
        embedding_metrics = list(get_embedding_metrics(df))
        for embedding_metric in embedding_metrics:
            print(embedding_metric)
            print(embedding_metric.name)
    except ValueError as e:
        print(e)

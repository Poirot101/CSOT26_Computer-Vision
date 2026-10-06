"""Multi-object pedestrian tracking for dense crowds.

A small, dependency-light implementation of the pieces a tracking submission
needs, written to be read: box geometry, a Kalman motion model, Hungarian data
association, the ByteTrack and SORT trackers, and the HOTA / IDF1 / CLEAR
metrics implemented from their source papers.
"""

from .boxes import iou_matrix, tlwh_to_xyah, tlwh_to_xyxy, xyah_to_tlwh, xyxy_to_tlwh
from .bytetrack import ByteTracker, ByteTrackConfig
from .detect import Detections, GroundTruthDetector, YoloDetector
from .interpolate import interpolate_tracks
from .kalman import KalmanFilterXYAH
from .mot_io import SequenceInfo, discover_sequences, read_mot_file, read_seqinfo, write_mot_file
from .pipeline import RunStats, run_sequence
from .sort import SortConfig, SortTracker
from .track import STrack, TrackState

__version__ = "1.0.0"

__all__ = [
    "ByteTracker",
    "ByteTrackConfig",
    "SortTracker",
    "SortConfig",
    "KalmanFilterXYAH",
    "STrack",
    "TrackState",
    "Detections",
    "YoloDetector",
    "GroundTruthDetector",
    "SequenceInfo",
    "read_seqinfo",
    "read_mot_file",
    "write_mot_file",
    "discover_sequences",
    "run_sequence",
    "RunStats",
    "interpolate_tracks",
    "iou_matrix",
    "tlwh_to_xyxy",
    "xyxy_to_tlwh",
    "tlwh_to_xyah",
    "xyah_to_tlwh",
    "__version__",
]

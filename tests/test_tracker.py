"""
Tests for tracker module
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestBaseTrack:
    """Tests for basetrack module"""

    def test_track_state_enum(self):
        """Test TrackState enum values"""
        from tracker.basetrack import TrackState

        assert TrackState.New == 0
        assert TrackState.Tracked == 1
        assert TrackState.Lost == 2
        assert TrackState.Removed == 3

    def test_base_track_init(self):
        """Test BaseTrack initialization"""
        from tracker.basetrack import BaseTrack

        track = BaseTrack()
        assert track.track_id == 0
        assert track.is_activated is False
        assert track.state == 0  # New

    def test_base_track_mark_lost(self):
        """Test marking track as lost"""
        from tracker.basetrack import BaseTrack, TrackState

        track = BaseTrack()
        track.mark_lost()
        assert track.state == TrackState.Lost

    def test_base_track_mark_removed(self):
        """Test marking track as removed"""
        from tracker.basetrack import BaseTrack, TrackState

        track = BaseTrack()
        track.mark_removed()
        assert track.state == TrackState.Removed

    def test_base_track_end_frame(self):
        """Test BaseTrack end_frame property"""
        from tracker.basetrack import BaseTrack

        track = BaseTrack()
        track.frame_id = 10
        assert track.end_frame == 10


class TestMatching:
    """Tests for matching module"""

    def test_iou_distance(self):
        """Test IoU distance calculation"""
        from tracker.matching import iou_distance

        # Create two sets of boxes
        atlbrs = [np.array([0, 0, 100, 100])]
        btlbrs = [np.array([50, 50, 150, 150])]

        dist = iou_distance(atlbrs, btlbrs)
        assert dist.shape == (1, 1)
        assert 0 < dist[0, 0] < 1

    def test_iou_distance_empty(self):
        """Test IoU distance with empty inputs"""
        from tracker.matching import iou_distance

        dist = iou_distance([], [])
        assert dist.size == 0

    def test_iou_distance_identical(self):
        """Test IoU distance with identical boxes"""
        from tracker.matching import iou_distance

        box = np.array([0, 0, 100, 100])
        dist = iou_distance([box], [box])
        assert dist[0, 0] == 0.0  # IoU = 1, distance = 0

    def test_iou_distance_multiple(self):
        """Test IoU distance with multiple boxes"""
        from tracker.matching import iou_distance

        atlbrs = [
            np.array([0, 0, 100, 100]),
            np.array([200, 200, 300, 300])
        ]
        btlbrs = [
            np.array([50, 50, 150, 150]),
            np.array([250, 250, 350, 350])
        ]

        dist = iou_distance(atlbrs, btlbrs)
        assert dist.shape == (2, 2)

    def test_bbox_ious_function(self):
        """Test bbox_ious function"""
        from tracker.matching import bbox_ious

        atlbrs = [np.array([0, 0, 100, 100])]
        btlbrs = [np.array([50, 50, 150, 150])]

        iou_matrix = bbox_ious(atlbrs, btlbrs)
        assert iou_matrix.shape == (1, 1)
        assert 0 < iou_matrix[0, 0] < 1

    def test_linear_assignment(self):
        """Test linear assignment"""
        from tracker.matching import linear_assignment

        cost_matrix = np.array([
            [0.1, 0.9],
            [0.8, 0.2]
        ])

        try:
            matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh=0.5)
            # linear_assignment returns matches, unmatched_a, unmatched_b
            assert isinstance(matches, np.ndarray) or isinstance(matches, list)
        except ImportError:
            # lap library not installed
            pytest.skip("lap library not installed")

    def test_fuse_score(self):
        """Test fuse_score function"""
        from tracker.matching import fuse_score

        cost_matrix = np.array([[0.5]])
        # Create mock detection with score
        mock_det = MagicMock()
        mock_det.score = 0.9

        fused = fuse_score(cost_matrix, [mock_det])
        assert fused.shape == (1, 1)


class TestKalmanFilter:
    """Tests for kalman_filter module"""

    def test_kalman_filter_init(self):
        """Test KalmanFilter initialization"""
        from tracker.kalman_filter import KalmanFilter

        kf = KalmanFilter()
        assert kf is not None

    def test_kalman_filter_predict(self):
        """Test Kalman filter predict"""
        from tracker.kalman_filter import KalmanFilter

        kf = KalmanFilter()
        mean = np.zeros(8)
        covariance = np.eye(8)

        new_mean, new_cov = kf.predict(mean, covariance)
        assert new_mean.shape == (8,)
        assert new_cov.shape == (8, 8)

    def test_kalman_filter_project(self):
        """Test Kalman filter project"""
        from tracker.kalman_filter import KalmanFilter

        kf = KalmanFilter()
        mean = np.zeros(8)
        covariance = np.eye(8)

        projected_mean, projected_cov = kf.project(mean, covariance)
        assert projected_mean.shape == (4,)
        assert projected_cov.shape == (4, 4)


class TestBYTETracker:
    """Tests for byte_tracker module"""

    def test_byte_tracker_args(self):
        """Test BYTETrackerArgs"""
        from tracker.byte_tracker import BYTETrackerArgs

        args = BYTETrackerArgs()
        assert args.track_thresh == 0.5
        assert args.track_buffer == 30
        assert args.match_thresh == 0.8
        assert args.mot20 is False

    def test_byte_tracker_args_custom(self):
        """Test BYTETrackerArgs with custom values"""
        from tracker.byte_tracker import BYTETrackerArgs

        args = BYTETrackerArgs(
            track_thresh=0.3,
            track_buffer=60,
            match_thresh=0.9,
            mot20=True
        )
        assert args.track_thresh == 0.3
        assert args.track_buffer == 60
        assert args.match_thresh == 0.9
        assert args.mot20 is True

    def test_byte_tracker_init(self):
        """Test BYTETracker initialization"""
        from tracker.byte_tracker import BYTETracker, BYTETrackerArgs

        args = BYTETrackerArgs()
        tracker = BYTETracker(args=args, frame_rate=30)
        assert tracker is not None
        assert tracker.frame_id == 0

    def test_byte_tracker_update_empty(self):
        """Test BYTETracker update with empty detections"""
        from tracker.byte_tracker import BYTETracker, BYTETrackerArgs

        args = BYTETrackerArgs()
        tracker = BYTETracker(args=args, frame_rate=30)

        # Empty detections
        detections = np.array([]).reshape(0, 6)
        img_info = (480, 640)
        img_size = (480, 640)

        targets = tracker.update(detections, img_info, img_size)
        assert targets == []

    def test_byte_tracker_with_detections(self):
        """Test BYTETracker with mock detections"""
        from tracker.byte_tracker import BYTETracker, BYTETrackerArgs

        args = BYTETrackerArgs(track_thresh=0.5)
        tracker = BYTETracker(args=args, frame_rate=30)

        # Create mock detections: [x1, y1, x2, y2, score, cls]
        detections = np.array([
            [100, 100, 200, 200, 0.9, 2],
            [300, 300, 400, 400, 0.8, 2],
        ], dtype=np.float32)

        img_info = (480, 640)
        img_size = (480, 640)

        targets = tracker.update(detections, img_info, img_size)
        # Should return some targets
        assert isinstance(targets, list)

    def test_byte_tracker_increment_frame(self):
        """Test BYTETracker frame increment"""
        from tracker.byte_tracker import BYTETracker, BYTETrackerArgs

        args = BYTETrackerArgs()
        tracker = BYTETracker(args=args, frame_rate=30)

        # Update multiple times
        detections = np.array([]).reshape(0, 6)
        img_info = (480, 640)
        img_size = (480, 640)

        tracker.update(detections, img_info, img_size)
        assert tracker.frame_id == 1

        tracker.update(detections, img_info, img_size)
        assert tracker.frame_id == 2

    def test_s_track_init(self):
        """Test STrack initialization"""
        from tracker.byte_tracker import STrack

        tlwh = np.array([100, 100, 200, 200])
        score = 0.9

        track = STrack(tlwh, score)
        assert track is not None
        assert track.score == score

    def test_s_track_activate(self):
        """Test STrack activate"""
        from tracker.byte_tracker import STrack, KalmanFilter

        tlwh = np.array([100, 100, 200, 200])
        score = 0.9
        kalman_filter = KalmanFilter()

        track = STrack(tlwh, score)
        track.activate(kalman_filter, 1)
        assert track.is_activated is True
        assert track.track_id >= 0

    def test_s_track_update(self):
        """Test STrack update"""
        from tracker.byte_tracker import STrack, KalmanFilter

        tlwh = np.array([100, 100, 200, 200])
        score = 0.9
        kalman_filter = KalmanFilter()

        track = STrack(tlwh, score)
        track.activate(kalman_filter, 1)

        new_tlwh = np.array([105, 105, 205, 205])
        new_track = STrack(new_tlwh, 0.85)
        track.update(new_track, 2)  # update takes (new_track, frame_id)
        assert track.score == 0.85

    def test_s_track_properties(self):
        """Test STrack properties"""
        from tracker.byte_tracker import STrack

        tlwh = np.array([100, 100, 200, 200])
        track = STrack(tlwh, 0.9)

        # Test tlbr property
        tlbr = track.tlbr
        assert len(tlbr) == 4
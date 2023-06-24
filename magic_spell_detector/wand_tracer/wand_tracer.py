import time
import math
import logging
import enum
import sys
from collections import deque
from typing import Tuple, Any, Optional
import numpy as np
import cv2


TRACE_FRAME_WIDTH = 640
if sys.platform.startswith('darwin'):
    TRACE_FRAME_HEIGHT = 360
else:
    TRACE_FRAME_HEIGHT = 480
TRACEPOINTS_DEQ_SIZE = 40
MAX_TRACE_SPEED = 400   # pixels/sec (30p/0.2sec)
BLANK_TIME_GAP = 2          # sec
REQUIRED_DEQUE_SIZE = TRACEPOINTS_DEQ_SIZE - 5

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class TracePoint:
    def __init__(self, keypoint: cv2.KeyPoint):
        self.x: int = int(keypoint.pt[0])
        self.y: int = int(keypoint.pt[1])

    def distance_to(self, other_pt):
        return math.hypot((self.x-other_pt.x), (self.y-other_pt.y))

    @property
    def tup(self) -> Tuple[int, int]:
        return self.x, self.y


class TraceStatus(enum.Enum):
    READY_FOR_SPELLCHECK = enum.auto()
    TRACING = enum.auto()
    WAITING_FOR_TRACE = enum.auto()
    NO_TRACE = enum.auto()
    INVALID_TRACE = enum.auto()


def _set_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 200
    params.maxThreshold = 255

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.maxArea = 400

    params.filterByCircularity = True
    params.minCircularity = 0.3

    params.filterByConvexity = True
    params.minConvexity = 0.7

    params.filterByInertia = False

    return cv2.SimpleBlobDetector_create(params)


def blank_trace_frame():
    return np.zeros(shape=[TRACE_FRAME_HEIGHT, TRACE_FRAME_WIDTH], dtype=np.uint8)


class WandTracer:
    def __init__(self):
        self._back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=33)
        self._detector = _set_blob_detector()
        self._detected_blob: bool = False
        self._tracepoints: deque[TracePoint] = deque(maxlen=TRACEPOINTS_DEQ_SIZE)
        self._draw_frame: np.ndarray = blank_trace_frame()
        self._last_keypoint_time = 0

    def _detect_wand(self, frame) -> Optional[cv2.KeyPoint]:
        """ Detect wand tip using blob detector

        :param frame: grayscale camera feed
        :return: the first keypoint found
        """
        sub_frame = self.bg_subtract(frame)
        if not np.any(sub_frame):
            return None
        keypoints = self._detector.detect(sub_frame)
        # frame_w_keypoints = cv2.drawKeypoints(sub_frame, keypoints, np.array([]),
        #                                       (0, 0, 255),
        #                                       cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        return keypoints[0] if len(keypoints) != 0 else None

    def trace_wand(self, frame) -> np.ndarray:
        """ Create a trace from wans tip movement.

        This trace checks for speed of wand movement and area covered to make sure that
        it does not include a spurious point.
        :param frame: grayscale camera feed
        :return: frame containing the trace
        """
        keypoint = self._detect_wand(frame)
        if keypoint is None:
            self._detected_blob = False
        else:
            self._detected_blob = True
            new_tracepoint = TracePoint(keypoint)
            now = time.time()

            if len(self._tracepoints) != 0:
                """ If a tracepoints already exist, continue to draw a trace. """
                elapsed = now - self._last_keypoint_time
                movt_speed = self._tracepoints[-1].distance_to(new_tracepoint) / elapsed
                if movt_speed >= MAX_TRACE_SPEED:
                    """ Escape if draw speed is too high. """
                    return self._draw_frame

                # print(f"Movement speed: {movt_speed}")
                if len(self._tracepoints) > TRACEPOINTS_DEQ_SIZE:
                    """ If deque is full, remove the oldest entry. """
                    self._tracepoints.popleft()

                cv2.line(self._draw_frame,
                         self._tracepoints[-1].tup, new_tracepoint.tup,
                         (255, 255, 255), 8)
                self._tracepoints.append(new_tracepoint)
            else:
                """ First tracepoint. """
                self._tracepoints.append(new_tracepoint)
            self._last_keypoint_time = now
        return self._draw_frame

    @property
    def draw_frame(self):
        return self._draw_frame

    def bg_subtract(self, frame):
        if frame is None:
            return
        fg_mask = self._back_sub.apply(frame)
        frame2 = cv2.bitwise_and(frame, frame, mask=fg_mask)
        return frame2

    def get_trace_status(self) -> TraceStatus:
        """ Check if the trace qualifies for a possible spell.

        Conditions:
        1. It is not currently being drawn
           (5 seconds have passed since the last detected keypoint)
        2. It is made of at least 40 keypoints
        3. Area covered by the trace is sufficiently large
        :return: True/ False. Whether trace is valid
        """
        if self._detected_blob:
            return TraceStatus.TRACING
        """ Continue if there's no blob on screen and there are enough tracepoints. """

        current_keypoint_time = time.time()
        elapsed = current_keypoint_time - self._last_keypoint_time
        if elapsed > BLANK_TIME_GAP:
            # print(f"Number of tracepoints: {len(self._tracepoints)}")
            if len(self._tracepoints) == 0:
                return TraceStatus.NO_TRACE
            if len(self._tracepoints) < REQUIRED_DEQUE_SIZE:
                print(f"Trace length = {len(self._tracepoints)}")
                return TraceStatus.INVALID_TRACE
            print(f"Status = Ready | Deque.len = {len(self._tracepoints)}")
            return TraceStatus.READY_FOR_SPELLCHECK
        else:
            return TraceStatus.WAITING_FOR_TRACE

    def erase_trace(self):
        self._draw_frame = blank_trace_frame()
        self._tracepoints.clear()

    def add_wand_trace(self, frame: np.ndarray) -> np.ndarray:
        # self.trace_wand(frame)
        new_frame: np.ndarray
        try:
            new_frame = cv2.bitwise_or(frame, self._draw_frame)
        except Exception as e:
            print(f"Some exception has occurred.\n Shape of frame: {frame.shape}"
                  f"\n Shape of trace: {self._draw_frame.shape}."
                  f"\n Exception: {e}")
        return new_frame

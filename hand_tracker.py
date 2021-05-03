"""
This module accesses the device's camera and performs
hand-tracking from the camera's video feed.

======================================== ===================== ========================= =====================
Routine Names                            Inputs                Outputs                   Exceptions
======================================== ===================== ========================= =====================
__init__                                 int                                             ValueError
get_image_data                                                 bool, image, Dict, Dict 
__del__
======================================== ===================== ========================= =====================

**Assumptions:**

* __init__ is called before any other access routine.
* get_image_data is only called if a valid video source is found

**State variables**

Attributes:
    video_capture (VideoCapture): A video feed from a given video source.
    width (int): Width of video capture object
    height (int): Height of video capture object
    hands (Hands): A detection model to capture and track hand(s) with given confidence levels
    current_hand_landmarks (Dict): A dictionary that holds the current landmarks of the left and right hands.
    past_hand_landmarks (Dict): A dictionary that holds the past landmarks of the left and right hands.
    success (bool): A boolean that depends on a successful frame capture

"""
from copy import copy
import cv2
import mediapipe as mp

mp_drawing_utils = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
empty_landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
empty_landmarks = { "Right": empty_landmark_list, "Left": empty_landmark_list }

class HandTracker:
    """
    This is a class that captures live video feed using openCV and tracks handmovment using mediapipe.
    This live video feed is turned into frames with an overlay onto the tracked hand(s) and returns
    a singular frame when called.

    Args:
        video_source (int): A reference to which video source should be used (0 is the default webcam of the machine)

    Raises:
        ValueError: | If the video capture is unable to be opened by the given video source, a value error will be raised
            on that given video source|
    """
    def __init__(self, video_source = 0):
        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened(): raise ValueError("Unable to open video source", video_source)

        self.width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9, max_num_hands=2)
        self.current_hand_landmarks = empty_landmarks
        self.past_hand_landmarks = empty_landmarks

    def get_image_data(self):
        """
        Takes a frame from the video feed and tracks and overlays the hand(s) of the user. 

        Returns: 
            tuple: A Tuple containing:
                - success (bool): If frame was captured successfully
                - image (image): The current frame with the hand overlay
                - current_hand_landmarks (Dict): The current left/right landmakrs of each hand
                - past_hand_landmarks (Dict): The past left/right landmakrs of each hand
        """
        if self.video_capture.isOpened():
            success, image = self.video_capture.read()
            if success:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True

                past_hand_landmarks = copy(self.past_hand_landmarks)

                if results.multi_hand_landmarks:
                    multi_hand_landmarks = results.multi_hand_landmarks

                    landmarkList = [
                        landmark
                        for landmark in multi_hand_landmarks
                    ]

                    for landmarks in landmarkList:
                        self.current_hand_landmarks["Right" if is_right_hand(landmarks) else "Left"] = landmarks
                        mp_drawing_utils.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

                    self.past_hand_landmarks = copy(self.current_hand_landmarks)

                else:
                    self.past_hand_landmarks = empty_landmarks

                return success, image, self.current_hand_landmarks, past_hand_landmarks
        return (False, None)

    def __del__(self):
        self.clean_up()

    def clean_up(self):
        """
        A method that is called upon closing the program that closes and releases the video source.
        """
        if (self.video_capture.isOpened()):
            self.hands.close()
            self.video_capture.release()

        cv2.destroyAllWindows()

def is_right_hand(mp_landmarks):
    """
    Determines if given coordinates are from the left or right hand.
    Determined assuming the input image is mirrored,
    i.e., taken with a front-facing/selfie camera with images flipped horizontally.
    Args:
        mp_landmarks (NormalizedLandmarkList): Hand landmark coordinates from mediapipe.
    """
    coords = [
        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
        for landmark in mp_landmarks.landmark
    ]
    return (
        coords[2]["x"] < coords[17]["x"]
    )

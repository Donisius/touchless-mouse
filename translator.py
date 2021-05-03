"""
This module recieves present and past coordinates of hand landmarks piped from mediapipe
and tries it to find the best OS action to perform based on the positions and patterns of the
aforementioned coordinates.

======================================== ===================== ===================== =====================
Routine Names                            Inputs                Outputs               Exceptions
======================================== ===================== ===================== =====================
__init__                                 float, float, int     
translate_landmarks_to_keyboard_action   List, List            
translate_landmarks_to_mouse_action      List, List     
get_3d_distance                          Dict, Dict            float
get_hand_coords                          List                  Dict
is_fist                                  List                  bool
======================================== ===================== ===================== =====================

**Assumptions:**

* __init__ is called before any other access routine.

**State variables**

Attributes:
    mp_landmarks (NormalizedLandmarkList): Hand landmarks piped from mediapipe.
    mp_past_landmarks (NormalizedLandmarkList): Hand landmarks piped from the previous iteration from mediapipe.
    coords (List): Landmarks piped from mediapipe which are mapped to an indexable form to be used in translation.
    past_coords (List): The previous landmarks piped from mediapipe which are mapped to an indexable
                        form to be used in translation.
    hand_coords (Dict): Landmarks commonly used multiple times. The exact ones being palm coordinates
                        and index, middle finger, ring finger, and thumb tip coordinates.
    past_hand_coords (Dict): Landmarks from the previous iteration commonly used multiple times.
                             The exact ones being palm coordinates and index, middle finger, ring finger,
                             and thumb tip coordinates.

**State invariants**

* ``|mp_landmarks.landmark| = 20 ∧ (|mp_past_landmarks.landmark| = 20 ∨ |mp_past_landmarks.landmark| = 0)``
* ``|coords| = 20 ∧ |past_coords| = 20``
* ``∀ (i ε 0..19 | 0 <= coords[i].x ≤ 1 ∧ 0 ≤ coords[i].x ≤ 1 ∧ 0 ≤ coords[i].x ≤ 1)``
* ``∀ (i ε 0..19 | 0 ≤ coords[i].y ≤ 1 ∧ 0 ≤ coords[i].y ≤ 1 ∧ 0 ≤ coords[i].y ≤ 1)``
* ``∀ (i ε 0..19 | 0 ≤ past_coords[i].x ≤ 1 ∧ 0 ≤ past_coords[i].x ≤ 1 ∧ 0 ≤ past_coords[i].x ≤ 1)``
* ``∀ (i ε 0..19 | 0 ≤ past_coords[i].y ≤ 1 ∧ 0 ≤ past_coords[i].y ≤ 1 ∧ 0 ≤ past_coords[i].y ≤ 1)``
* ``0 ≤ hand_coords.x ≤ 1 ∧ 0 ≤ hand_coords.y ≤ 1``
* ``0 ≤ past_hand_coords.x ≤ 1 ∧ 0 ≤ past_hand_coords.y ≤ 1``
"""

import math
import os
import sys

# Third-party imports
import mouse
import keyboard

def get_3d_distance(coord1, coord2):
    """
    Calculates and returns the distance between 2 coordinates in 3 dimensional space.
    
    Args:
        coord1 (Dict): First landmark coordinate to be compared.
        coord2 (Dict): Second landmark coordinate to be compared.

    Returns:
        float: The distance between coord1 and coord2 in 3 dimensional space.
    """
    return math.sqrt(
        (coord1.get("x") - coord2.get("x")) ** 2 +
        (coord1.get("y") - coord2.get("y")) ** 2 +
        (coord1.get("z") - coord2.get("z")) ** 2
    )

def get_hand_coords(coords):
    """
    Creates a hand coordinate object from hand landmark coordinates.

    Args:
        coords (List): List of all hand landmark coordinates.

    Returns:
        Dict: A dictionary containing the coordinate positions of the tips of the thumb, index, middle, ring
              and pinkie finger and the palm.
    """
    return {
        "index": coords[8],
        "middle": coords[12],
        "ring": coords[16],
        "pinkie": coords[20],
        "thumb": coords[4],
        "palm": coords[0]
    }

def is_shaking(palm_coords, past_palm_coords, shake_sensitivity):
    """
    Shake detection, checks if changes in palm position from the last iteration is too small.
    
    Args:
        palm_coords (Dict): Coordinates of palm position.
        past_palm_position (Dict): Coordinates of palm position from the last iteration.
        shake_sensitivity (float): Distance threshold palm must travel in relation to its' previous position in
                                   order for it to be considered a movement.

    Returns:
        bool: `True` if changes in palm position from last iteration is too small, `False` otherwise.
    """
    return (
        abs(palm_coords.get("x") - past_palm_coords.get("x")) < shake_sensitivity
        and abs(palm_coords.get("y") - past_palm_coords.get("y")) < shake_sensitivity
        and abs(palm_coords.get("z") - past_palm_coords.get("z")) < shake_sensitivity
    )

def is_fist(coords):
    """
    Detects if palm is closed.
    
    Args:
        coords (List): List of all hand landmark coordinates.

    Returns:
        bool: `True` if palm is close, `False` otherwise.
    """
    return  (
        not coords[8].get("y") <= coords[6].get("y")
        and not coords[12].get("y") <= coords[10].get("y")
        and not coords[16].get("y") <= coords[14].get("y")
        and not coords[20].get("y") <= coords[18].get("y")
    )

class Translator:
    """
    This is a configurable translator that contains methods which take coordinates from
    mediapipe and translates them to mouse actions and keyboard actions.

    Args:
        distance_threshold (float): Distance between finger coordinates to be considered them touching.
        shake_sensitivity (float): Distance threshold palm must travel in relation to its' previous position in
                                   order for it to be considered a movement.
        mouse_sensitivity (float): Sensitivity of cursor movements in relation to hand movements.
        scroll_sensitivity (float): Sensitivity of scrolling in relation to hand movements.
    """
    def __init__(
        self,
        distance_threshold,
        shake_sensitivity,
        mouse_sensitivity,
        scroll_sensitivity
    ):
        self.distance_threshold = distance_threshold
        self.shake_sensitivity = shake_sensitivity
        self.mouse_sensitivity = mouse_sensitivity
        self.scroll_sensitivity = scroll_sensitivity
        self.enable_pickle = False

    def translate_landmarks_to_keyboard_action(self, mp_landmarks, mp_past_landmarks):
        """
        Takes current and present landmarks piped from mediapipe and attempts to interpret
        the best keyboard action for particular gestures or movements.
        
        The available translations are as follows:

===================  ==================================================
Keyboard action      Gesture
===================  ==================================================
Space                Touch index finger and thumb together and open
Enter                Touch middle finger and thumb together and open
Backspace            Touch ring finger and thumb together and open
Escape               Touch pinkie finger and thumb together and open
Alt + Tab            Make a fist
Navigate Alt + Tab   Move left, right down or up while making a fist
===================  ==================================================

        Args:
            mp_landmarks (NormalizedLandmarkList): Hand landmarks piped from mediapipe.
            mp_past_landmarks (NormalizedLandmarkList): Hand landmarks piped from the previous iteration from mediapipe.
        """
        if len(mp_past_landmarks.landmark) == 0:
            return

        coords = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in mp_landmarks.landmark
        ]

        past_coords = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in mp_past_landmarks.landmark
        ]

        """Present information"""
        hand_coords = get_hand_coords(coords)
        # Past information
        past_hand_coords = get_hand_coords(past_coords)

        # Get the current distances of each of the fingers and the thumb.
        index_thumb_distance = get_3d_distance(hand_coords["index"], hand_coords["thumb"])
        middle_thumb_distance = get_3d_distance(hand_coords["middle"], hand_coords["thumb"])
        ring_thumb_distance = get_3d_distance(hand_coords["ring"], hand_coords["thumb"])
        pinkie_thumb_distance = get_3d_distance(hand_coords["pinkie"], hand_coords["thumb"])

        # Get the past distances of each fingers and thumb.
        past_index_thumb_distance = get_3d_distance(past_hand_coords["index"], past_hand_coords["thumb"])
        past_middle_thumb_distance = get_3d_distance(past_hand_coords["middle"], past_hand_coords["thumb"])
        past_ring_thumb_distance = get_3d_distance(past_hand_coords["ring"], past_hand_coords["thumb"])
        past_pinkie_thumb_distance = get_3d_distance(past_hand_coords["pinkie"], past_hand_coords["thumb"])

        # When past finger distance is past threshold and current finger distance is not, execute action
        # Ensures actions is only executed once
        if is_fist(coords):
            if not is_fist(past_coords):
                keyboard.press("alt+tab")
            # Navigate windows while alt-tabbed
            if (not is_shaking(hand_coords["palm"], past_hand_coords["palm"], self.shake_sensitivity + 0.008)):
                if (hand_coords["palm"].get("x") <= past_hand_coords["palm"].get("x")):
                    keyboard.send("shift+tab")
                    return
                if (hand_coords["palm"].get("x") > past_hand_coords["palm"].get("x")):
                    keyboard.send("tab")
                    return
                if (hand_coords["palm"].get("y") <= past_hand_coords["palm"].get("y")):
                    keyboard.send("down arrow")
                    return
                if (hand_coords["palm"].get("y") > past_hand_coords["palm"].get("y")):
                    keyboard.send("up arrow")
                    return

        if (
            not index_thumb_distance <= self.distance_threshold
            and past_index_thumb_distance <= self.distance_threshold
        ):
            keyboard.send("space")
            return

        if (
            not middle_thumb_distance <= self.distance_threshold
            and past_middle_thumb_distance <= self.distance_threshold
        ):
            keyboard.send("enter")
            return

        if (
            not ring_thumb_distance <= self.distance_threshold
            and past_ring_thumb_distance <= self.distance_threshold
        ):
            keyboard.send("backspace")
            return

        if (
            not pinkie_thumb_distance <= self.distance_threshold
            and past_pinkie_thumb_distance <= self.distance_threshold
        ):
            keyboard.send("escape")
            return

        elif not is_fist(coords) and is_fist(past_coords):
            keyboard.release("alt+tab")
            return

    def translate_landmarks_to_mouse_action(self, mp_landmarks, mp_past_landmarks):
        """
        Takes current and present landmarks piped from mediapipe and attempts to interpret
        the best mouse action for particular gestures or movements.

                The available translations are as follows:

======================    ==========================================================
Mouse action              Gesture
======================    ==========================================================
Left mouse click          Touch index finger and thumb together
Right mouse click         Touch ring finger and thumb together
Scroll                    Hold middle finger and thumb together and move up and down
Move cursor               Move hand with an open palm
Lift mouse (no action)    Make a fist
======================    ==========================================================

        Args:
            mp_landmarks (NormalizedLandmarkList): Hand landmarks piped from mediapipe.
            mp_past_landmarks (NormalizedLandmarkList): Hand landmarks piped from the previous iteration from mediapipe.
        """
        if not mp_past_landmarks or len(mp_past_landmarks.landmark) == 0:
            return

        cursor_x_position, cursor_y_position = mouse.get_position()

        # Landmarks piped from mediapipe need to be mapped to an indexable form.
        coords = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in mp_landmarks.landmark
        ]

        past_coords = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in mp_past_landmarks.landmark
        ]

        # Present information
        hand_coords = get_hand_coords(coords)
        # Past information
        past_palm_coords = past_coords[0]

        # Get the current distances of each of the fingers and the thumb.
        index_thumb_distance = get_3d_distance(hand_coords["index"], hand_coords["thumb"])
        middle_thumb_distance = get_3d_distance(hand_coords["middle"], hand_coords["thumb"])
        ring_thumb_distance = get_3d_distance(hand_coords["ring"], hand_coords["thumb"])

        # if distance between coords is too small, exit
        if is_shaking(hand_coords["palm"], past_palm_coords, self.shake_sensitivity):
            return

        # Deactivate control of the mouse. This would be like lifting (check close palm)
        if is_fist(coords):
            return

        # Check if we should left mouse click and do it if we should. (index and thumb touching)
        if index_thumb_distance <= self.distance_threshold:
            mouse.click()
            return

        # Check if we should right mouse click and do it if we should. (ring and thumb touching)
        if ring_thumb_distance <= self.distance_threshold:
            mouse.right_click()
            return

        # Check if we should be scrolling and do it if we should. (check middle and thumb touching)
        if middle_thumb_distance <= self.distance_threshold:
            clicks_to_scroll = int(self.scroll_sensitivity * (hand_coords["palm"].get("y") - past_palm_coords.get("y")))
            mouse.wheel(-clicks_to_scroll)
            # Keep the cursor in place while scrolling.
            return

        # Calculate where the cursor should be positioned and move it there.
        new_x_position = (
            cursor_x_position +
            self.mouse_sensitivity * (hand_coords["palm"].get("x") - past_palm_coords.get("x"))
        )

        new_y_position = (
            cursor_y_position +
            self.mouse_sensitivity * (hand_coords["palm"].get("y") - past_palm_coords.get("y"))
        )

        mouse.move(new_x_position, new_y_position)

    def get_mouse_sensitivity(self):
        """
        Returns:
            float: current mouse sensitivity.
        """
        return self.mouse_sensitivity

    def set_mouse_sensitivity(self, mouse_sensitivity):
        """
        Args:
            mouse_sensitivity (float): Sensitivity of cursor movements in relation to hand movements.
        """
        self.mouse_sensitivity = mouse_sensitivity

    def get_distance_threshold(self):
        """
        Returns:
            float: current distance threshold.
        """
        return self.distance_threshold

    def set_distance_threshold(self, distance_threshold):
        """
        Args:
            distance_threshold (float): Distance between finger coordinates to be considered them touching.
        """
        self.distance_threshold = distance_threshold

    def get_shake_sensitivity(self):
        """
        Returns:
            float: current shake sensitivity.
        """
        return self.shake_sensitivity

    def set_shake_sensitivity(self, shake_sensitivity):
        """
        Args:
            shake_sensitivity (float): | Distance threshold palm must travel in relation to its' previous position in
            | order for it to be considered a movement.
        """
        self.shake_sensitivity = shake_sensitivity

    def get_scroll_sensitivity(self):
        """
        Returns:
            float: current scroll sensitivity.
        """
        return self.scroll_sensitivity

    def set_scroll_sensitivity(self, scroll_sensitivity):
        """
        Args:
            scroll_sensitivity (float): Sensitivity of scrolling in relation to hand movements.
        """
        self.scroll_sensitivity = scroll_sensitivity

    def __del__(self):
        keyboard.release("alt+tab")

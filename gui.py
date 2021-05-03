"""
This module creates a GUI for toucless mouse to operate. It initilizes the hand tracker and
displays the output and video feed on screen.
"""

import tkinter
import PIL.Image
import PIL.ImageTk
from hand_tracker import HandTracker, empty_landmark_list
from translator import Translator

class GUI:
    """Takes in title for the gui window, window initializer
    and video source (0 is default webcam), and intializes gui"""
    def __init__(self, window_title, window = tkinter.Tk(), video_source = 0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.hand_tracker = HandTracker(self.video_source)
        self.translator = Translator(0.06, 0.002, 3000, 200)
        self.canvas = tkinter.Canvas(window, width = self.hand_tracker.width, height = self.hand_tracker.height)
        self.canvas.pack()
        self.delay = 15
        self.update()
        self.window.mainloop()

    def update(self):
        """Updates video frame in gui"""
        success, frame, hand_landmarks, past_hand_landmarks = self.hand_tracker.get_image_data()
        if success:
            self.image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)
            if (
                    hand_landmarks["Left"] is not empty_landmark_list
                    and past_hand_landmarks["Left"] is not empty_landmark_list
            ):
                self.translator.translate_landmarks_to_mouse_action(
                    hand_landmarks["Left"],
                    past_hand_landmarks["Left"]
                )

            if (
                    hand_landmarks["Right"] is not empty_landmark_list
                    and past_hand_landmarks["Right"] is not empty_landmark_list
                ):
                self.translator.translate_landmarks_to_keyboard_action(
                    hand_landmarks["Right"],
                    past_hand_landmarks["Right"]
                )

        self.window.after(self.delay, self.update)

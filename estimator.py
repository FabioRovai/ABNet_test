from typing import List, Tuple
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import json

from body import *   
from hand import *

class PoseEstimator:
    def __init__(self, body_model_path: str, hand_model_path: str) -> None:
        """
        Initialize the PoseEstimator.

        Parameters:
            body_model_path (str): Path to the pre-trained body pose model file.
            hand_model_path (str): Path to the pre-trained hand pose model file.
        """
        self.body_estimation = Body(body_model_path)
        self.hand_estimation = Hand(hand_model_path)

    def estimate_pose(self, image_path: str) -> np.ndarray:
        """
        Estimate the pose in an image.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Image with the estimated pose drawn on it.
        """
        # Load the input image
        oriImg = cv2.imread(image_path)

        # Estimate body pose
        candidate, subset = self.body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = draw_bodypose(canvas, candidate, subset)

        # Detect hands and estimate hand pose
        hands_list = handDetect(candidate, subset, oriImg)
        all_hand_peaks = []

        for x, y, w, is_left in hands_list:
            peaks = self.hand_estimation(oriImg[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)

        # Draw hand pose on the canvas
        canvas = draw_handpose(canvas, all_hand_peaks)
        return canvas
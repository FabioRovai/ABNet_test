from typing import List, Tuple
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
from src import model
from src import util
from src.body import Body
from src.hand import Hand

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
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Detect hands and estimate hand pose
        hands_list = util.handDetect(candidate, subset, oriImg)
        all_hand_peaks = []

        for x, y, w, is_left in hands_list:
            peaks = self.hand_estimation(oriImg[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)

        # Draw hand pose on the canvas
        canvas = util.draw_handpose(canvas, all_hand_peaks)
        return canvas

# Load parameters from config.json
with open('/content/params.JSON', 'r') as config_file:
    config = json.load(config_file)

# Initialize the pose estimator
pose_estimator = PoseEstimator(config["body_model_path"], config["hand_model_path"])

# Estimate and plot the pose
estimated_pose = pose_estimator.estimate_pose(config["test_image_path"])

# Plot the original image and the estimated pose side by side
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(plt.imread(config["test_image_path"]))
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(estimated_pose[:, :, [2, 1, 0]])
plt.axis('off')
plt.title('Estimated Pose')

plt.tight_layout()
plt.show()

from body import *   
from hand import *
from estimator import *
from model import *
from util import *


def plot_estimated_pose(body_model_path, hand_model_path, test_image_path):
    # Initialize the pose estimator
    pose_estimator = PoseEstimator(body_model_path, hand_model_path)

    # Estimate the pose
    estimated_pose = pose_estimator.estimate_pose(test_image_path)

    # Plot the original image and the estimated pose side by side
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(test_image_path))
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(estimated_pose[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.title('Estimated Pose')
    
    plt.tight_layout()
    plt.show()
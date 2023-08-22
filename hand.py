import cv2
import json
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage.measure import label
import concurrent.futures

from model import*
from util import *

def process_scale(scale: float, oriImg: np.ndarray, model: torch.nn.Module, stride: int, padValue: int, thre: float) -> np.ndarray:
    """
    Process a single scale for hand pose estimation.

    Parameters:
        scale (float): The scale factor.
        oriImg (np.ndarray): The input image.
        model (torch.nn.Module): The pre-trained hand pose estimation model.
        stride (int): The stride value for processing.
        padValue (int): Padding value.
        thre (float): Threshold value.

    Returns:
        np.ndarray: Heatmap for the processed scale.
    """
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
    im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
    im = np.ascontiguousarray(im)

    data = torch.from_numpy(im).float()
    if torch.cuda.is_available():
        data = data.cuda()

    with torch.no_grad():
        output = model(data).cpu().numpy()

    heatmap = np.transpose(np.squeeze(output), (1, 2, 0))
    heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    return heatmap

class Hand:
    def __init__(self, model_path: str):
        """
        Initialize the Hand instance.

        Parameters:
            model_path (str): Path to the pre-trained model file.
        """
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg: np.ndarray) -> np.ndarray:
        """
        Perform hand pose estimation on the input image.

        Parameters:
            oriImg (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: An array of detected hand keypoints.
        """
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

        all_peaks = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for scale in multiplier:
                futures.append(executor.submit(process_scale, scale, oriImg, self.model, stride, padValue, thre))

            heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))
            for future, scale in zip(futures, multiplier):
                heatmap = future.result()
                heatmap_avg += heatmap / len(multiplier)

            for part in range(21):
                map_ori = heatmap_avg[:, :, part]
                one_heatmap = gaussian_filter(map_ori, sigma=3)
                binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
                if np.sum(binary) == 0:
                    all_peaks.append([0, 0])
                    continue
                label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
                max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
                label_img[label_img != max_index] = 0
                map_ori[label_img == 0] = 0

                y, x = np.unravel_index(map_ori.argmax(), map_ori.shape)
                all_peaks.append([x, y])

        return np.array(all_peaks)

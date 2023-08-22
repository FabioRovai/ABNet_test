import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2


import torch


def padRightDownCorner(img: np.ndarray, stride: int, pad_value: float) -> tuple[np.ndarray, list[int]]:
    """
    Pad an image to a multiple of a given stride.

    Args:
        img (np.ndarray): Input image.
        stride (int): Desired stride.
        pad_value (float): Value to fill padding.

    Returns:
        tuple[np.ndarray, list[int]]: Padded image and padding values [top, left, bottom, right].
    """
    h, w = img.shape[0], img.shape[1]

    pad = [0] * 4
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    pad_up = torch.FloatTensor([pad_value]).unsqueeze(1).unsqueeze(2).expand(pad[0], w, 3)
    img_padded = torch.cat((pad_up, torch.from_numpy(img)), dim=0)
    pad_left = torch.FloatTensor([pad_value]).unsqueeze(0).unsqueeze(2).expand(h + pad[0], pad[1], 3)
    img_padded = torch.cat((pad_left, img_padded), dim=1)
    pad_down = torch.FloatTensor([pad_value]).unsqueeze(1).unsqueeze(2).expand(pad[2], w + pad[1], 3)
    img_padded = torch.cat((img_padded, pad_down), dim=0)
    pad_right = torch.FloatTensor([pad_value]).unsqueeze(0).unsqueeze(2).expand(h + pad[2], pad[3], 3)
    img_padded = torch.cat((img_padded, pad_right), dim=1)

    return img_padded, pad

def transfer(model, model_weights):
    """
    Transfer weights from a dictionary to a PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model.
        model_weights (dict): Dictionary containing weights to transfer.

    Returns:
        dict: Transfered model weights.
    """
    transferred_model_weights = {}
    for weights_name in model.state_dict().keys():
        transferred_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transferred_model_weights



def draw_bodypose(canvas: np.ndarray, candidate: np.ndarray, subset: np.ndarray) -> np.ndarray:
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11],
               [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def draw_handpose(canvas: np.ndarray, all_hand_peaks: list, show_number: bool = False) -> np.ndarray:
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    fig = Figure(figsize=plt.figaspect(canvas))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]))

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            ax.plot(x, y, 'r.')
            if show_number:
                ax.text(x, y, str(i))

    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas

def draw_handpose_by_opencv(canvas: np.ndarray, peaks: np.ndarray, show_number: bool = False) -> np.ndarray:
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, e in enumerate(edges):
        if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for i, keypoint in enumerate(peaks):
        x, y = keypoint
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)

    return canvas

def handDetect(candidate: np.ndarray, subset: np.ndarray, oriImg: np.ndarray) -> list:
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]

    for person in subset.astype(int):
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue

        hands = []

        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])

        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            x -= width / 2
            y -= width / 2
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    return detect_result

# get max index of 2d array
def torch_max(array):
    arrayindex = torch.argmax(array, dim=1)
    arrayvalue = torch.max(array, dim=1).values
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
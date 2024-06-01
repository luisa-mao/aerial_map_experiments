import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def increase_contrast(costmap, mask):
    # get the min and max of the costmap within the mask
    min_val = np.min(costmap[mask > 0])
    max_val = np.max(costmap[mask > 0])

    # normalize the costmap
    costmap = (costmap - min_val) / (max_val - min_val) 
    return costmap   

def histogram_mode_separation(costmap, mask, param=0.4):
    mask_uint8 = (mask*255).astype(np.uint8)
    hist = cv2.calcHist([costmap*255], [0], mask_uint8, [256], [0,256])
    hist_norm = hist.ravel()/hist.max()

    # Find peaks using scipy's function
    peaks, _ = find_peaks(hist_norm, distance=20, prominence=0.05)
    # Invert the histogram
    hist_inv = 1.0 - hist_norm
    # Find peaks in the inverted histogram, which correspond to troughs in the original histogram
    troughs, _ = find_peaks(hist_inv, distance=20, prominence=0.05)

    # plt.plot(hist) 
    # plt.plot(peaks, hist_norm[peaks], "x")
    # plt.plot(troughs, hist_norm[troughs], "o")
    # plt.savefig("hist.png")

    print (len(peaks))

    if len(peaks) < 2:
        return costmap

    # print("here")
    # imwrite the before
    # cv2.imwrite("before_contrast.png", costmap * 255)
    new_peaks = []
    new_troughs = []

    a = len(peaks)-1
    for i in range(len(peaks)):
        new_peaks.append(i/a)
    for i in range(len(troughs)):
        new_troughs.append(i/a + 1.0/(2*a))
    new_troughs.append(1)

    peaks = peaks/255.0
    troughs = troughs/255.0
    # append 1 to the end of the troughs
    troughs = np.append(troughs, 1)

    # hack
    # if peaks and troughs are not of the same length, return
    if len(peaks) != len(troughs):
        return costmap
   

    # print the peaks and troughs
    # print(troughs)
    # print(peaks)
    # print(new_peaks)
    # print(new_troughs)


    # iterate over costmap
    for i in range(costmap.shape[0]):
        for j in range(costmap.shape[1]):
            if mask[i, j] > 0:
                c = costmap[i, j]
                # the peak c is in
                peak = 0
                for k in range(len(troughs)):
                    if c <= troughs[k]:
                        peak = k
                        break
                value = c + (new_peaks[peak]-peaks[peak])* param
                clamped_value = max(0, min(value, 1))
                # clamped_value = new_peaks[peak]
                costmap[i,j] = clamped_value
    # imwrite the costmap
    # print("written", cv2.imwrite("after_contrast.png", costmap * 255))
    
    # exit()
    return costmap


def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def draw_grid(image, cell_size = 64):
    for i in range(0, image.shape[1], cell_size):
        cv2.line(image, (i, 0), (i, image.shape[0]), (0, 255, 0), 1)  # Draw vertical lines
    for j in range(0, image.shape[0], cell_size):
        cv2.line(image, (0, j), (image.shape[1], j), (0, 255, 0), 1)  # Draw horizontal lines

    # Label the cells
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    font_thickness = 1
    for i in range(0, image.shape[1], cell_size):
        for j in range(0, image.shape[0], cell_size):
            cell_label = f"({i}, {j})"
            text_size, _ = cv2.getTextSize(cell_label, font, font_scale, font_thickness)
            text_x = i + (cell_size - text_size[0]) // 2
            text_y = j + (cell_size + text_size[1]) // 2
            cv2.putText(image, cell_label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)



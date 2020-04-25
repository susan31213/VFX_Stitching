import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.feature import corner_harris, corner_peaks

def feature_detection(gray, k, r_thre):
    # gradient
    x_kernal = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    y_kernal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = scipy.signal.convolve2d(gray, x_kernal, mode='same')
    Iy = scipy.signal.convolve2d(gray, y_kernal, mode='same')

    # Ixx Iyy Ixy
    Ixx = scipy.ndimage.gaussian_filter(Ix**2, sigma=1)
    Ixy = scipy.ndimage.gaussian_filter(Iy*Ix, sigma=1)
    Iyy = scipy.ndimage.gaussian_filter(Iy**2, sigma=1)

    # Harris response calculation
    detA = Ixx * Iyy - Ixy ** 2     # determinant
    traceA = Ixx + Iyy              # trace
    harris_response = detA - k * traceA ** 2

    # check R value
    r_arr = np.zeros(np.shape(gray))
    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            if r > r_thre:
                # this is a corner
                r_arr[rowindex, colindex] = r

    # Local maximum
    corners = peak_local_max(r_arr, min_distance=9)

    # feature
    f = []
    for c in corners:
        neighbors = []
        for u in range(-2, 3, 1):
            for v in range(-2, 3, 1):
                neighbors.append(gray[c[0]+u][c[1]+v])
        f.append(neighbors)

    return corners, f


def feature_matching(gray1, gray2):
    return

imgs = [imread('parrington/prtn02.jpg'), imread('parrington/prtn01.jpg')]
corner_vis = [np.copy(img) for img in imgs]

corner_vec = []
feature_vec = []

for img_idx in range(0, 17):
    img1 = imread("parrington/prtn{:02d}.jpg".format(img_idx))
    img2 = imread("parrington/prtn{:02d}.jpg".format(img_idx+1))
    for_showing_corner1 = np.copy(img1)
    for_showing_corner2 = np.copy(img2)

    gray1 = rgb2gray(img1)
    gray2 = rgb2gray(img2)
    c1, f1 = feature_detection(gray1, 0.04, 0.12)
    c2, f2 = feature_detection(gray2, 0.04, 0.12)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    ax[0].imshow(for_showing_corner1, interpolation='nearest', cmap=plt.cm.gray) 
    ax[1].imshow(for_showing_corner2, interpolation='nearest', cmap=plt.cm.gray)

    # matching
    match = []
    for f in f1:
        min_dis = 1000000
        min_index = -1
        threshold = 2
        for k in range(len(f2)):
            dis = np.sum(np.absolute(np.asarray(f)-np.asarray(f2[k])))
            if min_dis > dis and dis < threshold:
                min_dis = dis
                min_index = k
        match.append(min_index)

    # plot matching point
    for k in range(len(f1)):
        if(match[k] != -1):
            ax[0].plot(c1[k][1], c1[k][0], '.r', markersize=3)
            ax[0].text(c1[k][1] + 5, c1[k][0] - 5, str(k))
            ax[1].plot(c2[match[k]][1], c2[match[k]][0], '.r', markersize=3)
            ax[1].text(c2[match[k]][1] + 5, c2[match[k]][0] - 5, str(k))
    plt.show()

# for i in range(len(imgs)):
#     gray = rgb2gray(imgs[i])
#     corners, features = feature_detection(gray, 0.04, 0.12)
#     corner_vec.append(corners)
#     feature_vec.append(features)
#     ax[i].imshow(corner_vis[i], interpolation='nearest', cmap=plt.cm.gray)    

# feature_vec = np.asarray(feature_vec)
# for i in range(len(imgs)-1):
#     match = []
#     for f1 in feature_vec[i]:
#         min_dis = 1000000
#         min_index = -1
#         threshold = 2
#         # print(f1)
#         for k in range(len(feature_vec[i+1])):
#             dis = np.sum(np.absolute(np.asarray(f1)-feature_vec[i+1][k]))
#             if min_dis > dis and dis < threshold:
#                 min_dis = dis
#                 min_index = k
#         match.append(min_index)

# for k in range(len(feature_vec[0])):
#     if(match[k] != -1):
#         ax[0].plot(corner_vec[0][k][1], corner_vec[0][k][0], '.r', markersize=3)
#         ax[0].text(corner_vec[0][k][1] + 10, corner_vec[0][k][0] + 10, str(k))
#         ax[1].plot(corner_vec[1][match[k]][1], corner_vec[1][match[k]][0], '.r', markersize=3)
#         ax[1].text(corner_vec[1][match[k]][1] + 10, corner_vec[1][match[k]][0] + 10, str(k))
# plt.show()
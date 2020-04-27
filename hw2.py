import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def feature_detection(gray, k, r_thre):
    # gradient
    x_kernal = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    y_kernal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = cv2.filter2D(gray, -1, x_kernal)
    Iy = cv2.filter2D(gray, -1, y_kernal)
    print(Ix)
    # Ixx Iyy Ixy
    Ixx = cv2.GaussianBlur(Ix**2, ksize=(3,3), sigmaX=1)
    Ixy = cv2.GaussianBlur(Iy*Ix, ksize=(3,3), sigmaX=1)
    Iyy = cv2.GaussianBlur(Iy**2, ksize=(3,3), sigmaX=1)

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


def feature_matching(corners1, features1, corners2, features2, threshold, plot=False):
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
        for_showing_corner1 = np.copy(img1)
        for_showing_corner2 = np.copy(img2)
        ax[0].imshow(for_showing_corner1, interpolation='nearest', cmap=plt.cm.gray) 
        ax[1].imshow(for_showing_corner2, interpolation='nearest', cmap=plt.cm.gray)

    # matching
    match = {}
    for idx in range(len(features2)):
        match[idx] = [-1, 1000000]

    for i, f in enumerate(features1):
        min_dis = 1000000
        min_index = -1
        threshold = 2
        for k in range(len(features2)):
            dis = np.sum(np.absolute(np.asarray(f)-np.asarray(features2[k])))
            if min_dis > dis and dis < threshold:
                min_dis = dis
                min_index = k
        # check min_index in features2 is not matched
        if min_index != -1:
            if match[min_index][0] == -1:
                match[min_index] = [i, min_dis]
            elif match[min_index][1] > min_dis:
                match[min_index] = [i, min_dis]

    # plot matching point
    matching_cnt = 0
    print('img1\t->\timg2\tdistance')
    for m in match.keys():
        if(match[m][0] != -1):
            matching_cnt += 1
            print('{:d}\t->\t{:d}\t{:f}'.format(match[m][0], m, match[m][1]))
            if plot:
                ax[0].plot(corners1[match[m][0]][1], corners1[match[m][0]][0], '.r', markersize=3)
                ax[0].text(corners1[match[m][0]][1] + 5, corners1[match[m][0]][0] - 5, str(m))
                ax[1].plot(corners2[m][1], corners2[m][0], '.r', markersize=3)
                ax[1].text(corners2[m][1] + 5, corners2[m][0] - 5, str(m))
    print('features1: {:d}, features2: {:d}'.format(len(features1), len(features2)))
    print('total match: {:d}\n----------------------'.format(matching_cnt))
    if plot:
        plt.show()
    return

imgs = [cv2.imread('parrington/prtn02.jpg'), cv2.imread('parrington/prtn01.jpg')]
corner_vis = [np.copy(img) for img in imgs]

corner_vec = []
feature_vec = []

# for img_idx in range(0, 17):
for img_idx in range(0, 1):
    img1 = cv2.imread("parrington/prtn{:02d}.jpg".format(img_idx))
    img2 = cv2.imread("parrington/prtn{:02d}.jpg".format(img_idx+1))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
    c1, f1 = feature_detection(gray1, 0.04, 0.12)
    c2, f2 = feature_detection(gray2, 0.04, 0.12)

    feature_matching(c1, f1, c2, f2, 2, True)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from random import seed
from random import sample
import math

def feature_detection(gray, k, r_thre, right):
    # gradient
    x_kernal = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    y_kernal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = cv2.filter2D(gray, -1, x_kernal)
    Iy = cv2.filter2D(gray, -1, y_kernal)

    # Ixx Iyy Ixy
    Ixx = cv2.GaussianBlur(Ix**2, ksize=(3,3), sigmaX=1)
    Ixy = cv2.GaussianBlur(Iy*Ix, ksize=(3,3), sigmaX=1)
    Iyy = cv2.GaussianBlur(Iy**2, ksize=(3,3), sigmaX=1)

    # Harris response calculation
    detA = Ixx * Iyy - Ixy ** 2     # determinant
    traceA = Ixx + Iyy              # trace
    harris_responses = detA - k * traceA ** 2

    # check R value
    r_arr = np.zeros(np.shape(gray))
    for r_idx, r in enumerate(harris_responses):
        for c_idx, r in enumerate(r):
            if r > r_thre and ((right and c_idx <= gray.shape[1]*0.4) or ((not right) and c_idx >= gray.shape[1]*0.6)):
                # this is a corner
                r_arr[r_idx, c_idx] = r
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
        i1 = np.copy(img1)
        i2 = np.copy(img2)
        i1[..., 0] = img1[..., 2]
        i1[..., 1] = img1[..., 1]
        i1[..., 2] = img1[..., 0]
        i2[..., 0] = img2[..., 2]
        i2[..., 1] = img2[..., 1]
        i2[..., 2] = img2[..., 0]
        ax[0].imshow(i1, interpolation='nearest', cmap=plt.cm.gray) 
        ax[1].imshow(i2, interpolation='nearest', cmap=plt.cm.gray)

    # matching
    match = {}

    for i, f in enumerate(features1):
        min_dis = 1000000
        min_index = -1
        for k in range(len(features2)):
            dis = np.sum(np.absolute(np.asarray(f)-np.asarray(features2[k])))
            if min_dis > dis and dis < threshold:
                min_dis = dis
                min_index = k
        # check min_index in features2 is not matched
        if min_index != -1:
            if min_index not in match:
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
    return match


def cylindrical_warping(img, f, corners):
    proj = np.zeros((img.shape), dtype=np.uint8)
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            x = int(f*((i-np.shape(img)[0]/2)/math.sqrt((j-np.shape(img)[1]/2)**2+f**2)))
            x = np.shape(img)[0]-1 if x >= np.shape(img)[0] else x
            y = int(f*math.atan2((j-np.shape(img)[1]/2), f))
            y = np.shape(img)[1]-1 if y >= np.shape(img)[1] else y

            if x > np.shape(img)[0]/2 and y > np.shape(img)[1]/2:
                x -= int(np.shape(img)[0]/2)
                y -= int(np.shape(img)[1]/2)
            elif x < np.shape(img)[0]/2 and y < np.shape(img)[1]/2:
                x += int(np.shape(img)[0]/2)
                y += int(np.shape(img)[1]/2)
            elif x < np.shape(img)[0]/2 and y > np.shape(img)[1]/2:
                x += int(np.shape(img)[0]/2)
                y -= int(np.shape(img)[1]/2)
            elif x > np.shape(img)[0]/2 and y < np.shape(img)[1]/2:
                x -= int(np.shape(img)[0]/2)
                y += int(np.shape(img)[1]/2)
            
            proj[x][y] = img[i][j]

    new_corners = np.array(corners)
    for i, c in enumerate(corners):
        x = int(f*((c[0]-np.shape(img)[0]/2)/math.sqrt((c[1]-np.shape(img)[1]/2)**2+f**2)))
        x = np.shape(img)[0]-1 if x >= np.shape(img)[0] else x
        y = int(f*math.atan2((c[1]-np.shape(img)[1]/2), f))
        y = np.shape(img)[1]-1 if y >= np.shape(img)[1] else y

        if x > np.shape(img)[0]/2 and y > np.shape(img)[1]/2:
            x -= int(np.shape(img)[0]/2)
            y -= int(np.shape(img)[1]/2)
        elif x < np.shape(img)[0]/2 and y < np.shape(img)[1]/2:
            x += int(np.shape(img)[0]/2)
            y += int(np.shape(img)[1]/2)
        elif x < np.shape(img)[0]/2 and y > np.shape(img)[1]/2:
            x += int(np.shape(img)[0]/2)
            y -= int(np.shape(img)[1]/2)
        elif x > np.shape(img)[0]/2 and y < np.shape(img)[1]/2:
            x -= int(np.shape(img)[0]/2)
            y += int(np.shape(img)[1]/2)
        new_corners[i] = [x,y]
    return proj, new_corners


def find_translation(c1, c2, match_dict):
    # RANSAC: find how to translate proj1 to proj2 (move right)
    best_m1 = 0
    best_m2 = 0
    best_inliner = 0
    err_thre = 1.5
    n = 1
    K = math.log10(0.001)/math.log10(1-math.pow(0.4,n))
    print('RANSAC ' + str(int(K)) + ' times')
    for k in range(int(K)):
        # print(int(len(c1)*0.3))
        subset = sample(match_dict.keys(), n)
        m1 = 0
        m2 = 0
        for s in subset:
            m1 += c2[s][0] - c1[match_dict[s][0]][0]
            m2 += c2[s][1] - c1[match_dict[s][0]][1]
        m1 /= float(n)
        m2 /= float(n)
        inliner = 0
        for s in match_dict.keys():
            if s in subset:
                continue
            # print(subset, s, abs(c2[s][0] - c1[match_dict[s][0]][0] - m1) + abs(c2[s][1] - c1[match_dict[s][0]][1] - m2))
            if ((c2[s][0] - c1[match_dict[s][0]][0] - m1)**2 + (c2[s][1] - c1[match_dict[s][0]][1] - m2)**2)**0.5 < err_thre:
                inliner += 1
            
        if inliner > best_inliner:
            best_inliner = inliner
            best_m1 = m1
            best_m2 = m2
    print(best_m1, best_m2, best_inliner)
    return best_m1, best_m2, best_inliner


# Get focal lengthes
focal_length = []
file = open('parrington/pano.txt', 'r')
lines = file.readlines()
for i, l in enumerate(lines):
    if i != 0 and i % 13 == 11:
        focal_length.append(float(l))

start = 0
end = 1
# for img_idx in range(0, 17):
for img_idx in range(start, end):
    img1 = cv2.imread("parrington/prtn{:02d}.jpg".format(img_idx))
    img2 = cv2.imread("parrington/prtn{:02d}.jpg".format(img_idx+1))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
    c1, f1 = feature_detection(gray1, 0.06, 0.01, right=True)
    c2, f2 = feature_detection(gray2, 0.06, 0.01, right=False)

    match_dict = feature_matching(c1, f1, c2, f2, 1.5, False)  # key: f2 index, value: [f1 index, error]



# Cylindrical warping
for img_idx in range(start, end):
    img1 = cv2.imread("parrington/prtn{:02d}.jpg".format(img_idx))
    proj1, c1_proj = cylindrical_warping(img1, focal_length[img_idx], c1)

    img2 = cv2.imread("parrington/prtn{:02d}.jpg".format(img_idx+1))
    proj2, c2_proj = cylindrical_warping(img2, focal_length[img_idx], c2)


i1 = np.copy(proj1)
i2 = np.copy(proj2)
i1[..., 0] = proj1[..., 2]
i1[..., 1] = proj1[..., 1]
i1[..., 2] = proj1[..., 0]
i2[..., 0] = proj2[..., 2]
i2[..., 1] = proj2[..., 1]
i2[..., 2] = proj2[..., 0]
for m in match_dict.keys():
    c1_proj[match_dict[m][0]][1] += i2.shape[1]
    # plt.plot(c1_proj[match_dict[m][0]][1], c1_proj[match_dict[m][0]][0], '.r', markersize=3)
    # plt.plot(c2_proj[m][1], c2_proj[m][0], '.r', markersize=3)


x_offsetf, y_offsetf, _ = find_translation(c1_proj, c2_proj, match_dict)
print(x_offsetf,y_offsetf)
x_offset = int(x_offsetf)
y_offset = int(y_offsetf)
concate = np.zeros((i2.shape[0]+abs(x_offset), i2.shape[1]*2+y_offset, 3), dtype=np.uint8)


if x_offset > 0:
    for m in match_dict.keys():
        plt.plot(c1_proj[match_dict[m][0]][1]+y_offset, c1_proj[match_dict[m][0]][0]+x_offset, '.b', markersize=6)
        plt.plot(c2_proj[m][1], c2_proj[m][0], '.r', markersize=3)
    # plt.imshow(np.hstack([i2[x_offset:,:y_offset], i1[:-x_offset,:]]), interpolation='nearest', cmap=plt.cm.gray)
    concate[x_offset:, 0:i2.shape[1]+y_offset, :] = i2[:,:y_offset,:]
    concate[:-x_offset, i2.shape[1]:, :] = i1[:,-y_offset:,:]
    # blend
    for x in range(0,concate.shape[0]):
        for y in range(i2.shape[1]+y_offset, i2.shape[1]):
            if x-x_offset < 0:
                concate[x,y] = i1[x, y-i2.shape[1]-y_offset]
            elif x >= i1.shape[0]:
                concate[x,y] = i2[x-x_offset, y]
            else:
                if i2[x-x_offset, y,0] == 0 and i2[x-x_offset, y,1] == 0 and i2[x-x_offset, y,2] == 0:
                    concate[x,y] = i1[x, y-(i2.shape[1]+y_offset)]
                elif i1[x, y+y_offset,0] == 0 and i1[x, y+y_offset,1] == 0 and i1[x, y+y_offset,2] == 0:
                    concate[x,y] = i2[x-x_offset, y]
                else:
    
                    w = 1-(y-(i2.shape[1]+float(y_offset)))/abs(y_offset)
                    i2_weight = w #if (w > 0.4 and w < 0.5) else (1 if w<=0.4 else 0)
                    concate[x,y] = i2[x-x_offset, y] * (i2_weight) + i1[x, y-(i2.shape[1]+y_offset)] * (1-i2_weight)

else:
    for m in match_dict.keys():
        plt.plot(c1_proj[match_dict[m][0]][1]+y_offset, c1_proj[match_dict[m][0]][0]-x_offset, '.b', markersize=4)
        plt.plot(c2_proj[m][1], c2_proj[m][0], '.r', markersize=3)
    concate[:i2.shape[0], 0:i2.shape[1]+y_offset, :] = i2[:,:y_offset,:]
    concate[-x_offset:, i2.shape[1]:, :] = i1[:,-y_offset:,:]
    # blend
    for x in range(0,concate.shape[0]):
        for y in range(i2.shape[1]+y_offset, i2.shape[1]):
            # print(x,y)
            if x+x_offset < 0:
                concate[x,y] = i2[x, y]
            elif x >= i1.shape[0]:
                concate[x,y] = i1[x+x_offset, y]
            else:
                if i2[x, y,0] == 0 and i2[x, y,1] == 0 and i2[x, y,2] == 0:
                    concate[x,y] = i1[x+x_offset, y-(i2.shape[1]+y_offset)]
                elif i1[x+x_offset, y-(i2.shape[1]+y_offset),0] == 0 and i1[x+x_offset, y-(i2.shape[1]+y_offset),1] == 0 and i1[x+x_offset, y-(i2.shape[1]+y_offset),2] == 0:
                    concate[x,y] = i2[x, y]
                else:
                    w = (y-(i2.shape[1]+y_offsetf))/abs(y_offsetf)
                    i2_weight = w 
                    concate[x,y] = i2[x, y] * (i2_weight) + i1[x+x_offset, y-(i2.shape[1]+y_offset)] * (1-i2_weight)


plt.imshow(concate)
plt.show()
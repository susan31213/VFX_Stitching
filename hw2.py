import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from random import seed
from random import sample
import math
from cv2 import sort
import argparse

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
            if r > r_thre and ((right and c_idx <= gray.shape[1]*0.55) or ((not right) and c_idx >= gray.shape[1]*0.45)):
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


def feature_matching(corners1, features1, corners2, features2, threshold, debug=False):
    if debug:
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
    if debug:
        matching_cnt = 0
        print('img1\t->\timg2\tdistance')
        for m in match.keys():
            if(match[m][0] != -1):
                matching_cnt += 1
                print('{:d}\t->\t{:d}\t{:f}'.format(match[m][0], m, match[m][1]))
                ax[0].plot(corners1[match[m][0]][1], corners1[match[m][0]][0], '.r', markersize=3)
                ax[0].text(corners1[match[m][0]][1] + 5, corners1[match[m][0]][0] - 5, str(m))
                ax[1].plot(corners2[m][1], corners2[m][0], '.r', markersize=3)
                ax[1].text(corners2[m][1] + 5, corners2[m][0] - 5, str(m))
        print('features1: {:d}, features2: {:d}'.format(len(features1), len(features2)))
        print('total match: {:d}\n----------------------'.format(matching_cnt))
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


def find_translation(c1, c2, match_dict, debug=False):
    # RANSAC: find how to translate proj1 to proj2 (move right)
    best_m1 = 0
    best_m2 = 0
    best_inliner = 0
    err_thre = 1.5
    n = 1
    K = math.log10(0.001)/math.log10(1-math.pow(0.1,n))
    if debug:
        print('RANSAC ' + str(int(K)) + ' times')
    for k in match_dict.keys():
        # print(int(len(c1)*0.3))
        # subset = sample(match_dict.keys(), n)
        m1 = 0
        m2 = 0
        m1 += c2[k][0] - c1[match_dict[k][0]][0]
        m2 += c2[k][1] - c1[match_dict[k][0]][1]
        m1 /= float(n)
        m2 /= float(n)
        inliner = 0
        for s in match_dict.keys():
            if s == k:
                continue
            # print(subset, s, abs(c2[s][0] - c1[match_dict[s][0]][0] - m1) + abs(c2[s][1] - c1[match_dict[s][0]][1] - m2))
            if ((c2[s][0] - c1[match_dict[s][0]][0] - m1)**2 + (c2[s][1] - c1[match_dict[s][0]][1] - m2)**2)**0.5 < err_thre:
                inliner += 1
            
        if inliner > best_inliner:
            best_inliner = inliner
            best_m1 = m1
            best_m2 = m2
    print('x offset: {0}, y offset: {1}, highest # of inliners: {2}'.format(best_m1, best_m2, best_inliner))
    return best_m1, best_m2, best_inliner


def stitching(imgs, match_dict, offsets, blender, debug_plot=False, c1=None, c2=None):
    i1 = imgs[0].copy()
    i2 = imgs[1].copy()
    x_offsetf, y_offsetf = offsets[0], offsets[1]
    
    x_offset = int(round(x_offsetf))
    y_offset = int(round(y_offsetf))
    concate = np.zeros((i1.shape[0]+abs(x_offset), i1.shape[1]+y_offset, 3), dtype=np.uint8)

    # Apply offset
    if x_offset < 0:
        if debug_plot:
            for m in match_dict.keys():
                plt.plot(c1_proj[match_dict[m][0]][1]+y_offset, c1_proj[match_dict[m][0]][0], '.b', markersize=6)
                plt.plot(c2_proj[m][1], c2_proj[m][0]-x_offset, '.r', markersize=3)
        concate[:x_offset, :y_offset, :] = i2[:,:y_offset,:]
        concate[-x_offset:, i2.shape[1]:, :] = i1[:,i2.shape[1]-y_offset:,:]
        # Fill in pixels
        blend_area = {}
        for x in range(0,concate.shape[0]):
            for y in range(y_offset, i2.shape[1]):
                if x+x_offset < 0:
                    concate[x,y] = i1[x, y-y_offset]
                elif x >= i1.shape[0]:
                    concate[x,y] = i2[x+x_offset, y]
                else:
                    if i2[x+x_offset, y,0] == 0 and i2[x+x_offset, y,1] == 0 and i2[x+x_offset, y,2] == 0:
                        concate[x,y] = i1[x, y-y_offset]
                    elif i1[x, y-y_offset,0] == 0 and i1[x, y-y_offset,1] == 0 and i1[x, y-y_offset,2] == 0:
                        concate[x,y] = i2[x+x_offset, y]
                    else:
                        if blender == 'alpha':
                            # alpha blend
                            w = ((y-y_offsetf)/(i2.shape[1]-y_offsetf))
                            i2_weight = 1-w
                            concate[x,y] = i2[x+x_offset, y] * (i2_weight) + i1[x, y-y_offset] * (1-i2_weight)
                        elif blender == 'min-error-alpha':
                            # record blending area
                            if y not in blend_area:
                                blend_area[y] = [x, x]
                            else:
                                if blend_area[y][0] > x:
                                    blend_area[y][0] = x;
                                elif blend_area[y][1] < x:
                                    blend_area[y][1] = x
        if blender == 'min-error-alpha':                
            order = sorted(blend_area.keys())
            bandwidth = args['bandwidth']
            w = int(bandwidth/2)
            min_err = 100000000000
            min_colidx = 0
            for col in range(order[0]+w, order[-1]-w):
                err = 0
                for x in range(col-w, col+w):
                    for y in range(blend_area[col][0], blend_area[col][1]):
                        diff = i2[x+x_offset, y] - i1[x, y-y_offset]
                        err += (diff[0]**2+diff[1]**2+diff[2]**2)**0.5
                if err < min_err:
                    min_err = err
                    min_colidx = col
            print(min_colidx, min_err)
            for x in range(x_offset, i2.shape[0]):
                for y in range(order[0], min_colidx-w):
                    if concate[x,y][0] == 255 and concate[x,y][1] == 0 and concate[x,y][2] == 0:
                        concate[x,y] = i2[x+x_offset, y]
            for x in range(x_offset, i2.shape[0]):
                for y in range(min_colidx-w, min_colidx+w+1):
                    if concate[x,y][0] == 255 and concate[x,y][1] == 0 and concate[x,y][2] == 0:
                        weight = (y-min_colidx+w)/bandwidth
                        concate[x,y] = i2[x+x_offset, y] * weight + i1[x, y-y_offset] * (1-weight)
            for x in range(x_offset, i2.shape[0]):
                for y in range(min_colidx+w+1, order[-1]+1):
                    if concate[x,y][0] == 255 and concate[x,y][1] == 0 and concate[x,y][2] == 0:
                        concate[x,y] = i1[x, y-y_offset]

    else:
        if debug_plot:
            for m in match_dict.keys():
                plt.plot(c1_proj[match_dict[m][0]][1]+y_offset, c1_proj[match_dict[m][0]][0]+x_offset, '.b', markersize=4)
                plt.plot(c2_proj[m][1], c2_proj[m][0], '.r', markersize=3)
        concate[:i2.shape[0], :y_offset, :] = i2[:,:y_offset,:]
        concate[x_offset:, i2.shape[1]:, :] = i1[:,i1.shape[1]-y_offset:,:]
        # Fill in pixels
        blend_area = {}     # {y_axis value: [min on x_axis, max on x_axis]}
        for x in range(0,concate.shape[0]):
            for y in range(y_offset, i2.shape[1]):
                if x-x_offset < 0:
                    concate[x,y] = i2[x, y]
                elif x >= i2.shape[0]:
                    concate[x,y] = i1[x-x_offset, y-y_offset]
                else:
                    if i2[x, y,0] == 0 and i2[x, y,1] == 0 and i2[x, y,2] == 0:
                        concate[x,y] = i1[x-x_offset, y-y_offset]
                    elif i1[x-x_offset, y-y_offset,0] == 0 and i1[x-x_offset, y-y_offset,1] == 0 and i1[x-x_offset, y-y_offset,2] == 0:
                        concate[x,y] = i2[x, y]
                    else:
                        if blender == 'alpha':
                            w = (y-y_offsetf)/(i2.shape[1]-y_offsetf)
                            i2_weight = 1-w 
                            concate[x,y] = i2[x, y] * (i2_weight) + i1[x-x_offset, y-y_offset] * (1-i2_weight)
                        
                        elif blender == 'min-error-alpha':
                            # record blending area
                            if y not in blend_area:
                                blend_area[y] = [x, x]
                            else:
                                if blend_area[y][0] > x:
                                    blend_area[y][0] = x;
                                elif blend_area[y][1] < x:
                                    blend_area[y][1] = x
                            concate[x,y] = [255,0,0]
                        
                        
        if blender == 'min-error-alpha':
            order = sorted(blend_area.keys())
            bandwidth = args['bandwidth']
            w = int(bandwidth/2)
            min_err = 100000000000
            min_colidx = 0
            for col in range(order[0]+w, order[-1]-w):
                err = 0
                for x in range(col-w, col+w):
                    for y in range(blend_area[col][0], blend_area[col][1]):
                        diff = i2[y,x] - i1[y-y_offset, x-x_offset]
                        err += (diff[0]**2+diff[1]**2+diff[2]**2)**0.5
                if err < min_err:
                    min_err = err
                    min_colidx = col
            print(min_colidx, min_err)
            for x in range(x_offset, i2.shape[0]):
                for y in range(order[0], min_colidx-w):
                    # print(order[0], min_colidx-w, y)
                    if concate[x,y][0] == 255 and concate[x,y][1] == 0 and concate[x,y][2] == 0:
                        concate[x,y] = i2[x,y]
            for x in range(x_offset, i2.shape[0]):
                for y in range(min_colidx-w, min_colidx+w+1):
                    if concate[x,y][0] == 255 and concate[x,y][1] == 0 and concate[x,y][2] == 0:
                        weight = (y-min_colidx+w)/bandwidth
                        concate[x,y] = i2[x,y] * weight + i1[x-x_offset, y-y_offset] * (1-weight)
            for x in range(x_offset, i2.shape[0]):
                for y in range(min_colidx+w+1, order[-1]+1):
                    if concate[x,y][0] == 255 and concate[x,y][1] == 0 and concate[x,y][2] == 0:
                        concate[x,y] = i1[x-x_offset, y-y_offset]

    if debug_plot:    
        for x in range(concate.shape[0]):
            plt.plot(y_offset, x, '.g', markersize=3)
            plt.plot(i2.shape[1], x, '.g', markersize=3)
        plt.imshow(concate)
        plt.show()
    return concate



if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A Python implementation of image stitching")
    ap.add_argument('-n', '--prefix-filename', required=True,
                    help="image file path and image prefix name, need pano.txt\
                          for example: parrington/prtn")
    ap.add_argument('-f', '--format', required=True,
                    help="input image format")
    ap.add_argument('-i', '--number', required=True, type=int,
                    help="the number of input images")
    ap.add_argument('-o', '--output', required=True,
                    help="ouput file name")
    ap.add_argument('-b', '--blender', required=False,
                    help="blend method when stitching, default is alpha \
                          [alpha, min-error-alpha]")
    ap.add_argument('-w', '--bandwidth', required=False, type=int,
                    help="bandwidth of min error alpha blend method, default is 3 \
                          [alpha, min error alpha]")
    ap.add_argument('-D', '--debug', required=False, action='store_true',
                    help="option to show debug messages")
    ap.add_argument('-C', '--clip', required=False, action='store_true',
                    help="option to clip into rect image without black boundary")                    
    args = vars(ap.parse_args())

    args['blender'] = 'alpha' if args['blender'] == None else args['blender']
    args['bandwidth'] = 3 if args['bandwidth'] == None else args['bandwidth']


    splitIdx = args['prefix_filename'].rfind('/')
    # Get focal lengthes
    focal_length = []
    file = open('{0}/pano.txt'.format(args['prefix_filename'][:splitIdx]), 'r')
    lines = file.readlines()
    for i, l in enumerate(lines):
        if i != 0 and i % 13 == 11:
            focal_length.append(float(l))

    start = 0
    end = args['number']
    projs = []
    offsets = []
    stitch = []
    first = True
    print("{:s}{:02d}.{:s}".format(args['prefix_filename'], 0, args['format']))
    for img_idx in range(start, end):
        if len(projs) == 0:
            img1 = cv2.imread("{:s}{:02d}.{:s}".format(args['prefix_filename'], img_idx, args['format']))
        img2 = cv2.imread("{:s}{:02d}.{:s}".format(args['prefix_filename'], img_idx+1, args['format']))
        print("{:s}{:02d}.{:s}".format(args['prefix_filename'], img_idx+1, args['format']))

        # feature dectection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
        c1, f1 = feature_detection(gray1, 0.05, 0.01, right=True)
        c2, f2 = feature_detection(gray2, 0.05, 0.01, right=False)

        # feature matching
        match_dict = feature_matching(c1, f1, c2, f2, 1.8, debug=args['debug'])  # key: f2 index, value: [f1 index, error]

        # Cylindrical warping
        proj1, c1_proj = cylindrical_warping(img1, focal_length[img_idx], c1)
        proj2, c2_proj = cylindrical_warping(img2, focal_length[img_idx+1], c2)
        if len(projs) == 0:
            projs.append(proj1)
        projs.append(proj2)

        # fit translation model
        x_offsetf, y_offsetf, _ = find_translation(c1_proj, c2_proj, match_dict)
        offsets.append([int(x_offsetf), int(y_offsetf)])

        img1 = img2

        # stitching
        stitch.append(stitching([proj1, proj2], match_dict, [x_offsetf, y_offsetf], args['blender'], debug_plot=args['debug'], c1=c1_proj, c2=c2_proj))

    result = stitch[0]
    for i in range(len(stitch)-1):
        height_diff = result.shape[0] - stitch[i+1].shape[0]
        if height_diff >= 0:
            result = np.vstack([np.zeros((offsets[i+1][0], result.shape[1], 3), dtype=np.uint8), result])
            stitch[i+1] = np.vstack([stitch[i+1], np.zeros((offsets[i+1][0]+height_diff, stitch[i+1].shape[1], 3), dtype=np.uint8)])
        else:
            result = np.vstack([np.zeros((offsets[i+1][0], result.shape[1], 3), dtype=np.uint8), result])
            stitch[i+1] = np.vstack([stitch[i+1], np.zeros((offsets[i+1][0]+height_diff, stitch[i+1].shape[1], 3), dtype=np.uint8)])
        
        result = np.hstack([stitch[i+1][:, :-(projs[i+1].shape[1]-offsets[i+1][1]), :], result[:,offsets[i+1][1]:, :]])

    # clip
    if args['clip']:
        l, r, u, d = -1, -1, -1, -1
        cnt = 0
        while(True):
            if result[int(result.shape[0]/2), cnt, 0] != 0 or result[int(result.shape[0]/2), cnt, 1] != 0 or result[int(result.shape[0]/2), cnt, 2] != 0:
                l = cnt
                break
            else:
                cnt += 1
        cnt = 0
        while(True):
            if result[int(result.shape[0]/2), -cnt, 0] != 0 or result[int(result.shape[0]/2), -cnt, 1] != 0 or result[int(result.shape[0]/2), -cnt, 2] != 0:
                r = cnt
                break
            else:
                cnt += 1
        cnt = 0
        while(True):
            if result[-cnt, l, 0] != 0 or result[-cnt, l, 1] != 0 or result[-cnt, l, 2] != 0:
                d = cnt
                break
            else:
                cnt += 1
        cnt = 0
        while(True):
            if result[cnt, -r, 0] != 0 or result[cnt, -r, 1] != 0 or result[cnt, -r, 2] != 0:
                u = cnt
                break
            else:
                cnt += 1
        print(l, r, u, d)
        cv2.imwrite('{0}'.format(args['output']), result[u:-d, l:-r, :])
    else:
        cv2.imwrite('{0}'.format(args['output']), result)

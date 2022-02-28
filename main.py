import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from sklearn.cluster import DBSCAN
import BoundingBox
from scipy.spatial import distance as dist
from tracker import CentroidTracker
from sklearn import linear_model, datasets


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def animate(num):
    if num == 0:
        time.sleep(2)
    global ang3
    #global ang3
    global sign
    plt.clf()
    plt.subplot(polar=False)
    plt.grid(linewidth=0.5)
    plt.axis((-18, +18, -18, +18))
    # clustering
    clustering = DBSCAN(eps=0.8, min_samples=1).fit(np.array(data2[num]))

    # create the cluster
    list_cluster = []
    for lu in range(np.amax(clustering.labels_) + 1):
        list_cluster.append([])

    for la in range(len(data[num])):
        list_cluster[clustering.labels_[la]].append(data2[num][la])

    # tracking
    dict_rect = {}
    dict_points = {}
    dict_vertex = {}
    max_ratio = 0.0
    pos_max = -1
    for cluster in list_cluster:
        if len(cluster) < 4:
            continue
        rect = (BoundingBox.MinimumBoundingBox(cluster))
        min_side = min(rect.length_parallel, rect.length_orthogonal)
        max_side = max(rect.length_parallel, rect.length_orthogonal)
        if max_side / min_side > max_ratio and rect.area > 2 and min_side < 0.6 and max_side > 3:
            max_ratio = max_side / min_side
            pos_max = rect.rectangle_center

        center = rect.rectangle_center
        dict_rect[center] = list(rect.corner_points)
        dict_points[center] = list(cluster)
        dict_vertex[center] = order_points(np.array(list(rect.corner_points)))
    obj, col = trk.update(dict_rect.keys())

    c = "black"
    for key, value in obj.items():
        if tuple(value) == pos_max:
            # ransac
            ransac = linear_model.RANSACRegressor()
            # id -> [[x1,y1],[]]
            X = np.array(list(map(lambda elem: elem[0], dict_points[tuple(value)]))).reshape(
                (len(dict_points[tuple(value)]), 1))
            y = np.array(list(map(lambda elem: elem[1], dict_points[tuple(value)])))

            ransac.fit(X, y)

            # Predict data of estimated models
            line_X = np.arange(X.min(), X.max(), (X.max() - X.min()) / len(dict_points[tuple(value)]))[:, np.newaxis]
            line_y_ransac = ransac.predict(line_X)

            # plt.plot(
            #     line_X,
            #     line_y_ransac,
            #     color="red",
            #     linewidth=2,
            #     label="RANSAC regressor",
            # )
            old_ang = ang3
            old_sign = sign
            coef = ransac.estimator_.coef_
            sign = -1
            if 0 < coef < 1 or 0 > coef > -1:
                ang3 = old_ang
                sign = old_sign
            elif coef > 1:
                ang3 = math.pi / 2 - math.atan(coef)
                sign = +1
            elif coef < -1:
                sign = -1
                ang3 = math.atan(coef) + math.pi / 2

            break


            # compute angle
            '''min_y = +100
            max_y = -100
            min_x = +100
            max_x = -100
            for pos in range(len(dict_points[tuple(value)])):
                if dict_points[tuple(value)][pos][1] < min_y:
                    min_y = dict_points[tuple(value)][pos][1]
                    min_x = dict_points[tuple(value)][pos][0]
                if dict_points[tuple(value)][pos][1] > max_y:
                    max_y = dict_points[tuple(value)][pos][1]
                    max_x = dict_points[tuple(value)][pos][0]
            l_ad = max_y - min_y
            l_op = max_x - min_x
            old_a = ang
            if min_x > max_x:
                ang = - abs(math.atan(l_op / l_ad))
            else:
                ang = abs(math.atan(l_op / l_ad))

            if ang > math.pi / 4:
                ang2 = old_a
            else:
                ang2 = ang
            print(min_x, min_y, max_x, max_y)
            print(key, ang)
            break'''

    print(num, ang3, sign)
    for key, value in obj.items():
        if tuple(value) in dict_points:
            # convert back to polar if needed
            for pos in range(len(dict_points[tuple(value)])):
                r = math.sqrt(dict_points[tuple(value)][pos][1] ** 2 + dict_points[tuple(value)][pos][0] ** 2)
                a = np.arctan2(dict_points[tuple(value)][pos][1],
                               dict_points[tuple(value)][pos][0]) 
                dict_points[tuple(value)][pos] = (a, r)
            # convert back to cartesian
            for pos in range(len(dict_points[tuple(value)])):
                x = dict_points[tuple(value)][pos][1] * math.cos(dict_points[tuple(value)][pos][0])
                y = dict_points[tuple(value)][pos][1] * math.sin(dict_points[tuple(value)][pos][0])
                dict_points[tuple(value)][pos] = (x, y)
            if key == 1:
                c = "red"
            else:
                c = "black"

        if tuple(value) in dict_points:
            plt.scatter([t[0] for t in dict_points[tuple(value)]], [t[1] for t in dict_points[tuple(value)]], s=5,
                        c=col[key])

            # plot the rectangles
            '''plt.plot([dict_vertex[tuple(value)][0][0], dict_vertex[tuple(value)][1][0]], [dict_vertex[tuple(value)][0][1], dict_vertex[tuple(value)][1][1]], c=c,
                     linewidth=0.5)
            plt.plot([dict_vertex[tuple(value)][1][0], dict_vertex[tuple(value)][2][0]], [dict_vertex[tuple(value)][1][1], dict_vertex[tuple(value)][2][1]], c=c,
                     linewidth=0.5)
            plt.plot([dict_vertex[tuple(value)][2][0], dict_vertex[tuple(value)][3][0]], [dict_vertex[tuple(value)][2][1], dict_vertex[tuple(value)][3][1]], c=c,
                     linewidth=0.5)
            plt.plot([dict_vertex[tuple(value)][3][0], dict_vertex[tuple(value)][0][0]], [dict_vertex[tuple(value)][3][1], dict_vertex[tuple(value)][0][1]], c=c,
                     linewidth=0.5)'''


file = open("./12-02-38.yd", "rb")
final_data = pickle.load(file)
file.close()

data = []
for i in range(len(final_data)):
    row = []
    for j in range(len(final_data[i][0])):
        if 12 > final_data[i][1][j] > -12:
            row.append([final_data[i][0][j], final_data[i][1][j]])
    # filter distance 0
    row = list(filter(lambda xa: xa[1] != 0.0, row))
    data.append(row)

# fix the orientation
for i in range(len(data)):
    for j in range(len(data[i])):
        newTup = (data[i][j][0] + math.pi - math.pi / 19, data[i][j][1])
        data[i][j] = newTup

# convert to cartesian x dbscan
data2 = data
for i in range(len(data)):
    for j in range(len(data[i])):
        x = - data[i][j][1] * math.cos(data[i][j][0])
        y = data[i][j][1] * math.sin(data[i][j][0])
        data2[i][j] = (x, y)

# animate
fig = plt.figure()
trk = CentroidTracker(40)
ang3 = 0
sign = 1
ani2 = animation.FuncAnimation(fig, animate, interval=100)
plt.show()

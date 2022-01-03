import random
import math
import numpy as np
import pickle
import time

import Features_file
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def findAngle(M1, M2):
    PI = np.pi

    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))

    # Calculate tan inverse of the angle
    ret = math.atan(angle)

    # Convert the angle from
    # radian to degree
    val = (ret * 180) / PI

    # Print the result
    return round(val, 4)


FeatureMAP = Features_file.Features()

file = open("filter1", "rb")
data = pickle.load(file)
# plt.axes(projection = 'polar')

list_x = []
list_y = []
list_new = []
for p in data:
    if p[0] == 0:
        continue

    list_new.append([p[0] * math.cos(p[1]), p[0] * math.sin(p[1])])

file2 = open("filter2", "rb")
data2 = pickle.load(file2)

for p in data2:
    list_new.append([p[0] * math.cos(p[1]) + 5.07, p[0] * math.sin(p[1]) + 8.97])

# list_new.append([4, 2])
# list_new.append([4.01, 2.2])
# list_new.append([4.02, 2.4])
# list_new.append([4, 2.7])
# list_new.append([4.04, 3])
# list_new.append([4.06, 3.2])
# list_new.append([4.1, 3.4])

clustering = DBSCAN(eps=0.7, min_samples=5).fit(np.array(list_new))
color = ["cyan", "yellow", "palegreen", "green", "pink", "grey", "teal", "orange", "gold", "purple", "lime", "peru",
         "yellow", "palegreen", "red"]

# create the cluster
list_cluster = []
for i in range(np.amax(clustering.labels_) + 1):
    list_cluster.append([])

for i in range(len(list_new)):
    list_cluster[clustering.labels_[i]].append(list_new[i])

# segmentation
t1 = time.time()
t_par = time.time()

i = 0
for c in list_cluster:
    print(time.time() - t_par)
    t_par = time.time()
    list_lines = []

    if i == 7:
        i += 1
        continue
    i += 1
    # convert to polar
    list_polar = []
    for point in c:
        list_polar.append([math.sqrt(point[0] ** 2 + point[1] ** 2), math.atan2(point[1], point[0]), (0, 0)])
    FeatureMAP = Features_file.Features()
    running = True
    FeatureDetection = True
    BREAK_POINT_IND = 0
    FeatureDetection = True
    BREAK_POINT_IND = 0
    END_POINTS = [0, 0]
    PREDICTED_POINTS_DRAW = []

    position = (0, 0)
    # laser.position = position
    sensor_data = list_polar
    FeatureMAP.laser_point_set(sensor_data)
    while BREAK_POINT_IND < (FeatureMAP.NP - FeatureMAP.PMIN):
        seed_seg = FeatureMAP.seed_segment_detection(position, BREAK_POINT_IND)
        if not seed_seg:
            break
        else:
            seedSegment = seed_seg[0]
            PREDICTED_POINTS_DRAW = seed_seg[1]
            INDICES = seed_seg[2]
            results = FeatureMAP.seed_segment_growing(INDICES, BREAK_POINT_IND)
            if not results:
                BREAK_POINT_IND = INDICES[1]
                continue
            else:
                line_eq = results[1]
                m, c = results[5]
                line_seg = results[0]
                OUTERMOST = results[2]
                BREAK_POINT_IND = results[3]

                END_POINTS[0] = FeatureMAP.projection_point2line(OUTERMOST[0], m, c)
                END_POINTS[1] = FeatureMAP.projection_point2line(OUTERMOST[1], m, c)

                # plt.plot([END_POINTS[0][0], END_POINTS[1][0]], [END_POINTS[0][1], END_POINTS[1][1]], c="red")
                list_lines.append([END_POINTS[0], END_POINTS[1]])
                #plt.plot([END_POINTS[0][0], END_POINTS[1][0]], [END_POINTS[0][1], END_POINTS[1][1]], c="black")

    # look for possibly contiguous pieces
    '''if list_lines != []:
        list_res_line = [list_lines[0]]
        pos = -1
        for x in range(1, len(list_lines)):
            pos += 1
            m1, b1 = FeatureMAP.points_2line(list_res_line[pos][0], list_res_line[pos][1])
            m2, b2 = FeatureMAP.points_2line(list_lines[x][0], list_lines[x][1])
            delta = findAngle(m1, m2) * np.pi / 180

            if delta < 0.5:
                max_y = max(max(list_res_line[pos][1][1], list_res_line[pos][0][1]),
                            max(list_lines[x][1][1], list_lines[x][0][1]))
                min_y = min(min(list_res_line[pos][1][1], list_res_line[pos][0][1]),
                            min(list_lines[x][1][1], list_lines[x][0][1]))
                max_x = max(max(list_res_line[pos][1][0], list_res_line[pos][0][0]),
                            max(list_lines[x][1][0], list_lines[x][0][0]))
                min_x = min(min(list_res_line[pos][1][0], list_res_line[pos][0][0]),
                            min(list_lines[x][1][0], list_lines[x][0][0]))
                list_res_line[pos] = [(min_x, min_y), (max_x, max_y)]
                pos -= 1
            else:
                list_res_line.append(list_lines[x])

    for elem in list_res_line:
        pass'''
        #plt.plot([elem[0][0], elem[1][0]], [elem[0][1], elem[1][1]], c="black")

t2 = time.time()
print("final")
print(t2-t1)

color = ["cyan", "yellow", "palegreen", "green", "pink", "grey", "teal", "orange", "gold", "purple", "lime", "peru",
         "yellow", "palegreen", "red"]
'''for w in range(len(list_cluster)):
    if w == 7:
        continue
    for i in range(len(list_cluster[w])):
        plt.scatter(list_cluster[w][i][0], list_cluster[w][i][1], s=5, c=color[w % 14])
plt.show()'''

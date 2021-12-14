import pickle
from sklearn.cluster import DBSCAN
import math
import matplotlib.pyplot as plt
import numpy as np


def distPointLine(point, A, B, C):
    return abs(A * point[0] + B * point[1] + C) / math.sqrt(A * A + B * B)


def split_merge(pointList, eps):
    dmax = 0
    index = 0
    end = len(pointList)
    # A = y2 - y1 / x2 - x1

    varA = float(pointList[end - 1][1] - pointList[0][1]) / float(pointList[end - 1][0] - pointList[0][0])
    # B = -1
    varB = -1.00
    # C = y - Ax
    varC = float(pointList[0][1] - varA * pointList[0][0])
    for j in range(0, end):
        d = distPointLine(pointList[j], varA, varB, varC)
        if d > dmax:
            index = j
            dmax = d

    if dmax > eps:
        res1 = split_merge(pointList[0:index + 1], eps)
        res2 = split_merge(pointList[index:end], eps)
        result = res1 + res2
    else:
        result = [pointList[0], pointList[end - 1]]

    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    info = ()
    file = open("filter1", "rb")
    data = pickle.load(file)
    # plt.axes(projection = 'polar')

    list_r = []
    list_a = []
    list_x = []
    list_y = []
    list_new = []
    for p in data:
        if p[0] == 0:
            continue

        list_r.append(p[0])
        list_a.append(p[1])
        list_x.append(p[0] * math.cos(p[1]))
        list_y.append(p[0] * math.sin(p[1]))
        list_new.append([p[0] * math.cos(p[1]), p[0] * math.sin(p[1])])

    file2 = open("filter2", "rb")
    data2 = pickle.load(file2)

    for p in data2:
        list_r.append(p[0])
        list_a.append(p[1])
        list_x.append(p[0] * math.cos(p[1]) + 5.07)
        list_y.append(p[0] * math.sin(p[1]) + 8.97)
        list_new.append([p[0] * math.cos(p[1]) + 5.07, p[0] * math.sin(p[1]) + 8.97])

    list_x.append(4)
    list_y.append(2)
    list_new.append([4, 2])

    list_x.append(4)
    list_y.append(2.2)
    list_new.append([4, 2.2])

    list_x.append(4)
    list_y.append(2.4)
    list_new.append([4, 2.4])

    list_x.append(4)
    list_y.append(2.7)
    list_new.append([4, 2.7])

    list_x.append(4)
    list_y.append(3)
    list_new.append([4, 3])

    list_x.append(4)
    list_y.append(3.2)
    list_new.append([4, 3.2])

    list_x.append(4)
    list_y.append(3.4)
    list_new.append([4, 3.4])

    clustering = DBSCAN(eps=0.7, min_samples=5).fit(np.array(list_new))
    color = ["blue", "yellow", "black", "green", "pink", "grey", "teal", "orange", "gold", "purple", "lime", "peru",
             "cyan", "palegreen", "red"]

    list_cluster = []
    for i in range(np.amax(clustering.labels_) + 1):
        list_cluster.append([])

    for i in range(len(list_x)):
        list_cluster[clustering.labels_[i]].append(list_new[i])

    ##############################################################
    for w in range(len(list_cluster)):
        if w == 7:
            print(list_cluster[w])
            continue
        list_cluster[w].sort()
        res = split_merge(list_cluster[w], 0.8)

        for i in range(len(list_cluster[w])):
            plt.scatter(list_cluster[w][i][0], list_cluster[w][i][1], s=5, c=color[w])

        for i in range(len(res) - 1):
            plt.plot([res[i][0], res[i + 1][0]], [res[i][1], res[i + 1][1]], c="black")

    plt.show()

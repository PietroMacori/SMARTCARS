import pickle
from sklearn.cluster import DBSCAN
import math
import matplotlib.pyplot as plt
import numpy as np


def measPointToPoint(varP1, varP2):
    return math.sqrt(((varP2[0] - varP1[0]) * (varP2[0] - varP1[0])) + ((varP2[1] - varP1[1]) * (varP2[1] - varP1[1])))


# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
def measPointToLine(varPk, varPl, varP0):
    return abs(
        (varPl[1] - varPk[1]) * varP0[0] - (varPl[0] - varPk[0]) * varP0[1] + varPl[0] * varPk[1] - varPl[1] * varPk[
            0]) / math.sqrt(math.pow((varPl[0] - varPk[0]), 2) + math.pow((varPl[1] - varPk[1]), 2))


# Fungsi IEPF
# Input berupa list koordinat titik dan list endpoint dari titik
# Ouput berupa list persamaan garis dalam bentuk Ax + By + C = 0
def iepfFunction(dThreshold, ptList, ePtList):
    # print ePtList
    maxDPtToLine = 0
    breakPointIndex = -1
    _, jumlahEndpoint = ePtList.shape
    # loop sebanyak jumlah end point yang diinputkan
    for i in range(0, jumlahEndpoint - 1):
        # A = y2 - y1 / x2 - x1
        varA = float(ePtList[1, i + 1] - ePtList[1, i]) / float(ePtList[0, i + 1] - ePtList[0, i])
        # B = -1
        varB = -1.00
        # C = y - Ax
        varC = float(ePtList[1, i] - varA * ePtList[0, i])
        # print 'IEPF Line Function {}x  + {}y + {} = 0'.format(varA, varB, varC)
        # loop sebanyak jumlah titik yang berada diantara endpoint

        for j in range(int(ePtList[2, i]), int(ePtList[2, i + 1])):
            if j == 0 or j == ePtList[2, i]:
                continue
            # Pengukuran jarak titik ke line
            # d = | ax1 + by1 + c / sqrt(a^2 + b^2) |
            dPtToLine = float(
                abs((varA * ptList[0, j] + varB * ptList[1, j] + varC) / (math.sqrt(varA * varA + varB * varB))))
            if dPtToLine > dThreshold:
                if dPtToLine > maxDPtToLine:
                    maxDPtToLine = dPtToLine
                    breakPointIndex = j
    if breakPointIndex != -1:
        y = np.array([[ptList[0, breakPointIndex]], [ptList[1, breakPointIndex]], [breakPointIndex]])
        print("---")
        print(y)
        print("***")

        check = False
        for j in range(len(ePtList[0, :])):
            if ePtList[2, j] == y[2, 0]:
                check = True

        if not check:
            y = np.array([[ptList[0, breakPointIndex]], [ptList[1, breakPointIndex]], [breakPointIndex]])

            ePtList = np.insert(ePtList, [jumlahEndpoint - 1], y, axis=1)
            ePtList = iepfFunction(dThreshold, ptList, ePtList)

    return ePtList


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
    list_cluster[0].sort()
    list_x = np.array([item[0] for item in list_cluster[0]])
    list_y = np.array([item[1] for item in list_cluster[0]])

    list_res = np.vstack((list_x, list_y))

    npPoint = list_res
    endPoint0 = 0
    endPointN = npPoint[0].size - 1

    npEndpoint = np.zeros((3, 2), dtype=float)

    npEndpoint[0, 0] = npPoint[0, endPoint0]
    npEndpoint[1, 0] = npPoint[1, endPoint0]
    npEndpoint[2, 0] = endPoint0

    npEndpoint[0, 1] = npPoint[0, endPointN]
    npEndpoint[1, 1] = npPoint[1, endPointN]
    npEndpoint[2, 1] = endPointN

    # fungsi IEPF dengan threshold 100
    predictedLine = iepfFunction(0.1, npPoint, npEndpoint)

    _, jumlahEndpoint = predictedLine.shape
    jumlahGaris = jumlahEndpoint - 1

    clusterPoint = np.zeros((jumlahGaris, 2, 1))

    print(predictedLine)

    for i in range(len(list_x)):
        plt.scatter(list_x[i], list_y[i], c="red", s=7)

    for i in range(len(predictedLine[0]) - 1):
        plt.plot([predictedLine[0][i], predictedLine[0][i + 1]], [predictedLine[1][i], predictedLine[1][i + 1]],
                 c=color[i])
    plt.show()

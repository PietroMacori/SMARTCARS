import math
import time
from collections import defaultdict

# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import timeit
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd


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
        for j in range(ePtList[2, i], ePtList[2, i + 1]):
            if j == 0 or j == ePtList[2, i]:
                continue
            # Pengukuran jarak titik ke line
            # d = | ax1 + by1 + c / sqrt(a^2 + b^2) |
            dPtToLine = float(
                abs((varA * ptList[0, j] + varB * ptList[1, j] + varC) / (math.sqrt(varA * varA + varB * varB))))
            if dPtToLine > dThreshold:
                if (dPtToLine > maxDPtToLine):
                    maxDPtToLine = dPtToLine
                    breakPointIndex = j
    if breakPointIndex != -1:
        y = np.array([[ptList[0, breakPointIndex]], [ptList[1, breakPointIndex]], [breakPointIndex]])
        ePtList = np.insert(ePtList, [jumlahEndpoint - 1], y, axis=1)
        ePtList = iepfFunction(dThreshold, ptList, ePtList)

    return ePtList


def mergeLine(mneThreshold, ptList, ePtList):
    jumlahEndpoint = len(ePtList[0])
    for i in range(0, jumlahEndpoint - 2):
        # print 'garis'
        varPk = [ePtList[0][i], ePtList[1][i]]
        varPl = [ePtList[0][i + 2], ePtList[1][i + 2]]
        varP0 = [ePtList[0][i + 1], ePtList[1][i + 1]]
        varMaxDistance = measPointToLine(varPk, varPl, varP0)
        varPk = [ePtList[0][i], ePtList[1][i]]
        varPl = [ePtList[0][i + 1], ePtList[1][i + 1]]
        prevIndex = ePtList[2][i + 1] - 1
        nextIndex = ePtList[2][i + 1] + 1
        # print 'PREV {} NEXT{}'.format(prevIndex, nextIndex)
        varP0 = [ptList[0][prevIndex], ptList[1][nextIndex]]
        # print 'K {},{} L {},{} P0 {},{}'.format(varPk[0], varPk[1], varPl[0], varPl[1], varPl[0], varPl[1])
        print
        varPk
        print
        varPl
        print
        varP0
        xx = measPointToLine(varPk, varPl, varP0)
        varMNEprev = varMaxDistance / xx
        # print 'MNE Prev {}'.format(varMNEprev)
        print
        'Prev list'
        print
        ePtList
        if varMNEprev > 2:
            # remX = ePtList[0][i+1]
            # remY = ePtList[1][i+1]
            # remIndex = ePtList[2][i+1]
            # ePtList[0].remove(remX)
            ePtList[0].pop(i + 1)
            ePtList[1].pop(i + 1)
            ePtList[2].pop(i + 1)
        # print 'K {},{} L {},{}'.format(varPk[0], varPk[1], varPl[0], varPl[1],)
        # for j in range(ePtList[2][i],ePtList[2][i+1]):
        print
        'last list'
        print
        ePtList
    return ePtList


def testSpeed(loop):
    startTime = time.clock()
    for i in range(0, loop):
        P0 = [0, 0]
        P1 = [10, 10]
        measPointToPoint(P0, P1)
    endTime = time.clock()
    totalTime = (endTime - startTime) / loop
    totalTime *= 1000000
    print
    'Total time {} uSeconds'.format(totalTime)


def main_new():
    # Read dataset from CSV
    df = pd.read_csv('D:\Research\IEPF-Line-Extraction\Source-Code\dataset.csv')
    # Convert pandas to np array
    npDataset = df.as_matrix()
    # delete kolom no 0, transpose
    npPoint = np.transpose(np.delete(npDataset, 0, axis=1))
    # Ambil point x
    # npPointX = npPoint[:,0]
    # Ambil point y
    # npPointY = npPoint[:,1]
    # Input 2 buah endpoint untuk masukan awal
    endPoint0 = 0
    endPointN = npPoint[0].size - 1

    npEndpoint = np.zeros((3, 2), dtype=int)

    npEndpoint[0, 0] = npPoint[0, endPoint0]
    npEndpoint[1, 0] = npPoint[1, endPoint0]
    npEndpoint[2, 0] = endPoint0

    npEndpoint[0, 1] = npPoint[0, endPointN]
    npEndpoint[1, 1] = npPoint[1, endPointN]
    npEndpoint[2, 1] = endPointN

    # fungsi IEPF dengan threshold 100
    predictedLine = iepfFunction(50, npPoint, npEndpoint)

    _, jumlahEndpoint = predictedLine.shape
    jumlahGaris = jumlahEndpoint - 1

    clusterPoint = np.zeros((jumlahGaris, 2, 1))

    print
    clusterPoint

    plt.plot(npPoint[0], npPoint[1], 'ro')
    plt.plot(predictedLine[0], predictedLine[1])
    plt.axis([0, 640, 0, 640])
    plt.show()


class TG15:
    def __init__(self, lidar_online=True, to_classify=True, clustering_max_dist=0.1, circle_thresh=20,
                 file_name=None):

        # Max and min scan range
        self.RMAX = 15
        self.RMIN = 0.05
        self.clustering_max_dist = clustering_max_dist
        self.circle_thresh = circle_thresh

        self.labels = {}
        self.lidar_online = lidar_online
        self.to_classify = to_classify
        self.num_scans = -1
        self.colors = ['black', 'lime', 'blue', 'teal', 'violet', 'orange', 'tan', 'peru', 'olive', 'yellow', 'pink',
                       'red',
                       'aquamarine', 'darkred', 'fuchsia']

        self.file_name = file_name
        import pickle
        file = open(self.file_name, 'rb')
        self.data = pickle.load(file)
        self.current_data = self.data[0]

        file.close()

        self.reset()

    def reset(self):
        self.points = []  # Dictionary {point_idx: (angle, range, intensity)}
        self.blocks = defaultdict(list)  # Dictionary {block_idx: [list of point indices]}
        self.labels = {}  # Dictionary {block_idx: class_label}
        # circle (person): (0, [x,y], r), line (forward vehicle and barriers): (1, [(x1,y1), (x2,y2)]),
        # rectangle (other vehicles and buildings): (2, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
        self.angle = []
        self.ran = []
        self.intensity = []
        self.X = []
        self.Y = []
        self.num_points = 0  # The total number of points detected
        self.num_blocks = 0  # The total number of blocks detected
        self.num_scans += 1

    def read_pre_process(self):
        self.reset()
        for point in self.current_data:
            if point[0] >= 0.1:
                if point[1] < 0:
                    a = np.pi * 2 + point[1]
                else:
                    a = point[1]
                self.points.append((point[0], a))
                self.num_points += 1
        self.points.sort(key=(lambda x: x[1]))
        for elem in self.points:
            self.angle.append(elem[1])
            self.ran.append(elem[0])

    def plot_raw_perception(self, lidar_polar):
        # plotting
        lidar_polar.clear()
        for b in range(self.num_blocks):
            for i in self.blocks[b]:
                lidar_polar.scatter(self.angle[i], self.ran[i], cmap='hsv', alpha=0.95, s=14,
                                    color=self.colors[b % len(self.colors)])

    def classify(self):
        t1 = time.time()
        th = 0.02
        for i in range(len(self.ran)):

            if i == len(self.ran) - 1:
                if (abs(self.ran[0] - self.ran[i] * math.cos(self.angle[0] - self.angle[i])) / self.ran[0]) > th:
                    self.num_blocks += 1
            else:
                if (abs(self.ran[i + 1] - self.ran[i] * math.cos(self.angle[i + 1] - self.angle[i])) / self.ran[i]) > th:
                    self.num_blocks += 1

            self.blocks[self.num_blocks].append(i)

        t2 = time.time()

        print(str(t2 - t1))

    def run(self):

        lidar_polar = plt.subplot(polar=True)
        lidar_polar.autoscale_view(True, True, True)
        lidar_polar.set_rmax(self.RMAX)
        lidar_polar.grid(True)
        self.read_pre_process()

        self.classify()

        self.plot_raw_perception(lidar_polar)
        plt.plot()

        plt.figure(2)
        plt.axis([0, np.pi * 2, 0, 7])
        plt.plot(self.angle, self.ran)

        delta = []
        for i in range(len(self.ran)):
            if i == len(self.ran) - 1:
                delta.append(self.ran[0] - self.ran[i] * math.cos(self.angle[0] - self.angle[i]))
            else:
                delta.append(self.ran[i + 1] - self.ran[i] * math.cos(self.angle[i + 1] - self.angle[i]))

        plt.figure(3)
        plt.axis([0, np.pi * 2, 0, 4])
        plt.plot(self.angle, delta)
        plt.show()

        plt.close()

import math
import time
from collections import defaultdict

# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from random import randrange


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


            if i == len(self.ran)-1:
                if (abs(self.ran[0] - self.ran[i] * math.cos(self.angle[0] - self.angle[i]))/self.ran[0]) > th:
                    self.num_blocks += 1
            else:
                if (abs(self.ran[i+1] - self.ran[i] * math.cos(self.angle[i+1] - self.angle[i]))/self.ran[i]) > th:
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
        plt.axis([0, np.pi *2 , 0, 7])
        plt.plot(self.angle, self.ran)

        delta = []
        for i in range(len(self.ran)):
            if i == len(self.ran)-1:
                delta.append(self.ran[0] - self.ran[i] * math.cos(self.angle[0] - self.angle[i]))
            else:
                delta.append(self.ran[i+1] - self.ran[i] * math.cos(self.angle[i+1] - self.angle[i]))

        plt.figure(3)
        plt.axis([0, np.pi *2 , 0, 4])
        plt.plot(self.angle, delta)
        plt.show()

        plt.close()

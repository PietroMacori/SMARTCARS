import math
import time
from collections import defaultdict

# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import ydlidar
from scipy.spatial import ConvexHull
import matplotlib.animation as animation


def cartesian_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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
        self.laser = ydlidar.CYdLidar()
        self.scan = ydlidar.LaserScan()

        if self.lidar_online:
            ports = ydlidar.lidarPortList()
            port = "/dev/ydlidar"
            for key, value in ports.items():
                port = value

            self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
            self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 512000)
            self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
            self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
            self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5.0)
            self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 20)
            self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
            self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
            self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
            self.laser.setlidaropt(ydlidar.LidarPropMaxRange, 32.0)
            self.laser.setlidaropt(ydlidar.LidarPropMinRange, 0.01)

        else:
            self.file_name = file_name
            import pickle
            file = open(self.file_name, 'rb')
            self.data = pickle.load(file)
            file.close()

        self.reset()

    def reset(self):
        self.points = {}  # Dictionary {point_idx: (angle, range, intensity)}
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
        previous_d = 0
        previous_a = 0
        if self.lidar_online:
            r = self.laser.doProcessSimple(self.scan)
            if r:
                for point in self.scan.points:
                    d1 = np.sqrt(point.range ** 2 + previous_d ** 2 - 2 * point.range * previous_d * np.cos(
                        previous_a - point.angle))
                    if point.range >= 0.1 and d1 > 0.02:
                        previous_a = point.angle
                        previous_d = point.range
                        self.points[self.num_points] = (point.range, point.angle, point.intensity)
                        self.angle.append(point.angle)
                        self.ran.append(point.range)
                        self.intensity.append(point.intensity)
                        self.X.append(np.cos(point.angle) * point.range)
                        self.Y.append(np.sin(point.angle) * point.range)
                        self.num_points += 1

        else:
            self.current_data = self.data[self.num_scans]
            for point in self.current_data:
                if point[0] >= 0.2:
                    self.points[self.num_points] = (point[0], point[1], 10)
                    self.X.append(np.cos(point[1]) * point[0])
                    self.Y.append(np.sin(point[1]) * point[0])
                    self.angle.append(point[1])
                    self.ran.append(point[0])
                    self.intensity.append(10)
                    self.num_points += 1

    def polar_dist(self, pi, pj):
        r1 = self.points[pi][0]
        r2 = self.points[pj][0]
        a1 = self.points[pi][1]
        a2 = self.points[pj][1]
        d1 = np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(a2 - a1))
        return d1

    def minimum_bounding_rectangle(self, points):
        r_min = np.inf
        r_max = 0
        a_min = np.pi * 2
        a_max = 0

        real_min = a_min
        real_max = a_max

        for i in points:
            r = self.points[i][0]
            a = self.points[i][1]
            old_a = a
            if a < 0:
                a = np.pi * 2 + a
            if r < r_min:
                r_min = r
            if r > r_max:
                r_max = r
            if a < a_min:
                a_min = a
                real_min = old_a
            if a > a_max:
                a_max = a
                real_max = old_a
        return [r_min, r_max], [real_min, real_max]

    def merge_point_block(self, ip, ib):
        min_dist = np.inf
        for ind in self.blocks[ib]:
            dist = self.polar_dist(ip, ind)
            if dist < min_dist:
                min_dist = dist
        if min_dist >= self.clustering_max_dist:
            return False
        return True

    def create_blocks(self):
        for i in range(self.num_points):
            need_new_block = True

            for j in range(self.num_blocks):
                if self.merge_point_block(i, j):
                    self.blocks[j].append(i)
                    need_new_block = False
                    break
            if need_new_block:
                self.blocks[self.num_blocks].append(i)
                self.num_blocks += 1

    def calc_circle(self, bi):
        X = []
        Y = []
        bl = self.blocks[bi]
        for i in range(len(bl)):
            r = self.points[bl[i]][0]
            a = self.points[bl[i]][1]

            X.append(round(r * np.cos(a), 5))
            Y.append(round(r * np.sin(a), 5))

        center = [np.mean(X), np.mean(Y)]
        radius = 0.0
        for i in range(len(bl)):
            d = cartesian_dist([X[i], Y[i]], center)
            if d > radius:
                radius = d
        return center, radius

    def calc_line_rect(self, bi):
        max_dist = 0.0
        p1_line = None
        p2_line = None
        bl = self.blocks[bi]

        for i in range(len(bl)):
            for j in range(len(bl)):
                d = self.polar_dist(bl[i], bl[j])
                if d > max_dist:
                    max_dist = d
                    p1_line = i
                    p2_line = j

        r1_line = self.ran[self.blocks[bi][p1_line]]
        a1_line = self.angle[self.blocks[bi][p1_line]]
        r2_line = self.ran[self.blocks[bi][p2_line]]
        a2_line = self.angle[self.blocks[bi][p2_line]]

        p1 = np.array([r1_line * np.cos(a1_line), r1_line * np.sin(a1_line)])
        p2 = np.array([r2_line * np.cos(a2_line), r2_line * np.sin(a2_line)])

        tot_dist = np.sqrt((r2_line * np.cos(a2_line) - r1_line * np.cos(a1_line)) ** 2 + (
                r2_line * np.sin(a2_line) - r1_line * np.sin(a1_line)) ** 2)
        avg_dist = 0

        for i in range(len(bl)):
            r = self.points[bl[i]][0]
            a = self.points[bl[i]][1]
            x = r * np.cos(a)
            y = r * np.sin(a)
            p3 = np.array([x, y])
            d = np.abs(np.linalg.norm(np.cross(p2 - p1, p1 - p3))) / np.linalg.norm(p2 - p1)
            avg_dist += d

        avg_dist = avg_dist / len(bl)

        if avg_dist <= 0.2 * tot_dist/2:
            return 1, ([r1_line, r2_line], [a1_line, a2_line])
        else:
            x, y = self.minimum_bounding_rectangle(bl)
            return 2, (x, y)

    def classify(self):
        for i in range(self.num_blocks):
            if len(self.blocks[i]) <= self.circle_thresh:
                center, r = self.calc_circle(i)
                self.labels[i] = (0, center, r)
            else:
                rect, ret = self.calc_line_rect(i)
                if rect == 1:
                    self.labels[i] = (rect, ret[0], ret[1])
                else:
                    self.labels[i] = (rect, ret[0], ret[1])

    def plot_raw_perception(self, fig, lidar_polar):
        print('Scanning...')
        r = self.laser.doProcessSimple(self.scan)
        if r:
            angle = []
            ran = []
            intensity = []
            for point in self.scan.points:
                angle.append(point.angle)
                ran.append(point.range)
                intensity.append(point.intensity)

            # plotting
            lidar_polar.clear()
            lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)

    def plot_classified_perception(self, fig, lidar_polar):
        # circle (person): (0, [x,y], r), line (forward vehicle and barriers): (1, [(x1,y1), (x2,y2)]),
        # rectangle (other vehicles and buildings): (2, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
        # print('Scanning...')
        lidar_polar.clear()

        # times x benchmark
        # t1 = time.time()
        self.read_pre_process()
        # t2 = time.time()
        self.create_blocks()
        # t3 = time.time()
        self.classify()
        # t4 = time.time()

        # print(str(t2 - t1))
        # print(str(t3 - t2))
        # print(str(t4 - t3))

        # plotting
        lidar_polar.scatter(self.angle, self.ran, s=5, cmap='hsv', alpha=0.95)

        for i in range(self.num_blocks):
            if i in self.labels.keys() and self.labels[i][0] == 0:
                # plot circle
                x = self.labels[i][1][0]
                y = self.labels[i][1][1]
                r = self.labels[i][2]
                ran = np.sqrt(x ** 2 + y ** 2)
                if x != 0:
                    theta = math.atan2(y, x)
                else:
                    theta = 0

                if ran is not None and theta is not None:
                    lidar_polar.scatter(theta, ran, s=r * 50 * 20, c='r', cmap='hsv', alpha=0.45)

            elif i in self.labels.keys() and self.labels[i][0] == 1:
                # plot line
                ran = self.labels[i][1]
                theta = self.labels[i][2]

                if ran is not None and theta is not None:
                    lidar_polar.plot(theta, ran, color="orange", linewidth=3, alpha=0.70)
            elif i in self.labels.keys() and self.labels[i][0] == 2:
                # plot rectangle
                r_min = self.labels[i][1][0]
                r_max = self.labels[i][1][1]
                a_min = self.labels[i][2][0]
                a_max = self.labels[i][2][1]

                xy = (a_min, r_min)
                w = r_max - r_min

                if a_min < - np.pi / 2 and a_max > np.pi / 2:
                    a = abs(a_max + a_min)
                elif a_min < 0 and a_max > 0:
                    a = abs(a_max - a_min)
                else:
                    a = abs(abs(a_max) - abs(a_min))

                lidar_polar.add_patch(plt.Rectangle(xy, w, a, fill=False))

    def run(self):
        fig = plt.figure()
        lidar_polar = plt.subplot(polar=True)
        lidar_polar.autoscale_view(True, True, True)
        lidar_polar.set_rmax(self.RMAX)
        lidar_polar.grid(True)
        if self.lidar_online:
            ret = self.laser.initialize()
        else:
            ret = True
        # print('Lidar initialized!')
        if ret:
            if self.lidar_online:
                r = self.laser.turnOn()
            else:
                r = True

            if r:
                if self.to_classify:
                    if self.lidar_online:
                        ani = animation.FuncAnimation(fig, self.plot_classified_perception, interval=100,
                                                      fargs=(lidar_polar,))
                    else:
                        self.plot_classified_perception(fig, lidar_polar)
                else:
                    ani = animation.FuncAnimation(fig, self.plot_raw_perception, interval=2000, fargs=[lidar_polar, ])

                plt.show()

            print('Scan interrupted!')
            if self.lidar_online:
                self.laser.turnOff()
        print('Lidar disconnected!')
        if self.lidar_online:
            self.laser.disconnecting()
        plt.close()

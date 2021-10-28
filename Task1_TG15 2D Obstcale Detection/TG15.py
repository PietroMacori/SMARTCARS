from collections import defaultdict

# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
#import ydlidar
from scipy.spatial import ConvexHull
import matplotlib.animation as animation


def cartesian_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class TG15:
    def __init__(self, lidar_online=True, to_classify=True, clustering_max_dist=1.5, circle_thresh=500, rect_thresh=0.2, file_name=None):

        # Max and min scan range
        self.RMAX = 15
        self.RMIN = 0.05
        self.clustering_max_dist = clustering_max_dist
        self.circle_thresh = circle_thresh
        self.rect_thresh = rect_thresh

        self.labels = {}
        self.lidar_online = lidar_online
        self.to_classify = to_classify
        self.num_scans = -1

        if self.lidar_online:
            ports = ydlidar.lidarPortList()
            port = "/dev/ydlidar"
            for key, value in ports.items():
                port = value

            self.laser = ydlidar.CYdLidar()
            self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
            self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 512000)
            self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
            self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
            self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
            self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 20)
            self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
            self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
            self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
            self.laser.setlidaropt(ydlidar.LidarPropMaxRange, 32.0)
            self.laser.setlidaropt(ydlidar.LidarPropMinRange, 0.01)
            self.scan = ydlidar.LaserScan()
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
        if self.lidar_online:
            r = self.laser.doProcessSimple(self.scan)
            if r:
                for point in self.scan.points:
                    self.points[self.num_points] = (point.angle, point.range, point.intensity)
                    self.angle.append(point.angle)
                    self.ran.append(point.range)
                    self.intensity.append(point.intensity)
                    self.num_points += 1
        else:
            self.current_data = self.data[self.num_scans]
            for point in self.current_data:
                self.points[self.num_points] = (point[0], point[1], 10)
                self.X.append(np.cos(point[0])*point[1])
                self.Y.append(np.sin(point[0])*point[1])
                self.angle.append(point[0])
                self.ran.append(point[1])
                self.intensity.append(10)
                self.num_points += 1


    def polar_dist(self, pi, pj):
        r1 = self.points[pi][1]
        r2 = self.points[pj][1]
        a1 = self.points[pi][0]
        a2 = self.points[pj][0]
        return np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(a2 - a1))

    def minimum_bounding_rectangle(self, points):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :rval: an 4x2 matrix of coordinates
        """
        pi2: float = np.pi / 2.

        # get the convex hull for the points
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points) - 1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)]).T
        #     rotations = np.vstack([
        #         np.cos(angles),
        #         -np.sin(angles),
        #         np.sin(angles),
        #         np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = [0, 0, 0, 0]
        rval[0] = tuple(np.dot([x1, y2], r))
        rval[1] = tuple(np.dot([x2, y2], r))
        rval[2] = tuple(np.dot([x2, y1], r))
        rval[3] = tuple(np.dot([x1, y1], r))

        return rval

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
            r = self.points[bl[i]][1]
            a = self.points[bl[i]][0]
            X.append(r * np.cos(a))
            Y.append(r * np.sin(a))
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
        for i in range(len(bl) - 1):
            for j in range(i + 1, len(bl)):
                d = self.polar_dist(bl[i], bl[j])
                if d > max_dist:
                    max_dist = d
                    p1_line = i
                    p2_line = j

        r1_line = self.points[p1_line][1]
        a1_line = self.points[p1_line][0]
        r2_line = self.points[p2_line][1]
        a2_line = self.points[p2_line][0]
        p1 = np.array([r1_line * np.cos(a1_line), r1_line * np.sin(a1_line)])
        p2 = np.array([r2_line * np.cos(a2_line), r2_line * np.sin(a2_line)])

        coords = []
        dists = []
        is_rect = False
        X = []
        Y = []
        for i in range(len(bl)):
            r = self.points[bl[i]][1]
            a = self.points[bl[i]][0]
            x = r * np.cos(a)
            y = r * np.sin(a)
            X.append(x)
            Y.append(y)
            p3 = np.array([x, y])
            d = np.abs(np.linalg.norm(np.cross(p2 - p1, p1 - p3))) / np.linalg.norm(p2 - p1)
            dists.append(d)
            if d >= self.rect_thresh * max_dist:
                is_rect = True

        if not is_rect:
            coords = [(p1[0], p1[1]), (p2[0], p2[1])]
            return 1, coords

        points = np.array([X, Y]).T
        coords = self.minimum_bounding_rectangle(points)
        return 2, coords

    def classify(self):
        # circle (person): (0, [x,y], r), line (forward vehicle and barriers): (1, [(x1,y1), (x2,y2)]),
        # rectangle (other vehicles and buildings): (2, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
        for i in range(self.num_blocks):
            if len(self.blocks[i]) <= self.circle_thresh:
                center, r = self.calc_circle(i)
                self.labels[i] = (0, center, r)
            else:
                label, coords = self.calc_line_rect(i)
                self.labels[i] = (label, coords)

    def plot_raw_perception(self, lidar_polar):
        print('Scanning...')
        print('Initializing GUI...')
        r = self.laser.doProcessSimple(self.scan)
        if r:
            angle = []
            ran = []
            intensity = []
            for point in self.scan.points:
                angle.append(point.angle)
                ran.append(point.range)
                intensity.append(point.intensity)
                
            ### plotting
            lidar_polar.clear()
            lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)



    def plot_classified_perception(self, lidar_polar):
        # circle (person): (0, [x,y], r), line (forward vehicle and barriers): (1, [(x1,y1), (x2,y2)]),
        # rectangle (other vehicles and buildings): (2, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
        print('Scanning...')
        print('Initializing GUI...')

        self.read_pre_process()
        self.create_blocks()
        self.classify()

        ### plotting
        lidar_polar.clear()
        lidar_polar.scatter(self.angle, self.ran, cmap='hsv', alpha=0.95)
        for i in range(self.num_blocks):
            ## circle
            if i in self.labels.keys() and self.labels[i][0] == 0:
                x = self.labels[i][1][0]
                y = self.labels[i][1][1]
                r = self.labels[i][2]
                ran = np.sqrt(x**2+y**2)
                theta = np.arctan(y/x)
                print(r, theta, x, y)
                lidar_polar.scatter([theta], [ran], c='g', s=np.pi * (r ** 2), cmap='hsv', alpha=0.25)
            ## line
            # elif self.labels[i][0] == 1:
            #     ax.plot(x_values, y_values, color='r')
            # # ## rectangle
            # else:
            #     pass

    def run(self):

        fig = plt.figure()
        fig.canvas.set_window_title('TG15 LIDAR Monitor')
        lidar_polar = plt.subplot(polar=True)
        lidar_polar.autoscale_view(True, True, True)
        lidar_polar.set_rmax(self.RMAX)
        lidar_polar.grid(True)
        if self.lidar_online:
            ret = self.laser.initialize()
        else:
            ret = True
        print('Lidar initialized!')
        if ret:
            if self.lidar_online:
                r = self.laser.turnOn()
            else:
                r = True

            if r:
                if self.to_classify:
                    #ani = animation.FuncAnimation(fig, self.plot_classified_perception, interval=50, fargs=(lidar_polar,))
                    self.plot_classified_perception(lidar_polar)
                else:
                    ani = animation.FuncAnimation(fig, self.plot_raw_perception, interval=50, fargs=(lidar_polar,))
                    #self.plot_raw_inputs(lidar_polar)
                plt.show()
            print('Scan interrupted!')
            if self.lidar_online:
                self.laser.turnOff()
        print('Lidar disconnected!')
        if self.lidar_online:
            self.laser.disconnecting()
        plt.close()

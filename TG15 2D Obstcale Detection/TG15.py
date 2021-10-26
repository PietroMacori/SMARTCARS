import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ydlidar import CYdLidar as Laser
import ydlidar
from collections import defaultdict
import numpy as np

import numpy as np
from scipy.spatial import ConvexHull
import time

class TG15(Laser):
    def __init__(self, clustering_max_dist=1.5):
        
        # Max and min scan range
        self.RMAX = 15
        self.RMIN = 0.05
        self.clustering_max_dist = clustering_max_dist

        ports = ydlidar.lidarPortList()
        self.port = "/dev/ydlidar"
        for _, value in ports.items():
            self.port = value

        self.setlidaropt(ydlidar.LidarPropSerialPort, self.port)
        self.setlidaropt(ydlidar.LidarPropSerialBaudrate, 512000)
        self.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
        self.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
        self.setlidaropt(ydlidar.LidarPropSampleRate, 20)
        self.setlidaropt(ydlidar.LidarPropSingleChannel, False)
        self.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
        self.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        self.setlidaropt(ydlidar.LidarPropMaxRange, self.RMAX)
        self.setlidaropt(ydlidar.LidarPropMinRange, self.RMIN)
        
        self.reset()

    def reset(self):
        self.points = {}     # Dictionary {point_idx: (angle, range, intensity)}
        self.blocks = defaultdict(list)     # Dictionary {block_idx: [list of point indices]}
        self.labels = {}     # Dictionary {block_idx: class_label}    
        # circle (person): (0, [x,y], r), line (forward vehicle and barriers): (1, [(x1,y1), (x2,y2)]),
        # rectangle (other vehicles and buildings): (2, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])

        self.num_points = 0  # The total number of points detected
        self.num_blocks = 0  # The total number of blocks detected


    def animate_raw(self, i, lidar_polar):
        r = self.doProcessSimple(self.scan)
        if r:
            angle = []
            ran = []
            intensity = []
            for point in self.scan.points:
                angle.append(point.angle)
                ran.append(point.range)
                intensity.append(point.intensity)
            

    def animate_classified(self, i, lidar_polar):
        pass

    def plot_raw_perception(self):
        fig = plt.figure()
        fig.canvas.set_window_title('TG15 LIDAR Monitor')
        lidar_polar = plt.subplot(polar=True)
        lidar_polar.autoscale_view(True,True,True)
        lidar_polar.set_rmax(self.RMAX)
        lidar_polar.grid(True)
        ret = self.initialize()
        if ret:
            ret = self.turnOn()
            if ret:
                animation.FuncAnimation(fig, self.animate_raw, interval=50, fargs=(lidar_polar,))
                plt.show()
            self.turnOff()
        self.disconnecting()
        plt.close()

    def read_pre_process(self):
        self.reset()
        for point in self.scan.points:
            self.points[self.num_points] = (point.angle, point.range ,point.intensity)
            self.num_points += 1

    def polar_dist(self, pi, pj):
        r1 = self.points[pi][1]
        r2 = self.points[pj][1]
        a1 = self.points[pi][0]
        a2 = self.points[pj][0]
        return np.sqrt(r1**2+r2**2-r1*r2*np.cos(a2-a1))

    def cartesian_dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    
    def minimum_bounding_rectangle(self, points):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :rval: an nx2 matrix of coordinates
        """
        from scipy.ndimage.interpolation import rotate
        pi2 = np.pi/2.

        # get the convex hull for the points
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-pi2),
            np.cos(angles+pi2),
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
        max_dist = -1
        for ind in self.blocks[ib]:
            dist = self.polar_dist(ip, ind)
            if  dist > max_dist:
                max_dist = dist
        if max_dist <= self.clustering_max_dist:
            return True
        return False              

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
            X.append(r*np.cos(a))
            Y.append(r*np.sin(a))
        center = [np.mean(X), np.mean(Y)]
        radius = 0.0
        for i in range(len(bl)):
            d = self.cartesian_dist([X[i], Y[i]], center)
            if d > radius:
                radius = d
        return 0, center, radius

    def clacl_line_rect(self, bi):
        max_dist = 0.0
        p1_line = None
        p2_line = None
        bl = self.blocks[bi]
        for i in range(len(bl)-1):
            for j in range(i+1, len(bl)):
                d = self.polar_dist(bl[i], bl[j])
                if d > max_dist:
                    max_dist = d
                    p1_line = i
                    p2_line = j

        r1_line = self.points[p1_line][1]
        a1_line = self.points[p1_line][0]
        r2_line = self.points[p2_line][1]
        a2_line = self.points[p2_line][0]
        p1 = np.array([r1_line*np.cos(a1_line), r1_line*np.sin(a1_line)])
        p2 = np.array([r2_line*np.cos(a2_line), r2_line*np.sin(a2_line)])

        coords = []
        dists = []
        is_rect = False
        X = []
        Y = []
        bl = self.blocks[bi]
        for i in range(len(bl)):
            r = self.points[bl[i]][1]
            a = self.points[bl[i]][0]
            x = r*np.cos(a)
            y = r*np.sin(a)
            X.append(x)
            Y.append(y)
            p3 = np.array([x,y])
            d = np.abs(np.norm(np.cross(p2-p1, p1-p3)))/np.norm(p2-p1)
            dists.append(d)
            if d>=0.2*max_dist:
                is_rect = True
    
        if is_rect == False:
            coords = [(p1[0], p1[1]), (p2[0], p2[1])]
            return 1, coords

        points = np.array([X, Y]).T
        coords = self.minimum_bounding_rectangle(points)
        return 2, coords
        

    def classify(self):
        # circle (person): (0, [x,y], r), line (forward vehicle and barriers): (1, [(x1,y1), (x2,y2)]),
        # rectangle (other vehicles and buildings): (2, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
        for i in range(self.num_blocks):
            if len(self.blocks[i]) <= 5:
                center, r = self.calc_circle(i)
                self.labels[i] = (0, center, r)
            else:
                label, coords = self.calc_line_rect(i)
                self.labels[i] = (label, coords)

    def plot_classified_perception(self, fig, lidar_polar):
        ret = self.initialize()
        if ret:
            ret = self.turnOn()
            if ret:
                animation.FuncAnimation(fig, self.animate_raw, interval=50, fargs=(lidar_polar,))
                plt.show()
            self.turnOff()
        self.disconnecting()
        plt.close()

    def run(self):
        ret = self.initialize()
        print('Lidar initialized!')
        while ret:
            ret = self.turnOn()
            r = self.doProcessSimple(self.scan)
            print('Scanning...')
            print('Initializing GUI...')
            fig = plt.figure()
            fig.canvas.set_window_title('TG15 LIDAR Monitor')
            lidar_polar = plt.subplot(polar=True)
            lidar_polar.autoscale_view(True,True,True)
            lidar_polar.set_rmax(self.RMAX)
            lidar_polar.grid(True)
            while r:
                self.read_pre_process()
                self.create_blocks()
                self.classify()

                ### Plotting
                #time.sleep(1.0)
                # lidar_polar.clear()
                # lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
                
            print('Scan interrupted! Attempting to restart...')
            self.turnOff()
        print('Lidar disconnected!')
        self.disconnecting()
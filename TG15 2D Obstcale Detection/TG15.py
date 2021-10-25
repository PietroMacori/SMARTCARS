import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ydlidar import CYdLidar as Laser
import ydlidar

class TG15(Laser):
    def __init__(self):
        
        # Max and min scan range
        self.RMAX = 15
        self.RMIN = 0.05

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
        self.blocks = {}     # Dictionary {block_idx: [list of point indices]}
        self.labels = {}     # Dictionary {block_idx: class_label}    
        # circle (person): 0, line (forward vehicle and barriers): 1, rectangle (other vehicles and buildings): 2

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
            lidar_polar.clear()
            lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)

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

    def create_blocks(self):
        #self.blocks[0] = 
        for _, point in self.points:
            pass
            

    def classify(self):
        pass

    def plot_classified_perception(self):
        pass

    def run(self):
        ret = self.initialize()
        print('Lidar initialized!')
        while ret:
            ret = self.turnOn()
            r = self.doProcessSimple(self.scan)
            print('Scanning...')
            while r:
                self.read_pre_process()
                self.create_blocks()
                self.classify()
                print(self.labels)
            print('Scan interrupted! Attempting to restart...')
            self.turnOff()
        print('Lidar disconnected!')
        self.disconnecting()
from TG15 import TG15

def main():
    smartcars_2d_lidar = TG15(clustering_max_dist=1.5, circle_thresh=500, rect_thresh=0.2)
    smartcars_2d_lidar.run()

if __name__ == "__main__":
    main()
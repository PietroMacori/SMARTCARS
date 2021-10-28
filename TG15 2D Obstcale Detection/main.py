from TG15 import TG15

def main():
    smartcars_2d_lidar = TG15(lidar_online=False, to_classify=False, clustering_max_dist=1.5, circle_thresh=5000, rect_thresh=0.2, file_name='myFile')
    smartcars_2d_lidar.run()

if __name__ == "__main__":
    main()
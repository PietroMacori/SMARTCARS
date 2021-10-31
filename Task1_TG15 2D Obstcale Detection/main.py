from TG15 import TG15

def main():
    smartcars_2d_lidar = TG15(lidar_online=False, to_classify=False, clustering_max_dist=0.03, circle_thresh=30, rect_thresh=0.1, file_name='myFile')
    smartcars_2d_lidar.run()

if __name__ == "__main__":
    main()
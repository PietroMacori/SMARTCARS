from TG15 import TG15


def main():
    smartcars_2d_lidar = TG15(lidar_online=True, to_classify=True, clustering_max_dist=0.05, circle_thresh=30,
                              file_name='myFile')
    smartcars_2d_lidar.run()


if __name__ == "__main__":
    main()

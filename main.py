from TG15 import TG15


def main():
    smartcars_2d_lidar = TG15(lidar_online=False,file_name='myFile')
    smartcars_2d_lidar.run()


if __name__ == "__main__":
    main()

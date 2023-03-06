import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from get_user_pos import plotXYZ


def get_max_speed(df):
    return df["SpeedMps"].max()


def average_stop_time(df):
    df = df.sort_values("UnixTimeMillis")
    df = df.dropna()
    speeds = df["SpeedMps"].values
    speeds = speeds < 0.1

    flip = False
    stops = []
    a, b = 0, 0
    while b < len(speeds):
        if speeds[a] == 1:
            if speeds[b] == 1:
                b += 1
            else:
                stops.append((a, b - 1))
                a = b
        else:
            a += 1
            b += 1


    plt.plot(df["UnixTimeMillis"], df["SpeedMps"])
    for a,b in stops:
        plt.scatter(df["UnixTimeMillis"].iloc[a], df["SpeedMps"].iloc[a])
        plt.scatter(df["UnixTimeMillis"].iloc[b], df["SpeedMps"].iloc[b])

        print(df[["UnixTimeMillis", "SpeedMps"]].iloc[a-1:b+1])
    plt.show()


def pre_process_files(filePaths, transittype):
    results = {}

    max_speeds = []
    # Data from file
    for filePath in filePaths:
        Fix_df = pd.read_csv(filePath)
        average_stop_time(Fix_df)
        max_speeds.append(get_max_speed(Fix_df))

    results["max_speeds"] = max_speeds
    return results


def crawl():
    #go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_Logger_Fix")
    bike = pre_process_files(glob.glob(os.path.join(main_dir, "Bike/", "*.csv")), "Bike")    
    car = pre_process_files(glob.glob(os.path.join(main_dir, "Car/", "*.csv")), "Car")
    walk = pre_process_files(glob.glob(os.path.join(main_dir, "Walk/", "*.csv")), "Walk")
    # bus = pre_process_files(glob.glob(os.path.join(main_dir, "Bus/", "*.csv")), "Bus")

if __name__ == '__main__':
    crawl()
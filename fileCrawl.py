import os
import glob
import pandas as pd
import get_nasa_ephem as nasa
from datetime import datetime,timezone


def parseFile(filepath, transittype="n/a"):
    file1 = open(filepath, 'r')

    data = []
    header = ""
    for line in file1:
        if line[:5] == "# Raw":
            header = line[2:-1].split(",") + ["TransitType"]
        if line[:3] == "Raw":
            curRow = line[0:-1].split(",") + [transittype]
            data.append(curRow)

    df = pd.DataFrame(data)
    df.columns = header


    return df


def crawl():
    bike_list = []
    car_list = []
    walk_list = []
    bike_date_list = []
    car_date_list = []
    walk_date_list = []
    bike_ephem = []
    car_ephem=[]
    walk_ephem=[]



    #go over each text file in directory
    for filePath in glob.glob(os.path.join("Bike/", "*.txt")):
        output = parseFile(filePath, "Bike")
        file_name = os.path.splitext(filePath)[0]
        file_name = file_name.split("_")
        date_time = file_name[2:8]
        date_time = list(map(int,date_time))
        date_time = datetime(*date_time,tzinfo=timezone.utc)
        output['nasaSvid'] =  output.apply(lambda row: nasa.svid_constnum_2_nasa_svid(row), axis=1)

        bike_date_list.append(date_time)
        bike_list.append(output)
        # print(output)

    for filePath in glob.glob(os.path.join("Car/", "*.txt")):
        output1 = parseFile(filePath, "Car")
        file_name = os.path.splitext(filePath)[0]
        file_name = file_name.split("_")
        date_time = file_name[2:8]
        date_time = list(map(int,date_time))
        date_time = datetime(*date_time,tzinfo=timezone.utc)
        output1['nasaSvid'] =  output1.apply(lambda row: nasa.svid_constnum_2_nasa_svid(row), axis=1)

        car_date_list.append(date_time)
        car_list.append(output1)
        # print(output1)

    for filePath in glob.glob(os.path.join("Walk/", "*.txt")):
        output2 = parseFile(filePath, "Walk")
        file_name = os.path.splitext(filePath)[0]
        file_name = file_name.split("_")
        date_time = file_name[2:8]
        date_time = list(map(int,date_time))
        date_time = datetime(*date_time,tzinfo=timezone.utc)
        output2['nasaSvid'] =  output2.apply(lambda row: nasa.svid_constnum_2_nasa_svid(row), axis=1)

        walk_date_list.append(date_time)
        walk_list.append(output2)
        # print(output2)



    # print(bike_list)
    # print(car_list)
    # print(walk_list)
    # print(bike_date_list)
    # print(car_date_list)
    # print(walk_date_list)
    # print(bike_date_list[0])
    print(bike_list[0]['nasaSvid'].unique())
    print(nasa.get_nasa_ephem(bike_date_list[0],bike_list[0]['nasaSvid'].unique()))


if __name__ == '__main__':
    crawl()
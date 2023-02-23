import os
import glob
import pandas as pd


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


def main():
    bike_list = []
    car_list = []
    walk_list = []



    #go over each text file in directory
    for filePath in glob.glob(os.path.join("Bike/", "*.txt")):
        output = parseFile(filePath, "Bike")

        #do something with output
        bike_list.append(output)
        # print(output)

    for filePath in glob.glob(os.path.join("Car/", "*.txt")):
        output1 = parseFile(filePath, "Car")

        #do something with output
        car_list.append(output1)
        # print(output1)

    for filePath in glob.glob(os.path.join("Walk/", "*.txt")):
        output2 = parseFile(filePath, "Walk")

        #do something with output
        walk_list.append(output2)
        # print(output2)

    print(bike_list)
    print(car_list)
    print(walk_list)

if __name__ == '__main__':
    main()
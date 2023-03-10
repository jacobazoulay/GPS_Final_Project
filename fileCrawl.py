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


def parseFileFix(filepath, transittype="n/a"):
    file1 = open(filepath, 'r')

    data = []
    header = ""
    for line in file1:
        if line[:5] == "# Fix":
            header = line[2:-1].split(",") + ["TransitType"]
        if line[:3] == "Fix":
            curRow = line[0:-1].split(",") + [transittype]
            data.append(curRow)

    df = pd.DataFrame(data)
    df.columns = header

    return df


def pre_process_files_Fix(filePaths, transittype):
    results = {}

    # Data from file
    for filePath in filePaths:
        # Data
        output = parseFileFix(filePath, transittype)
        transit_types = output["TransitType"]

        output = output.apply(pd.to_numeric, errors='ignore')
        output["TransitType"] = transit_types

        # Date time
        file_name_full = os.path.basename(filePath).split('/')[0]
        outpath = filePath.replace("Data", "Fix")[:-4]

        results[file_name_full.split('.')[0]] = output
        output.to_csv(outpath[:-4] + '.csv', index=False)

    return results


def pre_process_files(filePaths, transittype):
    results = {}
    
    # Data from file
    for filePath in filePaths:
        # Data
        output = parseFile(filePath, transittype)
        transit_types = output["TransitType"]

        needed_cols = ['Svid', 'ReceivedSvTimeNanos',"FullBiasNanos", "TimeNanos", "TimeOffsetNanos", "BiasNanos","ConstellationType"]
        output = output[output.columns.intersection(needed_cols)]
        output = output.apply(pd.to_numeric)
        output["TransitType"] = transit_types
        
        # Date time
        file_name_full = os.path.basename(filePath).split('/')[0]
        file_name = file_name_full.split("_")
        date_time = file_name[2:7]
        date_time = list(map(int,date_time))
        date_time = datetime(*date_time,tzinfo=timezone.utc)
        
        # SVID
        output = output[output["ConstellationType"] != 6] # galileo doesn't work :(
        output['Svid'] =  output.apply(lambda row: nasa.svid_constnum_2_nasa_svid(row), axis=1)
        output.drop("ConstellationType", axis=1, inplace=True)
        
        # Ephemeris
        ephem = nasa.get_nasa_ephem(date_time, output['Svid'].unique().tolist())
        ephem["Svid"] = list(ephem.index)
        results[file_name_full.split('.')[0]] = {"data": output, "ephemeris": ephem}
          
    return results


def crawl():
    #go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_Logger_Data")
    bike = pre_process_files(glob.glob(os.path.join(main_dir, "Bike/", "*.txt")), "Bike")    
    car = pre_process_files(glob.glob(os.path.join(main_dir, "Car/", "*.txt")), "Car")
    walk = pre_process_files(glob.glob(os.path.join(main_dir, "Walk/", "*.txt")), "Walk")
    bus = pre_process_files(glob.glob(os.path.join(main_dir, "Bus/", "*.txt")), "Bus")

    return {"Bike": bike, "Car": car, "Walk": walk, "Bus": bus}


def crawlFix():
    # go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_Logger_Data")
    bike = pre_process_files_Fix(glob.glob(os.path.join(main_dir, "Bike/", "*.txt")), "Bike")
    car = pre_process_files_Fix(glob.glob(os.path.join(main_dir, "Car/", "*.txt")), "Car")
    walk = pre_process_files_Fix(glob.glob(os.path.join(main_dir, "Walk/", "*.txt")), "Walk")
    bus = pre_process_files_Fix(glob.glob(os.path.join(main_dir, "Bus/", "*.txt")), "Bus")

    return {"Bike": bike, "Car": car, "Walk": walk, "Bus": bus}


if __name__ == '__main__':
    # crawl()
    crawlFix()
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


if __name__ == "__main__":
    df = parseFile("gnss_log_2023_02_15_16_25_25.txt", "Bike")
    print(df.head())

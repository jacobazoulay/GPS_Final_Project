import pandas as pd


def parseFile(filepath):
    file1 = open(filepath, 'r')

    data = []
    header = ""
    for line in file1:
        if line[:5] == "# Raw":
            header = line[2:-1].split(",")
        if line[:3] == "Raw":
            data.append(line[0:-1].split(","))

    df = pd.DataFrame(data)
    df.columns = header

    return df


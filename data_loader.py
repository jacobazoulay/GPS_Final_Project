import pandas as pd

file1 = open('gnss_log_2023_02_15_16_25_25.txt', 'r')

data = []
for line in file1:
    if line[:3] == "Raw":
        data.append(line)

df = pd.DataFrame(data)

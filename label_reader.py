import os
import numpy as np
import csv
from tabula import read_pdf
from tabulate import tabulate

with open('data/spectra.csv', mode ='r') as file:
    csv_data = csv.reader(file)
    order = []
    first = True
    for line in csv_data:
        if first:
            first = False
            continue
        order.append(line[0])

df = read_pdf("data/spectra_info.pdf", pages="all")
z_dict = {}
for table in df:
    for i in range(1, table.shape[0]):
        row = table.iloc[i]
        name = row["Unnamed: 0"]
        z = row["Classification Results"].split()[1]
        if z == "...":
            z = row["Best-Matching SNID Template"].split()[2]
        z = float(z)
        print(name+" "+str(z))
        z_dict[name] = z

labels = np.array([0. for i in range(len(order) - 1)])
sind = 0
for i in range(len(order)):
    if order[i] not in z_dict.keys():
        print(order[i]+" no z-data available, index: "+str(i))
        continue
    labels[sind] = z_dict[order[i]]
    sind += 1

print(len(labels))
print(labels)
np.save("processed/labeldata", labels)

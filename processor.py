import os
import numpy as np
import csv

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def read_file(filename):
    with open(filename, "r") as f:
        raw_lines = f.readlines()
    processed_lines = [line.strip() for line in raw_lines]
    processed_lines = [line.strip("#") for line in processed_lines]
    data = [line.split() for line in processed_lines]
    while len(data[0]) == 0 or not isfloat(data[0][0]):
        data = data[1:]
    while len(data[-1]) == 0:
        data = data[:-1]
    if len(data[0]) > 2:
        data = np.delete(data,2,1)
    data = np.array(data)
    data = data.astype('float32')
    return data

def convert_to_vector(data):
    # Consider angstrom range 3500-10000
    marked = np.array([-1 for i in range(int((10000-3500)/2+1))])
    wavelengths = np.array([0. for i in range(int((10000-3500)/2+1))])
    for item in data:
        ind = int((int(item[0])-3500)/2)
        if (ind < 0 or ind >= len(wavelengths)):
            continue
        wavelengths[ind] = item[1]
        marked[ind] = 1
    wavelengths -= wavelengths.mean()
    wavelengths /= wavelengths.std()
    for i in range(len(wavelengths)):
        if marked[i] == -1:
            wavelengths[i] = 0
    return wavelengths

with open('data/spectra.csv', mode ='r') as file:
    csv_data = csv.reader(file)
    order = []
    first = True
    for line in csv_data:
        if first:
            first = False
            continue
        order.append(line[1])

df = []
for i in range(len(order)):
    filename = order[i]
    if i == 347:
        print(filename + " skipped as label not available")
        continue
    print(filename)
    data = read_file("data/spectra/"+filename)
    vectorized_data = convert_to_vector(data)
    df.append(vectorized_data)
df = np.array(df)
print(len(df))
print(df)
np.save('processed/spectradata', df)

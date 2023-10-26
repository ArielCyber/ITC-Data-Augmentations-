#!/usr/bin/env python
"""
Read traffic_csv
"""

import os
import csv

import matplotlib.pyplot as plt

from sessions_plotter import *
import glob
import re

"""
input:
first 7 columns- are not used

(first column = 0)
 column 7: the length of the flow, number of packets. (let n = length)
 column 8 - 8 + n: relative time of arrival (column 8 will always be 0's)
 column 8 + n: empty 
 column 9 + n - 9 + n + n: sizes respectivley to time 
 column 9 + n + n  - end: empty

"""
CLASSES_DIR = ""


FlowPic = False  # True - create FlowPic , False - create miniFlowPic
if FlowPic:
    TPS = 60  # TimePerSession in secs
    DELTA_T = 60  # Delta T between splitted sessions
    MIN_TPS = 50
    MIN_LENGTH = 10
    IMAGE_SIZE = 1500
else:
    TPS = 15  # TimePerSession in secs
    DELTA_T = 15  # Delta T between splitted sessions
    MIN_TPS = 0
    MIN_LENGTH = 100
    IMAGE_SIZE = 32
DEBUG = False
FIRST_15 = True

def export_class_dataset(dataset, class_dir, name=None):
    print("Start export dataset")
    if name is None:
        name = class_dir + "/" + "_".join(re.findall(r"[\w']+", class_dir)[-2:])
    np.save(name, dataset)
    print(dataset.shape)


def traffic_csv_converter(file_path):
    global failed_count
    global counter_dir = {}
    print("Running on " + file_path)
    dataset = []
    if FIRST_15:
        dataset_first_15 = []
    counter = 0
    total = 0
    less_t_15 = 0
    less_t_2 = 0
    less_t_100_p = 0
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for _ in range(TIMES):
            for i, row in enumerate(reader):
                total += 1
                length = int(float(row[7]))
                try:
                    ts = np.array(row[8:8 + length-1], dtype=float)
                    sizes = np.array(row[9 + length:9 + length + length-1], dtype=float)
                    sizes.astype(int)
                    if max(ts) < 2:
                        less_t_2 += 1
                    if max(ts) < 15:
                        less_t_15 += 1
                        continue
                    if length <= MIN_LENGTH:
                        less_t_100_p += 1
                except:
                    continue
                if DEBUG:
                    print("max ts:" , max(ts))
                    plt.scatter(x=ts, y=sizes)
                    plt.show()
                

                if length > MIN_LENGTH:
                    for t in range(int(ts[-1] / DELTA_T - TPS / DELTA_T) + 1):
                        mask = ((ts >= t * DELTA_T) & (ts <= (t * DELTA_T + TPS)))
                        ts_mask = ts[mask]
                        sizes_mask = sizes[mask]
                        if DEBUG:
                            print("mask length =", len(ts_mask), "range =", ts_mask[-1] - ts_mask[0])
                        if len(ts_mask) > MIN_LENGTH and ts_mask[-1] - ts_mask[0] >= MIN_TPS:
                            h = session_2d_histogram(ts_mask, sizes_mask, DEBUG)
                            dataset.append([h])
                            if FIRST_15 and t == 0:
                                dataset_first_15.append([h])
                            counter += 1
                            if counter % 100 == 0:
                                pass
                                
    if FIRST_15:
        class_dir = file_path.split('/')
        label = class_dir[-1]
        if not label in counter_dir.keys():
            counter_dir[label] = 0
        class_dir = '/'.join(class_dir[:-1]) + '/'
        counter_file = 0
        for key, item in counter_dir.items():
            if key in str(file_path):
                counter = item
                counter_dir[key] += 1
                counter_file = counter_dir[key]
        name = class_dir + "/" + "_".join(re.findall(r"[\w']+", class_dir)[-2:]) + "_first_15" +"_"+ str(counter_file)
        dataset_tuple = np.asarray(dataset_first_15)
        dataset_tuple = (np.asarray(dataset_first_15),)
        dataset_first_15 = np.concatenate(dataset_tuple, axis=0)
        print(dataset_first_15.shape)
        export_class_dataset(dataset_first_15, class_dir, name)
    return np.asarray(dataset)


def traffic_class_converter(dir_path):
    dataset_tuple = ()
    for file_path in [os.path.join(dir_path, fn) for fn in next(os.walk(dir_path))[2] if
                      (".csv" in os.path.splitext(fn)[-1])]:
        print("working on:", file_path)
        dataset_tuple = (traffic_csv_converter(file_path),)
    return np.concatenate(dataset_tuple, axis=0)


def iterate_all_classes():
    for class_dir in glob.glob(CLASSES_DIR):
        print("working on " + class_dir)
        dataset = traffic_class_converter(class_dir)


if __name__ == '__main__':
    iterate_all_classes()

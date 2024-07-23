import numpy as np
from pathlib import Path
import os


def get_label(file_path, classes):
    for i, _class in enumerate(classes):
        if _class in file_path:
            return i


def iterate_all_classes(data_dir, classes, p, data_max_size):
    for _dir in Path(data_dir).glob('*/'):
        # check if dir is folder. if not, continue
        if not os.path.isdir(_dir):
            continue
        _dir = str(_dir)
        print("\nworking on " + _dir)
        data = []
        label = classes[get_label(_dir, classes)]
        files_to_remove = []
        for file_name in Path(_dir).glob('*first_15_[0-9].npy'):
            # print(f"*********************************{file_name}")
            files_to_remove.append(file_name)
            file_name = str(file_name)
            print("working on:", file_name)
            data.append(np.load(file_name))
        # for file_name in files_to_remove:
        #     file_name.unlink()
        data = np.concatenate(data)
        print("data shape:", data.shape)
        size = data_max_size if data_max_size != -1 else data.shape[0]
        index = np.random.choice(range(len(data)), size = size, replace = False)
        data = data[index]
        split = int(data.shape[0] * p)
        train = data[:split]
        test = data[split:]
        print("train data shape", train.shape)
        print("test data shape", test.shape)
        print("Exported to:", data_dir+label +'/' +label +f'_first_15.npy')
        np.save(data_dir+label +'/' +label +f'_first_15.npy', train)
        print("Exported to:", data_dir+label +'/' +label +f'_first_15_test.npy')
        np.save(data_dir+label +'/' +label +f'_first_15_test.npy', test)
        del data
        del train
        del test
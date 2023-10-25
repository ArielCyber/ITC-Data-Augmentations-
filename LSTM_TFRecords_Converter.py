import numpy as np
from pathlib import Path
from LSTM_TFRecords_Utils import *
from TFRecords_Converter import get_label


def iterate_all_classes(data_dir, mask, label, max_len, split, classes):
    masks = np.load(mask, allow_pickle= True)
    for _dir in Path(data_dir).glob('*/'):
        for file_name in Path(_dir).glob('*first_15.npy'):
            file_name = str(file_name)
            print("\nImporting Data from:", file_name)
            if label:
                convert_to_lstm_tfrecord_label(file_name,masks[get_label(file_name, classes)] , get_label(file_name, classes), max_len, split)
            else:
                convert_to_lstm_tfrecord(file_name,masks[get_label(file_name, classes)] , get_label(file_name, classes), max_len, split) 
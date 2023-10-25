from pathlib import Path
from TFRecords_Utils import *


def get_label(file_path, classes):
    for i, _class in enumerate(classes):
        if _class in file_path:
            return i
    
def iterate_all_classes(data_dir, mask, classes, files_re= '*_first_15.npy'):
    masks = np.load(mask, allow_pickle= True)
    for _dir in Path(data_dir).glob('*/'):
        for file_name in Path(_dir).glob(files_re):
            print("\nImporting Data from:", file_name)
            label = get_label(str(file_name), classes)
            convert_to_tfrecord(file_name, label, masks[label])
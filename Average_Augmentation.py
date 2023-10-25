import numpy as np
import itertools as it
from pathlib import Path
from Train_Validation_Split import train_validation_split
from TFRecords_Converter import get_label


def average_images(images, n):
    out = []
    for combination in it.combinations(images, n):
        img = combination[0][0]
        for i in range(1, n):
            img += combination[i][0]
        img //= n
        out.append([img])
    return np.array(out)

def image_augmentor( file_path, mask, n):
    data = train_validation_split(file_path, mask, 1)
    out = [average_images(data['train'], n), average_images(data['val'], n)]
    train_length = out[0].shape[0]
    val_length = out[1].shape[0]
    choice = np.arange(start = train_length, stop = val_length + train_length )
    print("validation size :", val_length)
    print("train size :", train_length)
    mask = np.full(train_length + val_length, False)
    mask[choice] = True
    out = np.concatenate(out)
    file_name_new = file_path[:-4] + '_avg_' + str(n) + '.npy'
    np.save(file_name_new, out)
    print("Exported To:", file_name_new)
    return mask
    

def iterate_all_classes(data_dir, mask, classes, n = 2):
    masks = np.load(mask, allow_pickle= True)
    masks_new = []
    for _dir in Path(data_dir).glob('*/'):
        _dir = str(_dir)
        print("\nworking on: " + _dir)
        for file_name in Path(_dir).glob('*first_15.npy'):
            file_name = str(file_name)
            print("working on file:", file_name)
            label = get_label(str(file_name), classes)
            masks_new.append(image_augmentor(file_name, masks[label], n))
    masks_new = np.array(masks_new, dtype=object)
    output_file = data_dir + "val_avg_first_15_masks.npy"
    np.save(output_file, np.array(masks_new))
    return output_file

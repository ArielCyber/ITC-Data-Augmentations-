import numpy as np
from pathlib import Path

def create_mask(file_path, p = 0.20):
    data = np.load(file_path)
    shape = data.shape
    print("data shape:", shape)
    choice = np.random.choice(shape[0], int(shape[0]*p), replace = False)
    print("validation data size:", len(choice))
    print("train data size:", len(data) - len(choice))
    mask = np.full(shape[0], False)
    mask[choice] = True
    return mask
    

def train_validation_split(file_path, mask, times):
    images = np.load(file_path)
    mask = np.concatenate([mask for _ in range(times)])
    mask = mask.astype(bool)
    train = images[np.logical_not(mask)]
    val = images[mask]
    print("train size:", train.shape[0], "| val size:", val.shape[0])
    print("train %:", train.shape[0]/images.shape[0], "| val %:", val.shape[0]/images.shape[0])
    return {'train': train, 'val': val}


def iterate_all_classes(data_dir, p):
    masks = []
    for _dir in Path(data_dir).glob('*/'):
        _dir = str(_dir)
        print("\nworking on " + str(_dir))
        for file_name in Path(_dir).glob('*first_15.npy'):
            file_name = str(file_name)
            print("working on:", file_name)
            masks.append(create_mask(file_name,p))
    output_file = data_dir + "val_first_15_masks.npy"
    print("Exported to:", output_file)
    masks = np.array(masks, dtype=object)
    masks = np.save(output_file, masks, allow_pickle= True)
    return output_file
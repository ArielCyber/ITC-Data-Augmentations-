from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path


def packet_loss(ts, sizes):
    t = np.random.choice(ts, 1)
    sizes[(ts > (t-0.1)) & (ts < (t+0.1))] = -10
    return ts, sizes


def image_augmentor(file_path, th_min, th_max, loop = 1):
    data = np.load(file_path)
    out = []
    for _ in range(loop):
        for img in data:
            img_size = img[0].shape[0]
            ts, size = image_to_ts_and_size(img[0],img_size)
            ts, size = packet_loss(ts, size)
            packet_loss_img = session_2d_histogram(ts, size, img_size)
            out.append([packet_loss_img])   
    out = np.array(out)
    file_name_new = file_path[:-4] + '_packet_loss.npy'
    np.save(file_name_new, out)
    print("Exported To:", file_name_new)


def image_to_ts_and_size(img, image_size):
    ts = []
    size = []
    for t in range(img.shape[1]):
        for s in range(img.shape[0]):
            ts += [t  for _ in range(img[s, t])]
            size += [s for _ in range(img[s, t])]
    ts = (np.array(ts) / 32) * 15
    return ts , np.array(size)


def session_2d_histogram(ts, sizes, image_size, plot=False):
    max_delta_time = 15
    ts_norm = ((np.array(ts)) / max_delta_time) * image_size
    size_norm = np.array(sizes)
    H, x_edges, y_edges = np.histogram2d(size_norm, ts_norm, bins=(range(0, image_size + 1), range(0, image_size + 1)))

    if plot:
        plt.pcolormesh(x_edges, y_edges, H,cmap= 'binary')
        plt.colorbar()
        plt.xlim(0, image_size)
        plt.ylim(0, image_size)
        plt.show()
    return H.astype(np.uint16)

def iterate_all_classes(data_dir, th_min, th_max, files_re = '*first_15.npy'):
    for _dir in Path(data_dir).glob('*/'):
        _dir = str(_dir)
        print("\nworking on: " + _dir)
        for file_name in Path(_dir).glob(files_re):
            file_name = str(file_name)
            print("working on file:", file_name)
            image_augmentor(file_name, th_min, th_max)
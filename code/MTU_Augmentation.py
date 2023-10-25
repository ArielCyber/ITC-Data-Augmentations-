import numpy as np
from pathlib import Path

def MTU(img, th_min, th_max):
    th = np.random.randint(th_min, th_max, 1)[0]
    th = int((th/1500)*32)-1
    img = np.array(img)
    out = img.copy()
    out[th] += np.sum(img[th:], axis= 0)
    out[:32-th] += img[th:]
    out[th + 1:] = 0
    return out

def image_augmentor(file_path, th_min, th_max, loop = 1):
    data = np.load(file_path)
    out = []
    for _ in range(loop):
        for img in data:
            out.append([np.array(MTU(img[0], th_min, th_max))])   
    out = np.array(out)
    file_name_new = file_path[:-4] + '_mtu.npy'
    np.save(file_name_new, out)
    print("Exported To:", file_name_new)
    

def iterate_all_classes(data_dir, th_min, th_max, files_re = '*first_15.npy'):
    for _dir in Path(data_dir).glob('*/'):
        _dir = str(_dir)
        print("\nworking on: " + _dir)
        for file_name in Path(_dir).glob(files_re):
            file_name = str(file_name)
            print("working on file:", file_name)
            image_augmentor(file_name, th_min, th_max)


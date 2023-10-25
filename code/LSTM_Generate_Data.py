import tensorflow as tf
import keras
import glob
from LSTM_TFRecords_Utils import *
from TFRecords_Converter import get_label
from sklearn.metrics import classification_report
from Train_Validation_Split import train_validation_split
from LSTM import ssim_loss, classifier_loss


def detranspose_images(imgs_1, imgs_2):
    images = np.concatenate([imgs_1, imgs_2], axis=1)
    out = []
    for img in images:
        out.append(img.T)
    return np.array(out)

def generate_data(data_dir, masks_name, max_len, split, classes):
    # masks = np.load(masks_name, allow_pickle= True)
    # masks = masks.astype(bool)
    filenames_test = glob.glob(data_dir + '**/*first_15.npy')

    model_name = './models/lstm_split_'+ str(split) + "_max_len_" + str(max_len)
    print("\nloading model:", model_name,'\n')
    model = keras.models.load_model(model_name , custom_objects= {"ssim_loss": ssim_loss, "classifier_loss": classifier_loss})

    for _path in filenames_test:
        print("\nimporting data from:", str(_path))
        # label = get_label(_path, classes)
        # dataset = train_validation_split(_path, masks[label], 1)
        # temp = np.zeros_like(np.concatenate([dataset['train'],dataset['val']]))
        # temp[masks[label]] = dataset['val']
        # temp[np.logical_not(masks[label])] = dataset['train']
        # dataset = temp
        data = np.load(_path)
        
        transpose = transpose_images(data)

        imgs_f_half = transpose[:,:split , :]

        imgs_s_half_pred= model.predict(imgs_f_half)
            
        out_data = detranspose_images(imgs_f_half, imgs_s_half_pred)
        out_data = out_data.reshape(out_data.shape[0], 1 , out_data.shape[1], out_data.shape[2])
        
        output_file_name =  str(_path)[:-4] + '_' + model_name.split('/')[-1] + '.npy'
        print('created_file:',output_file_name)
        np.save(output_file_name, out_data)
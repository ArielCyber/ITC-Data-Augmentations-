import numpy as np
import tensorflow as tf
import keras
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")


from Main import reset_keras
from LSTM_TFRecords_Utils import *
from TFRecords_Converter import get_label
from sklearn.metrics import confusion_matrix, classification_report
from Train_Validation_Split import train_validation_split
from LSTM import ssim_loss, classifier_loss


def detranspose_images(imgs_1, imgs_2):
    images = np.concatenate([imgs_1, imgs_2], axis=1)
    out = []
    for img in images:
        out.append(img.T)
    return np.array(out)

def main():
    # set code arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('augmentation', choices = ['lstm', 'average', 'mtu'])
    parser.add_argument('--split', type=int, default= 16, required = False)
    parser.add_argument('--max_len', type=int, default= 32, required = False)
    parser.add_argument('--avg_n', type=int, default= 2, required = False)

    args = parser.parse_args()
    # fix data directory path 
    if args.data_dir[-1] != '/':
        args.data_dir += '/'
    
    #list all classes
    classes = glob.glob(args.data_dir + "*/")
    classes = [_dir[len(args.data_dir):-1] for _dir in classes ]
    print("\nclasses:", classes)

    if args.augmentation == 'lstm':
        print("\n---------- Running LSTM Tests ----------")
        filenames_test = glob.glob(args.data_dir + '**/*first_15_test.npy')

        model_name = 'lstm_split_'+ str(args.split) + "_max_len_" + str(args.max_len)
        print("\nloading model:", model_name,'\n')
        lstm_model = keras.models.load_model('./models/'+ model_name , custom_objects= {"ssim_loss": ssim_loss, "classifier_loss": classifier_loss})
        labels = []
        data_1 = []
        data_2 = []
        data_3 = []
        for _path in filenames_test:
            print("\nloading data:", _path)
            label = get_label(_path, classes)
            data = np.load(_path)
            labels.append(np.full(data.shape[0], label))
            data_1.append(data)

            transpose = transpose_images(data)
            imgs_f_half = transpose[:,:args.split , :]

            data_2.append(imgs_f_half)

            imgs_s_half_pred= lstm_model.predict(imgs_f_half)
            data_3.append(detranspose_images(imgs_f_half, imgs_s_half_pred))
        
        reset_keras(lstm_model)
        y_true = np.concatenate(labels)

        X_1 = np.concatenate(data_1)
        X_1 = X_1.reshape(X_1.shape[0], 32, 32, 1)

        X_2 = np.concatenate(data_2)
        X_2 = X_2.reshape(X_2.shape[0], args.split, 32)

        X_3 = np.concatenate(data_3)
        X_3 = X_3.reshape(X_3.shape[0], 32, 32, 1)

        print("\nloading model: classifier\n")
        model_1 = keras.models.load_model("./models/classifier")

        print("\nloading model: classifier_lstm_generated\n")
        model_2 = keras.models.load_model("./models/classifier_lstm_generated")

        print(f"\nloading model: lstm_classifier_split_0_max_len_{args.split}\n")
        model_3 = keras.models.load_model(f"./models/lstm_classifier_split_0_max_len_{args.split}")


        print("\n---------- Running Test 1: ----------")
        print(  "---------- The contribution of generated data as augmentation technique. ----------")

        y_pred_1 = model_1.predict(X_1)
        y_pred_1 =np.argmax(y_pred_1,axis=1)

        y_pred_2 = model_2.predict(X_1)
        y_pred_2 =np.argmax(y_pred_2,axis=1)

        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier_lstm_generated performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))




        print("\n---------- Running Test 2: ----------")
        print(  "----------  The system’s utility of reducing classification time.         ----------")
        
        y_pred_1 = model_3.predict(X_2)
        y_pred_1 =np.argmax(y_pred_1,axis=1)

        y_pred_2 = model_1.predict(X_1)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(f"\n\nmodel: lstm_classifier_split_0_max_len_{args.split} performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))




        print("\n---------- Running Test 3: ----------")
        print(  "----------  The combination of augmentation and classification time reduction. ----------")


        y_pred_1 = model_1.predict(X_3)
        y_pred_1 =np.argmax(y_pred_1,axis=1)

        y_pred_2 = model_2.predict(X_3)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier_lstm_generated performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

    elif args.augmentation == 'average':
        print("\n---------- Running Average Augmentation Tests ----------")
        filenames_test = glob.glob(args.data_dir + '**/*first_15_test.npy')
        labels = []
        data_1 = []
        for _path in filenames_test:
            print("\nloading data:", _path)
            label = get_label(_path, classes)
            data = np.load(_path)
            labels.append(np.full(data.shape[0], label))
            data_1.append(data)

        y_true = np.concatenate(labels)
        X = np.concatenate(data_1)
        X = X.reshape(X.shape[0], 32, 32, 1)

        print("\nloading model: classifier\n")
        model_1 = keras.models.load_model("./models/classifier")

        print("\nloading model: classifier_avg_ft\n")
        model_2 = keras.models.load_model("./models/classifier_avg_ft")

        print("\n---------- Running Test 1: ----------")
        print(  "----------  The contribution of generated data as augmentation technique. ----------")
        

        y_pred_1 = model_1.predict(X)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        y_pred_2 = model_2.predict(X)
        y_pred_2 =np.argmax(y_pred_2,axis=1)

        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier_avg_ft performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

    elif args.augmentation == 'mtu':
        print("\n---------- Running MTU Augmentation Tests ----------")
        filenames_test = glob.glob(args.data_dir + '**/*first_15_test.npy')
        filenames_test_mtu = glob.glob(args.data_dir + '**/*first_15_test_mtu.npy')
        labels = []
        data_1 = []
        data_2 = []
        for _path in filenames_test:
            print("\nloading data:", _path)
            label = get_label(_path, classes)
            data = np.load(_path)
            labels.append(np.full(data.shape[0], label))
            data_1.append(data)
        
        for _path in filenames_test_mtu:
            print("\nloading data:", _path)
            data = np.load(_path)
            data_2.append(data)
        
        y_true = np.concatenate(labels)
        
        X_1 = np.concatenate(data_1)
        X_1 = X_1.reshape(X_1.shape[0], 32, 32, 1)
        

        X_2 = np.concatenate(data_2)
        X_2 = X_2.reshape(X_2.shape[0], 32, 32, 1)
        

        print("\nloading model: classifier\n")
        model_1 = keras.models.load_model("./models/classifier")

        print("\nloading model: classifier_mtu\n")
        model_2 = keras.models.load_model("./models/classifier_mtu")

        print("\n----------  Running Test 1: ----------")
        print(  "---------- A comparison to show the performance drop when the MTU is less than 1500. ----------")

        y_pred_1 = model_1.predict(X_1)
        y_pred_1 =np.argmax(y_pred_1,axis=1)

        y_pred_2 = model_1.predict(X_2)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

        print("\n---------- Running Test 2: ----------")
        print(  "---------- A comparison to show the performance drop when MTU augmentation data is added to the train dataset. ----------")
        
        y_pred_1 = model_1.predict(X_1)
        y_pred_1 =np.argmax(y_pred_1,axis=1)

        y_pred_2 = model_2.predict(X_1)
        y_pred_2 =np.argmax(y_pred_2,axis=1)

        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier_mtu performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

        print("\n---------- Running Test 3: ----------")
        print(  "---------- A comparison to show the performance improvementwhen MTU augmentation data is added to the train dataset. ----------")

        y_pred_1 = model_1.predict(X_2)
        y_pred_1 =np.argmax(y_pred_1,axis=1)

        y_pred_2 = model_2.predict(X_2)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print("\n\nmodel: classifier performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\n\nmodel: classifier_mtu performance")
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

if __name__ == "__main__":
    main()
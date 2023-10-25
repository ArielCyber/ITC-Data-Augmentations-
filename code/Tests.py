import numpy as np
import tensorflow as tf
import keras
import glob
import argparse

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

        model_name = './models/lstm_split_'+ str(args.split) + "_max_len_" + str(args.max_len)
        print("\nloading model:", model_name,'\n')
        lstm_model = keras.models.load_model(model_name , custom_objects= {"ssim_loss": ssim_loss, "classifier_loss": classifier_loss})
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

        print("\n---------- Running Test 1: ----------")
        print(  "----------  The contribution of generated data as augmentation technique. ----------")
        X = np.concatenate(data_1)
        X = X.reshape(X.shape[0], 32, 32, 1)

        print("\nloading model: classifier\n")
        model_1 = keras.models.load_model("./models/classifier")

        y_pred_1 = model_1.predict(X)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\nloading model: ./models/classifier_lstm_generated\n")
        model_2 = keras.models.load_model("./models/classifier_lstm_generated")
        y_pred_2 = model_2.predict(X)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

        print("\n---------- Running Test 2: ----------")
        print(  "----------  The system’s utility of reducing classification time.         ----------")
        X = np.concatenate(data_2)
        X = X.reshape(X.shape[0], args.split, 32)

        print(f"\nloading model: ./models/lstm_classifier_split_0_max_len_{args.split}\n")
        model_3 = keras.models.load_model(f"./models/lstm_classifier_split_0_max_len_{args.split}")
        y_pred_1 = model_3.predict(X)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        y_pred_2 = model_1.predict(X)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

        print("\n---------- Running Test 3: ----------")
        print(  "----------  The combination of augmentation and classification time reduction. ----------")
        X = np.concatenate(data_3)
        X = X.reshape(X.shape[0], 32, 32, 1)

        y_pred_1 = model_1.predict(X)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        y_pred_2 = model_2.predict(X)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

    if args.augmentation == 'average':
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

        print("\n---------- Running Test 1: ----------")
        print(  "----------  The contribution of generated data as augmentation technique. ----------")
        X = np.concatenate(data_1)
        X = X.reshape(X.shape[0], 32, 32, 1)

        print("\nloading model: ./models/classifier\n")
        model_1 = keras.models.load_model("./models/classifier")

        y_pred_1 = model_1.predict(X)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\nloading model: ./models/classifier_avg_ft\n")
        model_2 = keras.models.load_model("./models/classifier_avg_ft")
        y_pred_2 = model_2.predict(X)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

    if args.augmentation == 'mtu':
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
            labels.append(np.full(data.shape[0], label))
            data_2.append(data)
        
        y_true = np.concatenate(labels)

        print("\n----------  Running Test 1: ----------")
        print(  "---------- A comparison to show the performance drop when the MTU is less than 1500. ----------")

        X_1 = np.concatenate(data_1)
        X_1 = X_1.reshape(X.shape[0], 32, 32, 1)

        X_2 = np.concatenate(data_2)
        X_2 = X_2.reshape(X.shape[0], 32, 32, 1)

        print("\nloading model: ./models/classifier\n")
        model_1 = keras.models.load_model("./models/classifier")

        y_pred_1 = model_1.predict(X_1)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        y_pred_2 = model_1.predict(X_2)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

        print("\n---------- Running Test 2: ----------")
        print(  "----------  A comparison to show the performance drop when MTU augmentation data is added to the train dataset. ----------")
        
        y_pred_1 = model_1.predict(X_1)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        print("\nloading model: ./models/classifier_mtu\n")
        model_2 = keras.models.load_model("./models/classifier_mtu")
        y_pred_2 = model_2.predict(X_1)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))

        print("\n---------- Running Test 3: ----------")
        print(  "---------- A comparison to show the performance improvementwhen MTU augmentation data is added to the train dataset. ----------")

        y_pred_1 = model_1.predict(X_2)
        y_pred_1 =np.argmax(y_pred_1,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_1))
        print(classification_report(y_true= y_true, y_pred= y_pred_1))

        y_pred_2 = model_2.predict(X_2)
        y_pred_2 =np.argmax(y_pred_2,axis=1)
        
        print(confusion_matrix(y_true= y_true, y_pred= y_pred_2))
        print(classification_report(y_true= y_true, y_pred= y_pred_2))


    # # masks = np.load(f"../../MiniFlowPic_Data/MiniFlowPic/{DATA_DIR}/{DATA_DIR}_mfp_first_15_val_masks.npy", allow_pickle= True)

    # # FlowPic_Path = f"../../MiniFlowPic_Data/MiniFlowPic/{DATA_DIR}"
    # # filenames_test = glob.glob(FlowPic_Path + '/**/*first_15.npy')

    # # for i in range(6, 7):
    #     # model_name = MODEL +"att_lc_split_8_max_len_32"
    #     # print("\nloading model:", model_name,'\n')
    #     # model = keras.models.load_model(model_name , custom_objects= {"ssim_loss": ssim_loss, "classifier_loss": classifier_loss}) # , custom_objects= {"ssim_loss": ssim_loss, "classifier_loss": classifier_loss} , 'MQE': MQE
    #     # print(model.summary())
    #     labels = []
    #     preds = []
    #     show = "t"
    #     for _path in filenames_test:
            
    #         label = get_label(_path)
    #         dataset = train_validation_split(_path, masks[label], 1)
    #         dataset = dataset['val']
    #         labels.append(np.full(dataset.shape[0], label))

    #         transpose = transpose_images(dataset)
            
    #         imgs_f_half = transpose[:,:SPLIT , :]

    #         # data = imgs_f_half

    #         # imgs_s_half = transpose[:,SPLIT:MAX_LEN , :]

    #         data = dataset

    #         # # _model = keras.Model(inputs= model.layers[0].input, outputs= model.layers[11].output)

    #         # imgs_s_half_pred= model.predict(imgs_f_half)
    #         # # imgs_s_half_pred = imgs_f_half
    #         # for i in range(imgs_s_half.shape[0]):
    #         #     if show == "f":
    #         #         break
    #         #     org = imgs_s_half[i]
    #         #     new = imgs_s_half_pred[i]
    #         #     comparison(org.T, new.T , True ,'', show= True) # , save= f'../../plots/lstm_results/r_{i}', save= f'../../plots/lstm_results/r_label_{label}_num_{i}'
    #         #     show = input("continue drawing?\n")
    #         #     print("show=", show)
                

    #         # preds.append(detranspose_images(imgs_f_half, imgs_s_half_pred)) #_pred
    #         preds.append(data) #_pred


    #     y_true = np.concatenate(labels)
    #     X = np.concatenate(preds)
    #     X = X.reshape(X.shape[0], SPLIT, 32)
    #     # X = X.reshape(X.shape[0], 32, 32, 1)

    #     print(X.shape, y_true.shape)

    #     print("\nloading model:", CLASSIFIER,'\n')
    #     classifier = keras.models.load_model(CLASSIFIER)
    #     print(classifier.summary())
    #     print(classifier.evaluate(X ,y_true))
    #     y_pred = classifier.predict(X)
    #     y_pred=np.argmax(y_pred,axis=1)
        
    #     # _map = np.vectorize(lambda x: 2 if x >= 2 else x)
    #     # y_true = _map(y_true)
    #     # y_pred = _map(y_pred)
    #     print(confusion_matrix(y_true= y_true, y_pred= y_pred))
    #     print(classification_report(y_true= y_true, y_pred= y_pred))

    # return 0
    

if __name__ == "__main__":
    main()
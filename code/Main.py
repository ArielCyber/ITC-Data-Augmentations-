import argparse
import os
import glob
import gc

import Average_Augmentation
import Classifier_Train 
import Concat_Files 
import Fine_Tuning
import LSTM_Classifier_Loss
import LSTM_Generate_Data
import LSTM_TFRecords_Converter
import LSTM_TFRecords_Utils
import LSTM
import MTU_Augmentation
import TFRecords_Converter
import TFRecords_Utils
import Train_Validation_Split
import Change_RTT_Augmentation
import Time_Shift_Augmentation
import Packet_Loss_Augmentation

from keras.backend import clear_session
import tensorflow

# Reset Keras Session
def reset_keras(model):
    clear_session()
    try:
        del model
    except:
        pass

    gc.collect() 



def main():
    # set code arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing the training data.')
    parser.add_argument('augmentation', choices=['lstm', 'average', 'mtu', 'change_rtt', 'time_shift', 'packet_loss'], help='Augmentation method to use.')
    parser.add_argument('--test_split', type=float, default=0.2, required=False, help='Fraction of the data to use for the test set.')
    parser.add_argument('--val_split', type=float, default=0.2, required=False, help='Fraction of the data to use for the validation set.')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Batch size to use for training.')
    parser.add_argument('--split', type=int, default=16, required=False, help='The split of the flow.')
    parser.add_argument('--max_len', type=int, default=32, required=False, help='Maximum length of a flow.')
    parser.add_argument('--data_max_size', type=int, default=-1, required=False, help='Maximum number of data points to use.')
    parser.add_argument('--avg_n', type=int, default=2, required=False, help='Number of data points to average for the Average augmentation.')
    parser.add_argument('--th_min', type=int, default=750, required=False, help='Minimum threshold for the MTU augmentation.')
    parser.add_argument('--th_max', type=int, default=1200, required=False, help='Maximum threshold for the MTU augmentation.')

    args = parser.parse_args()
    # fix data directory path 
    if args.data_dir[-1] != '/':
        args.data_dir += '/'
    
    #list all classes
    classes = glob.glob(args.data_dir + "*/")
    classes = [_dir[len(args.data_dir):-1] for _dir in classes ]
    print("\nclasses:", classes)
    num_of_classes = len(classes)

    print("\n---------- Running Train Test split with test split of:", args.test_split, "----------")

    if args.data_max_size != -1:
        print("---------- with data max size of:", args.data_max_size)

    Concat_Files.iterate_all_classes(args.data_dir, classes, 1 - args.test_split, args.data_max_size)
    
    print("\n---------- Running Train Validation split with validation split of:", args.val_split, "----------")
    masks_name = Train_Validation_Split.iterate_all_classes(args.data_dir, args.val_split)


    print("\n---------- Converting data to tfrecords ----------")
    TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, classes)

    no_aug_train = glob.glob(args.data_dir + '**/*_first_15_train.tfrecords')
    no_aug_val = glob.glob(args.data_dir + '**/*_first_15_val.tfrecords')
    aug_train = []
    aug_val = []
    model_name_suffix = ""

    print("\n---------- Training test classifier ----------")
    model = Classifier_Train.train_model(args.batch_size, num_of_classes, no_aug_train, no_aug_val, aug_train, aug_val, model_name_suffix)

    reset_keras(model)

    if args.augmentation == 'average':
        print("\n---------- Generating average data with average of:",args.avg_n, "----------")
        avg_masks = Average_Augmentation.iterate_all_classes(args.data_dir, masks_name, classes, args.avg_n)

        print("\n---------- Converting average data to tfrecords ----------")
        TFRecords_Converter.iterate_all_classes(args.data_dir, avg_masks, classes, f'*_first_15_avg_{args.avg_n}.npy')

        aug_train = glob.glob(args.data_dir + f'**/*_first_15_avg_{args.avg_n}_train.tfrecords')
        aug_val = glob.glob(args.data_dir + f'**/*_first_15_avg_{args.avg_n}_val.tfrecords')
        model_name_suffix = f"_avg_{args.avg_n}"
        no_aug_train = []
        no_aug_val = []
    
    elif args.augmentation == 'mtu':
        print("\n---------- Generating MTU data with threshold min of:", args.th_min, "and threshold max of:", args.th_max, "----------")
        MTU_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max)
        MTU_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max, '*first_15_test.npy')

        print("\n---------- Converting MTU data to tfrecords ----------")
        TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, classes, f'*_first_15_mtu.npy')

        aug_train = glob.glob(args.data_dir + '**/*_first_15_mtu_train.tfrecords')
        aug_val = glob.glob(args.data_dir + '**/*_first_15_mtu_val.tfrecords')
        model_name_suffix = "_mtu"

    elif args.augmentation == 'change_rtt':
        print("\n---------- Generating a random changed rtt data ----------")
        Change_RTT_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max)
        Change_RTT_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max, '*first_15_test.npy')

        print("\n---------- Converting changed RTT data to tfrecords ----------")
        TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, classes, f'*_first_15_rtt.npy')

        aug_train = glob.glob(args.data_dir + '**/*_first_15_rtt_train.tfrecords')
        aug_val = glob.glob(args.data_dir + '**/*_first_15_rtt_val.tfrecords')
        model_name_suffix = "_rtt"

    elif args.augmentation == 'time_shift':
        print("\n---------- Generating a random time shifted data ----------")
        Time_Shift_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max)
        Time_Shift_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max, '*first_15_test.npy')

        print("\n---------- Converting time shift data to tfrecords ----------")
        TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, classes, f'*_first_15_time_shifted.npy')

        aug_train = glob.glob(args.data_dir + '**/*_first_15_time_shifted_train.tfrecords')
        aug_val = glob.glob(args.data_dir + '**/*_first_15_time_shifted_val.tfrecords')
        model_name_suffix = "_time_shift"
    
    elif args.augmentation == 'packet_loss':
        print("\n---------- Generating a random packet loss data ----------")
        Time_Shift_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max)
        Time_Shift_Augmentation.iterate_all_classes(args.data_dir, args.th_min, args.th_max, '*first_15_test.npy')

        print("\n---------- Converting packet loss data to tfrecords ----------")
        TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, classes, f'*_first_15_packet_loss.npy')

        aug_train = glob.glob(args.data_dir + '**/*_first_15_packet_loss_train.tfrecords')
        aug_val = glob.glob(args.data_dir + '**/*_first_15_packet_loss_val.tfrecords')
        model_name_suffix = "_packet_loss"


    if args.augmentation != 'lstm':
        print("\n---------- Training classifier ----------")
        model = Classifier_Train.train_model(args.batch_size, num_of_classes, no_aug_train, no_aug_val, aug_train, aug_val, model_name_suffix)

        reset_keras(model)

    if args.augmentation == "average":
        no_aug_train = glob.glob(args.data_dir + '**/*_first_15_train.tfrecords')
        no_aug_val = glob.glob(args.data_dir + '**/*_first_15_val.tfrecords')

        print("\n---------- Fine tuning classifier ----------")
        model = Fine_Tuning.train_model("./models/classifier" + model_name_suffix, args.batch_size, num_of_classes, no_aug_train, no_aug_val)

        reset_keras(model)

    if args.augmentation != 'lstm':
        return
    
    print("\n---------- Converting data to LSTM tfrecords ----------")
    LSTM_TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, True, args.max_len, args.split, classes )
    LSTM_TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, False, args.max_len, args.split, classes )

    LSTM_TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, True, args.split, 0, classes )

    print("\n---------- Training LSTM loss classifier ----------")
    classifier_loss_train = glob.glob(args.data_dir + '**/*_train_split_'+ str(args.split)+'_max_len_'+str(args.max_len)+ '_lstm_label.tfrecords')
    classifier_loss_val = glob.glob(args.data_dir + '**/*_val_split_'+ str(args.split)+'_max_len_'+str(args.max_len)+ '_lstm_label.tfrecords')

    model = LSTM_Classifier_Loss.train_model(num_of_classes, args.max_len, args.split, classifier_loss_train, classifier_loss_val, args.batch_size)

    reset_keras(model)

    print("\n---------- Training first", args.split ,"test classifier ----------")
    classifier_loss_train = glob.glob(args.data_dir + '**/*_train_split_'+ str(0)+'_max_len_'+str(args.split)+ '_lstm_label.tfrecords')
    classifier_loss_val = glob.glob(args.data_dir + '**/*_val_split_'+ str(0)+'_max_len_'+str(args.split)+ '_lstm_label.tfrecords')

    model = LSTM_Classifier_Loss.train_model(num_of_classes, args.split, 0, classifier_loss_train, classifier_loss_val, args.batch_size)

    reset_keras(model)

    print("\n---------- Training LSTM model ----------")
    lstm_train = glob.glob(args.data_dir + '**/*_train_split_'+ str(args.split)+'_max_len_'+str(args.max_len)+ '_lstm.tfrecords')
    lstm_val = glob.glob(args.data_dir + '**/*_val_split_'+ str(args.split)+'_max_len_'+str(args.max_len)+ '_lstm.tfrecords')

    model = LSTM.train_model(lstm_train, lstm_val, args.max_len, args.split, args.batch_size)
    
    reset_keras(model)

    print("\n---------- Generating LSTM data ----------")
    LSTM_Generate_Data.generate_data(args.data_dir, masks_name, args.max_len, args.split, classes)

    print("\n---------- Converting LSTM generated data to tfrecords ----------")
    TFRecords_Converter.iterate_all_classes(args.data_dir, masks_name, classes, f'*_first_15_lstm_split_{args.split}_max_len_{args.max_len}.npy')

    print("\n---------- Training classifier with LSTM generated data ----------")
    aug_train = glob.glob(args.data_dir + f'**/*_first_15_lstm_split_{args.split}_max_len_{args.max_len}_train.tfrecords')
    aug_val = glob.glob(args.data_dir + f'**/*_first_15_lstm_split_{args.split}_max_len_{args.max_len}_val.tfrecords')
    model = Classifier_Train.train_model(args.batch_size, num_of_classes, no_aug_train, no_aug_val, aug_train, aug_val, "_lstm_generated")

    reset_keras(model)

if __name__ == "__main__":
    main()
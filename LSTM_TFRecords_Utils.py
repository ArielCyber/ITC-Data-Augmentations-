import numpy as np
import tensorflow as tf
from Train_Validation_Split import train_validation_split

AUTOTUNE = tf.data.AUTOTUNE

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transpose_images(images):
    images = np.reshape(images , (images.shape[0], images.shape[2], images.shape[3]))
    out = []
    for img in images:
        out.append(img.T)
    return np.array(out)

def convert_to_lstm_tfrecord(file_name, mask, label, max_len, split):
    datasets = train_validation_split(file_name, mask, 1 )
    for name, dataset in datasets.items():
        tfrecord_filename = str(file_name)[:-4] +'_' + name+ '_split_'+ str(split) +'_max_len_'+str(max_len)+ '_lstm'+ '.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecord_filename)
        dataset = transpose_images(dataset)
        for img in dataset:
            img = img.astype(np.float32)
            img_f_half = img[:split , :]
            img_s_half = img[split:max_len , :]
            feature = { 
                'height_f': _int64_feature(img_f_half.shape[0]),
                'height_s': _int64_feature(img_s_half.shape[0]),
                'width_f': _int64_feature(img_f_half.shape[1]),
                'width_s': _int64_feature(img_s_half.shape[1]),
                'f_half': _bytes_feature(img_f_half.tobytes()),
                's_half': _bytes_feature(img_s_half.tobytes())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

        writer.close()
        print("created file:",tfrecord_filename)



def decode_fn(record_bytes):

	parse_dict = {
        'height_f': tf.io.FixedLenFeature([], tf.int64),
        'height_s': tf.io.FixedLenFeature([], tf.int64),
        'width_f': tf.io.FixedLenFeature([], tf.int64),
        'width_s': tf.io.FixedLenFeature([], tf.int64),
        'f_half': tf.io.FixedLenFeature([], tf.string),
        's_half': tf.io.FixedLenFeature([], tf.string)
	}

	example = tf.io.parse_single_example(record_bytes,parse_dict)


	img_raw_f_half = example['f_half']
	height_f = example['height_f']
	width_f = example['width_f']


	img_raw_s_half = example['s_half']
	height_s = example['height_s']
	width_s = example['width_s']


	feature = tf.io.decode_raw(img_raw_f_half, tf.float32)
	feature = tf.reshape(feature, shape=[height_f, width_f])
	target = tf.io.decode_raw(img_raw_s_half, tf.float32)
	target = tf.reshape(target, shape=[height_s, width_s])
	return (feature, target)



def convert_to_lstm_tfrecord_label(file_name, mask, label, max_len, split):
    datasets = train_validation_split(file_name, mask, 1 )
    for name, dataset in datasets.items():
        tfrecord_filename = str(file_name)[:-4] +'_' + name+ '_split_'+ str(split)+'_max_len_'+str(max_len)+ '_lstm_label'+ '.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecord_filename)
        dataset = transpose_images(dataset)
        for img in dataset:
            img = img.astype(np.float32)
            img_s_half = img[:split, : ]
            feature = { 
                'height_s': _int64_feature(img_s_half.shape[0]),
                'width': _int64_feature(img_s_half.shape[1]),
                's_half': _bytes_feature(img_s_half.tobytes()),
                'label': _int64_feature(label)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

        writer.close()
        print("created file:",tfrecord_filename)



def decode_fn_label(record_bytes):

	parse_dict = {
        'height_s': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        's_half': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
	}

	example = tf.io.parse_single_example(record_bytes,parse_dict)

	width = example['width']
	

	img_raw_s_half = example['s_half']
	height_s = example['height_s']


	feature = tf.io.decode_raw(img_raw_s_half, tf.float32)
	feature = tf.reshape(feature, shape=[height_s, width])
	
	return (feature, example['label'])



def get_dataset(filename, set_type, label= False, batch_size = 32):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable native order, increase speed
    print("\nimported data:", filename,'\n')
    dataset = tf.data.TFRecordDataset(filename, buffer_size= 100 , num_parallel_reads= 4)
    
    dataset = dataset.with_options(
        ignore_order
    )  
    decoder = decode_fn_label if label else decode_fn
    dataset = dataset.map(
        decoder, num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.shuffle(2048 , reshuffle_each_iteration=True ) 
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size, num_parallel_calls= AUTOTUNE, drop_remainder= True)
    dataset = dataset.cache()
    return dataset

def tfrecord_to_numpy(dataset):
    return list(dataset.as_numpy_iterator())

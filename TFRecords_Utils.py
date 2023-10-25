import numpy as np
import tensorflow as tf
from Train_Validation_Split import train_validation_split


AUTOTUNE = tf.data.AUTOTUNE

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(file_name, label, mask):
    datasets = train_validation_split(file_name, mask, 1)
    for name, dataset in datasets.items():
        tfrecord_filename = str(file_name)[:-4] +'_' + name + '.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecord_filename)
        print(name, "shape:", dataset.shape)
        for img in dataset:
            img = img.astype(np.float32)
            feature = { 
                'depth': _int64_feature(img.shape[0]),
                'height': _int64_feature(img.shape[1]),
                'width': _int64_feature(img.shape[2]),
                'label': _int64_feature(label),
                'image': _bytes_feature(img.tobytes()) }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

        writer.close()
        print("created file:",tfrecord_filename)



def decode_fn(record_bytes):

	parse_dict = {
        'depth':tf.io.FixedLenFeature([], tf.int64),
		'height': tf.io.FixedLenFeature([], tf.int64),
		'width':tf.io.FixedLenFeature([], tf.int64),
        'label':tf.io.FixedLenFeature([], tf.int64),
		'image' : tf.io.FixedLenFeature([], tf.string)
	}

	example = tf.io.parse_single_example(record_bytes,parse_dict)

	img_raw = example['image']
	height = example['height']
	width = example['width']
	depth = example['depth']
	label = example['label']
	feature = tf.io.decode_raw(img_raw, tf.float32)
	feature = tf.reshape(feature, shape=[height, width, depth])
	return (feature, label)



def get_dataset(filename, set_type, batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable native order, increase speed
    print("\nimported data:", filename,'\n')
    dataset = tf.data.TFRecordDataset(filename, buffer_size= 100 , num_parallel_reads= 4)
    
    dataset = dataset.with_options(
        ignore_order
    )  

    dataset = dataset.map(
        decode_fn, num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.shuffle(2048 , reshuffle_each_iteration=True ) 
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size, num_parallel_calls= AUTOTUNE, drop_remainder= True)
    dataset = dataset.cache()
    
    return dataset

def tfrecord_to_numpy(dataset):
    return list(dataset.as_numpy_iterator())

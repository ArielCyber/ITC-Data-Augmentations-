from datetime import datetime
import tensorflow as tf
import keras
import keras.layers as layers
from keras.layers import *
import glob
from TFRecords_Utils import *

tf.config.run_functions_eagerly(True)

def cnn(num_of_classes):
	input_shape = (32,32,1)
	sizes = [(6,5), (16,5), 120, 84]
	
	model = keras.Sequential()

	model.add(layers.Conv2D(filters=sizes[0][0], kernel_size=sizes[0][1], activation='relu', input_shape=input_shape))
	model.add(layers.MaxPooling2D())

	model.add(layers.Conv2D(filters=sizes[1][0], kernel_size=sizes[1][1], activation='relu'))
	model.add(layers.Dropout(0.25))
	model.add(layers.MaxPooling2D())
	

	model.add(layers.Flatten())

	model.add(layers.Dense(units=sizes[2], activation='relu'))
	

	model.add(layers.Dense(units=sizes[3], activation='relu'))
	model.add(layers.Dropout(0.5))


	model.add(layers.Dense(units=num_of_classes, activation = 'softmax', name= "Classifier"))

	model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate= 5e-4), metrics=['accuracy'] )
	
	return model


def train_model(batch_size, num_of_classes, no_aug_train = [], no_aug_val = [], aug_train = [], aug_val = [], model_name_suffix = ""):
	filenames_train =  list(no_aug_train)  +  list(aug_train)
	filenames_val =    list(no_aug_val) + list(aug_val) 

	dataset_train = get_dataset(filenames_train, 'train', batch_size)
	dataset_val = get_dataset(filenames_val, 'val', batch_size)
	model = cnn(num_of_classes)
	callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience= 3, restore_best_weights = True)

	print(model.summary())
	model.fit(dataset_train, epochs= 10 , callbacks= [callback], validation_data= dataset_val)
	model_name = "./models/classifier" + model_name_suffix
	keras.models.save_model(model, model_name)
	print("\ncreated model:", model_name)
	return model
import tensorflow as tf
import keras
import keras.layers as layers
from keras.layers import *
import glob
from TFRecords_Utils import *
from LSTM_TFRecords_Utils import *

def cnn(num_of_classes, max_len, split):
	input_shape = (max_len - split,32)
	sizes = [(6,5), (16,3), 60, 42]
	
	model = keras.Sequential()
	model.add(layers.Reshape((max_len - split, 32 ,1), input_shape=input_shape))

	model.add(layers.Conv2D(filters=sizes[0][0], kernel_size=sizes[0][1], activation='relu'))

	model.add(layers.Conv2D(filters=sizes[1][0], kernel_size=sizes[1][1], activation='relu'))
	model.add(layers.Dropout(0.25))
	model.add(layers.MaxPooling2D())
	

	model.add(layers.Flatten())

	model.add(layers.Dense(units=sizes[2], activation='relu'))
	

	model.add(layers.Dense(units=sizes[3], activation='relu'))
	model.add(layers.Dropout(0.5))

	model.add(layers.Dense(units=num_of_classes, activation = 'softmax', name= "Classifier"))

	model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate= 0.001), metrics=['accuracy'] )
	
	return model


def train_model(num_of_classes, max_len, split, train_data, val_data, batch_size):
	dataset_train = get_dataset(train_data, 'train', True, batch_size)
	dataset_val= get_dataset(val_data, 'val', True, batch_size)
	model = cnn(num_of_classes, max_len, split)
	callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience= 5, restore_best_weights = True)

	print(model.summary())
	model.fit(dataset_train, epochs= 10 , callbacks= [callback], validation_data= dataset_val) 
	model_name = "./models/lstm_classifier_split_" + str(split) + '_max_len_'+str(max_len)
	keras.models.save_model(model,  model_name )
	print("\ncreated model:", model_name)
	return model


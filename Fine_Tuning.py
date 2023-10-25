from datetime import datetime
import tensorflow as tf
import keras
import keras.layers as layers
import glob
from TFRecords_Utils import *

def train_model(model_name, batch_size, num_of_classes, train_data, val_data):
	dataset_train = get_dataset(train_data, 'train', batch_size)
	dataset_val= get_dataset(val_data, 'val', batch_size)

	base_model = keras.models.load_model(model_name)
	print("\nloading model:", model_name)

	
	model_length = len(base_model.layers)
	for i in range(model_length):
		base_model.layers[i].trainable = False
	
	l = base_model.layers[model_length -2].output
	l = layers.Dense(50, activation = 'relu', name= 'dense_1_new')(l)
	l = layers.Dense(10, activation = 'relu', name= 'dense_2_new')(l)
	l = layers.Dense(num_of_classes, activation = 'softmax', name= 'classification_new')(l)

	model = keras.Model(inputs = base_model.input, outputs = l)

	model.compile(loss =keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate= 0.005), metrics=['accuracy'] )

	callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience= 5, restore_best_weights = True)

	print(model.summary())
	model.fit(dataset_train, epochs= 20 , callbacks= [callback] , validation_data= dataset_val) 
	model_name = f"./models/classifier_avg_tl"
	keras.models.save_model(model, model_name)
	print("\ncreated model:", model_name)

	model_length = len(model.layers)
	for i in range(model_length):
		model.layers[i].trainable = True
	
	model.compile(loss =keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate= 5e-4), metrics=['accuracy'] )
	print(model.summary())
	model.fit(dataset_train, epochs= 30  , callbacks= [callback], validation_data= dataset_val)

	model_name = f"./models/classifier_avg_ft"
	keras.models.save_model(model, model_name)
	print("\ncreated model:", model_name)
	return model
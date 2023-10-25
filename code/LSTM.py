import keras
import keras.layers as layers
import glob
from LSTM_TFRecords_Utils import *

def ssim_loss(y_true, y_pred):
	y_true = tf.cast(y_true, tf.float32)
	mse = tf.keras.losses.MeanSquaredError()
	return tf.reduce_mean(
		tf.add(
			tf.math.scalar_mul(
					tf.convert_to_tensor(1/256,dtype= tf.float32),
					mse(y_true, y_pred)
				),
			tf.math.subtract(
					tf.convert_to_tensor(1,dtype= tf.float32),
					tf.image.ssim(y_true, y_pred, 1.0, filter_size= 3)
				)
				)
			)

class classifier_loss:
	def __init__(self, max_len, split):
		self.max_len = max_len
		self.split = split
		self.classifier = keras.models.load_model("./models/lstm_classifier_split_" + str(self.split) + '_max_len_'+str(self.max_len))

	def __call__(self, y_true, y_pred):
		y_true = tf.cast(y_true, dtype= tf.float32)
		y_pred = tf.cast(y_pred, dtype= tf.float32)
		l_true = self.classifier(y_true)
		l_pred = self.classifier(y_pred)
		cce = tf.keras.losses.CategoricalCrossentropy()
		mse = tf.keras.losses.MeanSquaredError()
		return tf.reduce_mean(
				cce(l_true, l_pred)+
					(tf.convert_to_tensor(1,dtype= tf.float32)/tf.size(y_true, out_type= tf.float32))*
					mse(y_true, y_pred)+
					tf.math.subtract(
						tf.convert_to_tensor(1,dtype= tf.float32),
						tf.image.ssim(y_true, y_pred, 1.0, filter_size= 3)
					)
					)
	def get_config(self):
		config = {
			'max_len': self.max_len,
			'split': self.split
		}
		return config


def lstm(max_len, split):
	input_shape = (split, 32)

	inputs = layers.Input(shape= input_shape)
	l = inputs
	
	l = layers.LocallyConnected1D(500, 1, activation= 'relu')(l)
	l = layers.LocallyConnected1D(1000, 1, activation= 'relu')(l)

	lstm_out = 1000
	lstm = layers.LSTM( lstm_out, return_sequences= True)(l) 

	att = layers.Dense(1, activation='tanh')(lstm)
	att = layers.Flatten()(att)
	att = layers.Activation(activation="softmax")(att)
	att = layers.RepeatVector(lstm_out)(att)
	att = layers.Permute((2,1))(att)
	l = layers.multiply([att, lstm])
	l = layers.Flatten()(l)
	l = layers.RepeatVector(max_len-split)(l)


	l = layers.LSTM(1000, return_sequences= True )(l)

	l = layers.Dense(500, activation = 'relu')(l)
	l = layers.Dropout(0.25)(l)

	l = layers.Dense(250, activation = 'relu')(l)

	l = layers.Dense(100, activation = 'relu')(l)

	l = layers.Dense(50, activation = 'relu')(l)

	l = layers.Dense(32, activation = 'linear')(l)

	model = keras.Model(inputs= inputs, outputs= l)

	model.compile(loss= classifier_loss(max_len, split), optimizer=keras.optimizers.Adam(learning_rate= 1e-5 ), metrics= [ 'mean_absolute_error', ssim_loss, 'mse'])
	
	return model


def train_model(train_data, val_data, max_len, split, batch_size):
	with tf.device('/device:GPU:1'):
		dataset_train = get_dataset(train_data, 'train', False, batch_size)
		dataset_val= get_dataset(val_data, 'val', False, batch_size)

		model = lstm(max_len, split)
		print(model.summary())

		callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience= 100, restore_best_weights = True)

		model.fit(dataset_train, epochs= 5000 , validation_data= dataset_val)
		model_name= "./models/lstm_split_" + str(split) + '_max_len_'+str(max_len)
		keras.models.save_model(model,model_name )
		print("\ncreated model:",model_name )
		return model


if __name__ == '__main__':
    main()

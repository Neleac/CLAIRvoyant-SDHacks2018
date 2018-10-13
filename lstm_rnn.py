import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#load data
file_path = 'data\\alice_in_wonderland.txt'
book = open(file_path, encoding = "utf8").read().lower()

#covert unique chars to int
chars = sorted(list(set(book)))
char_int_dict = {}
for integer, character in enumerate(chars):
	char_int_dict[character] = integer

total_chars = len(book)
unique_chars = len(chars)

filter_len = 100
dataX = []
dataY = []
for i in range(0, total_chars - filter_len):
	lineX = book[i: i + filter_len]
	lineY = book[i + filter_len]

	char_int_list = []
	for char in lineX:
		char_int_list.append(char_int_dict[char])
	
	dataX.append(char_int_list)
	dataY.append(char_int_dict[lineY])

total_filters = len(dataX)

#prepare training data
X_train = np.reshape(dataX, (total_filters, filter_len, 1))
X_train = X_train / float(unique_chars)
y_train = np_utils.to_categorical(dataY)

#print(X_train.shape)
#(144225, 100, 1)
#print(y_train.shape)
#(144225, 45)

'''
LSTM RNN MODEL

Pre: Input into this model (train,valid) should have been preprocessed and
reshaped into the following data format for X: [samples, time steps, features]
Post: Generates a keras sequential model instance to generate text predictions.
Model consists of a LSTM (Long Short Term Memory] layers.
'''

#========================MODEL==================++

starting_neurons =  16 # starting filters

if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(starting_neurons * 1, (5,)))
stroke_read_model.add(Conv1D(starting_neurons * 1, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 2, (5,)))
stroke_read_model.add(Conv1D(starting_neurons * 2, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 4, (3,)))
stroke_read_model.add(Conv1D(starting_neurons * 4, (3,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 8, (3,)))
stroke_read_model.add(Conv1D(starting_neurons * 8, (3,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization())
stroke_read_model.add(LSTM(starting_neurons * 16, return_sequences = True))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(LSTM(starting_neurons * 16, return_sequences = False))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Dense(starting_neurons * 32))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Dense(y.shape[1]), activation = 'softmax')
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
#stroke_read_model.summary()

#=======================HyperParameters===========

weight_path="weights.best.hdf5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4,
                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list = [checkpoint, early, reduceLROnPlat]

#========================TRAINING=================
from IPython.display import clear_output
stroke_read_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 128,
                      epochs = 60,
                      callbacks = callbacks_list,
                      verbose = 1)
clear_output()

#========================PREDICTIONS==============
model.load_weights('weights.best.hdf5')
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# reverse mapping to convert integers back into characters
int_to_char = dict((i, c) for i, c in enumerate(chars))

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "Done."
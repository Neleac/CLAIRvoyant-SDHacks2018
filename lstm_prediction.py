import sys
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.callbacks import *
from keras.utils import np_utils
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


#load data
file_path = 'data\\GoT.txt'
book = open(file_path, encoding = "utf8").read().lower()

#covert unique chars to int
chars = sorted(list(set(book)))
char_int_dict = {}
for integer, character in enumerate(chars):
	char_int_dict[character] = integer
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
total_chars = len(book)
unique_chars = len(chars)

filter_len = 100
dataX = []
dataY = []
for i in range(0, total_chars - filter_len):
	lineX = book[i: i + filter_len]
	lineY = book[i + filter_len]

	#char_int_list = []
	#for char in lineX:
	#	char_int_list.append(char_int_dict[char])
	
	#dataX.append(char_int_list)
	#dataY.append(char_int_dict[lineY])


	dataX.append([char_to_int[char] for char in lineX])
	dataY.append(char_to_int[lineY])

total_filters = len(dataX)

#prepare training data
X_train = np.reshape(dataX, (total_filters, filter_len, 1))
X_train = X_train / float(unique_chars)
y_train = np_utils.to_categorical(dataY)


starting_neurons =  16 # starting filters

if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(starting_neurons * 1, (5,)))
stroke_read_model.add(Conv1D(starting_neurons * 1, (5,)))
stroke_read_model.add(Dropout(0.3))
#stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 2, (5,)))
stroke_read_model.add(Conv1D(starting_neurons * 2, (5,)))
stroke_read_model.add(Dropout(0.3))
#stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 4, (3,)))
stroke_read_model.add(Conv1D(starting_neurons * 4, (3,)))
stroke_read_model.add(Dropout(0.3))
#stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 8, (3,)))
stroke_read_model.add(Conv1D(starting_neurons * 8, (3,)))
stroke_read_model.add(Dropout(0.3))
#stroke_read_model.add(BatchNormalization())
stroke_read_model.add(LSTM(starting_neurons * 16, return_sequences = True))
stroke_read_model.add(Dropout(0.3))
#stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
stroke_read_model.add(LSTM(starting_neurons * 16, return_sequences = False))
stroke_read_model.add(Dropout(0.3))
#stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
stroke_read_model.add(Dense(starting_neurons * 32))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
stroke_read_model.add(Dense(y_train.shape[1], activation = 'softmax'))


stroke_read_model.load_weights('weights.best.hdf5')
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# reverse mapping to convert integers back into characters
int_to_char = dict((i, c) for i, c in enumerate(chars))

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(unique_chars)
    prediction = stroke_read_model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    #sys.stdout.flush()
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

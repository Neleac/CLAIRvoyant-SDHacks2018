from clarifai.rest import ClarifaiApp
import cv2
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.callbacks import *
from keras.utils import np_utils

MY_API_KEY = 'e77c49d9b41845bb82caf94ab3f1471f'
app = ClarifaiApp(api_key = MY_API_KEY)

model = app.public_models.general_model

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def take_image():
	cam = cv2.VideoCapture(0)

	while True:
		ret, frame = cam.read()
		cv2.imshow('frame', frame)
		
		key = cv2.waitKey(1)

		if key%256 == 27:
			# ESC pressed
			print("Escape hit, closing...")
			break
		elif key%256 == 32:
			# SPACE pressed
			cv2.imwrite('images\\opencv_frame.jpg', frame)
			print("Image taken")
			break

	cam.release()
	cv2.destroyAllWindows()

seed = "hmm. well what do we have here, let’s take a look. now this is a very interesting picture. in this picture i detect "
seed_left = 134

def predict(address, seed, seed_left):

	response = model.predict_by_filename(address)

	concepts = response['outputs'][0]['data']['concepts']

	for concept in concepts:
		if seed_left > len(concept['name']):
			seed += concept['name'] + ", "
			seed_left -= len(concept['name']) + 2
		else:
			for i in range(seed_left):
				seed += " "
			break
	return seed

#MAIN
print("Hi there, my name is Clair. I am an AI image recognition agent based on Clarifai's vision API. Give me an image, and I'll give you my interpretation.\n")

#get image address
valid_response = False
while not valid_response:
	print("Enter 1 to take a picture from webcam\n")
	print("Enter 2 to choose a local image\n")
	print("Enter Q to quit")
	response = input()
	if response == str(1):
		valid_response = True
		print("Press Space to take picture")
		take_image()
		seed = predict('images\\opencv_frame.jpg', seed, seed_left)
	elif response == str(2):
		print("Enter image location")
		address = input()
		try:
			seed = predict(address, seed, seed_left)
			valid_response = True
		except:
			print("Invalid address\n")
	elif response == "Q":
		print("Goodbye!")
		break
	else:
		print("Invalid input\n")

#load data
file_path = 'data\\alice_in_wonderland.txt'
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

filter_len = 250#100
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
stroke_read_model.add(LSTM(starting_neurons * 16,input_shape=(X_train.shape[1], X_train.shape[2]) ,return_sequences = True))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(LSTM(starting_neurons * 16, return_sequences = False))
stroke_read_model.add(Dropout(0.2))
#stroke_read_model.add(Dense(starting_neurons * 32))
#stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Dense(y_train.shape[1], activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

stroke_read_model.load_weights('weights.best.hdf5')
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# reverse mapping to convert integers back into characters
int_to_char = dict((i, c) for i, c in enumerate(chars))

pattern = []
#for char in seed:
#	pattern.append(char_to_int[char])
pattern = [char_to_int[char] for char in seed]

print(seed)
#print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
#print(seed)
# generate characters
for i in range(0, 250):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(unique_chars)
    prediction = stroke_read_model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    sys.stdout.flush()
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")
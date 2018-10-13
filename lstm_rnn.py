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
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#========================MODEL==================

starting_neurons =  16

if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(starting_neurons * 3, (5,)))
stroke_read_model.add(Conv1D(starting_neurons * 3, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 4, (5,)))
stroke_read_model.add(Conv1D(starting_neurons * 4, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Conv1D(starting_neurons * 6, (3,)))
stroke_read_model.add(Conv1D(starting_neurons * 6, (3,)))
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
stroke_read_model.add(Dense(512))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
stroke_read_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
stroke_read_model.summary()

#=======================HyperParameters============

weight_path="{}_weights.best.hdf5".format('stroke_lstm_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

#========================PREDICTIONS===============
from IPython.display import clear_output
stroke_read_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 3000,#2048,
                      epochs = 50,
                      callbacks = callbacks_list)
clear_output()

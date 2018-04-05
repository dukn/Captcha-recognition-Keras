from __future__ import print_function
import os
import gzip, pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard

N_Data = 20000

batch_size = 32
digit = 4
alphabet = 10
num_classes = digit*alphabet
epochs = 20

img_rows, img_cols = 40, 150

f = gzip.open('data.pkl.gz', 'rb')
loaded_object = pickle.load(f)
x_train, x_test, y_train, y_test = loaded_object
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train /= 255.
x_test /= 255.
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def captcha_metric(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1,alphabet))
    y_true = K.reshape(y_true, (-1,alphabet))
    y_p = K.argmax(y_pred,axis=1)
    y_t = K.argmax(y_true,axis=1)
    r = K.mean( K.cast(K.equal(y_p, y_t), 'float32'))
    return r

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=[captcha_metric])

#if os.path.isfile('Network/model2.h5'):
#  model = load_model('Network/model.h5',custom_objects={'captcha_metric': captcha_metric})

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])

score = model.evaluate(x_test, y_test, verbose=0)
model.save('Network/model.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

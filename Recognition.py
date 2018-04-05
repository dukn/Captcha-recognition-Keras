from __future__ import print_function
import gzip, pickle
import numpy as np
import pylab
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from utils import *


N_Data = 20000

batch_size = 32
digit = 4
alphabet = 10
num_classes = digit*alphabet
epochs = 30

img_rows, img_cols = 40, 150

def captcha_metric(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1,alphabet))
    y_true = K.reshape(y_true, (-1,alphabet))
    y_p = K.argmax(y_pred,axis=1)
    y_t = K.argmax(y_true,axis=1)
    r = K.mean( K.cast(K.equal(y_p, y_t), 'float32'))
    return r

f = gzip.open('data.pkl.gz', 'rb')
loaded_object = pickle.load(f)
x_train, x_test, y_train, y_test = loaded_object
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_test_ = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test_scaled = x_test_/255.

model = load_model('Network/model.h5',custom_objects={'captcha_metric': captcha_metric})


def getPharse(onehot):
  res = ""
  onehot = onehot.reshape(-1,alphabet)
  onehot = np.argmax(onehot,axis=1)
  for e in onehot:
    res += list_char[e]
  return res 
'''

'''

y_pred = model.predict(x_test_scaled[:20])
y = y_test[:20]
#for k in y_pred:
#  print (getPharse(k))
#for k in y:
#  print (getPharse(k))

for i, img in enumerate(x_test[:20]):
  pylab.subplot(4,5,i+1); pylab.axis('off')
  pylab.imshow(img)
  if getPharse(y_pred[i])  == getPharse(y[i]):
    pylab.text(40,65,getPharse(y_pred[i]),color = 'b',size = 15)
  else:
    pylab.text(40,65,getPharse(y_pred[i]),color = 'r',size = 15)
  pylab.text(40,92,getPharse(y[i]),color = 'g',size = 15)
pylab.show()

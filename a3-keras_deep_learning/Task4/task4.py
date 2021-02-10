import numpy as np
import keras as K # <-- This is all we need to get Keras up and running
# Function that reads the CSV file and loads the data
# x : contains the features as Nx4 matrix (N is the number of observations)
# y : contains the corresponding label Nx1 vector
import wget


def load_data(filename):
  x = np.loadtxt(filename, delimiter=',', usecols=[1,2,3,4], dtype=np.float32, skiprows=1)
  y = np.loadtxt(filename, delimiter=',', usecols=[5], dtype=np.float32, skiprows=1)
  return (x, y)

# Lets download the the data we need. This is remotely hosted somewhere on the
import os.path
from os import path

TRAIN_DATA = 'https://cloudstor.aarnet.edu.au/plus/s/ieCuexofrjaZaYq/download'
TEST_DATA =  'https://cloudstor.aarnet.edu.au/plus/s/eLwLDo91lenckK1/download'

if not path.exists('test.csv'):
  !wget https://cloudstor.aarnet.edu.au/plus/s/eLwLDo91lenckK1/download -O test.csv

if not path.exists('train.csv'):
  !wget https://cloudstor.aarnet.edu.au/plus/s/ieCuexofrjaZaYq/download -O train.csv


input_dim = 4 # <- This is fixed. We do not need to change it.

learning_rate = 0.01 # Ranges from very small 0.0001 to 0.1
max_epochs =    500 # Another controllable parameter.

model = K.models.Sequential()
model.add(K.layers.Dense(units= 12 , activation='relu', input_dim=input_dim)) #Hidden layer
model.add(K.layers.Dense(units= 1, activation='sigmoid')) # output layer
model.compile(loss='binary_crossentropy',
              optimizer= K.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.95, nesterov=True),
              metrics=['accuracy'])

(train_x, train_y) = load_data('train.csv')
(test_x, test_y)   = load_data('test.csv')

h = model.fit(train_x, train_y,
              batch_size=32,
              epochs=max_epochs,
              verbose=2,
              validation_data = (test_x, test_y)
              )



import matplotlib.pyplot as plt

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(("train accuracy","test accuracy"))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(("train loss","test loss"))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

eval_results = model.evaluate(test_x, test_y, verbose=0)
print("\nLoss, accuracy on test data: ")
print("%0.4f %0.2f%%" % (eval_results[0], eval_results[1]*100))
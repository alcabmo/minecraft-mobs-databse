from keras.models import Sequential

import tensorflow as tf

import tensorflow_datasets as tfds

#tf.enable_eager_execution()

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random 
from numpy import *
from PIL import Image
import theano

IMG_SIZE=50
salir=False


training = []


while not salir:


    label= input("Clase de la carpeta? {0:Zombies 1:Skeleton 2:Creeper} (E para salir) :")

    if label in [str(0),str(1),str(2)]:
        label=int(label)

        dire = input("Ruta de la carpeta con las imagenes: ")
        archivos = os.listdir(dire)
        num_archivos= len(archivos)

        counter_arch=0
        for archivo in archivos:
            counter_arch+=1

            nueva_route=dire+"/"+archivo
            image = cv2.imread(nueva_route)
            if image is None:
                print("No se pudo cargar la imagen en ",route)
            
            alto, ancho, canales = image.shape
            for y in range(alto):
              for x in range(ancho):

                image_sust[y][x][0]=image[y][x][0]/255
                image_sust[y][x][1]=image[y][x][1]/255
                image_sust[y][x][2]=image[y][x][2]/255

            training.append([image_sust, label])

    else:
        salir=True

random.shuffle(training)

X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X = X.astype('float32')
from keras.utils import np_utils
Y = np_utils.to_categorical(y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

batch_size = 16
nb_classes =4
nb_epochs = 5
img_rows, img_columns = 200, 200
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(50, 50, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(4,  activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test, verbose = 0 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])

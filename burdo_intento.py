import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

NUMBER_OF_CLASSES=2
PORCENTAJE_TEST=0.2

salir=False
training_images=[]
training_labels=[]
testing_images=[]
testing_labels=[]

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
            image = cv.imread(nueva_route)
            if image is None:
                print("No se pudo cargar la imagen en ",route)




            elif( counter_arch/num_archivos <= 1-PORCENTAJE_TEST): #esto es para dividirlo en datos de entrenamiento y datos de test
                alto, ancho, canales = image.shape
                image_sust = [[[0] * canales for _ in range(ancho)] for _ in range(alto)]
                for y in range(alto):
                  for x in range(ancho):

                      image_sust[y][x][0]=image[y][x][0]/255
                      image_sust[y][x][1]=image[y][x][1]/255
                      image_sust[y][x][2]=image[y][x][2]/255

                training_images.append(image_sust)
                training_labels.append(label)

            else:
                alto, ancho,canales = image.shape
                image_sust = [[[0] * canales for _ in range(ancho)] for _ in range(alto)]
                for y in range(alto):
                  for x in range(ancho):

                      image_sust[y][x][0]=image[y][x][0]/255
                      image_sust[y][x][1]=image[y][x][1]/255
                      image_sust[y][x][2]=image[y][x][2]/255

                testing_images.append(image_sust)
                testing_labels.append(label)

    else:
        salir=True



print(len(training_images),"-",len(training_images[0]),"-",len(training_images[0][0]),"-",len(training_images[0][0][0]))

training_images = tf.convert_to_tensor(training_images)
testing_images = tf.convert_to_tensor(testing_images)
training_labels = tf.convert_to_tensor(training_labels)
testing_labels = tf.convert_to_tensor(testing_labels)

print(type(training_images))
print("weeeeeweeeeeeeeeeewweeeeeeeeeee")
print(type(training_labels))

model = models.Sequential()
model.add(layers.Conv2D(50, (3,3), activation='relu', input_shape=(50,50,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(100, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(100, (3,3), activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(NUMBER_OF_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print("loss: ",loss)
print("accuracy: ",accuracy)

model.save("mdl_guardado.model")

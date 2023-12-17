import os
import  cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

NUMBER_OF_CLASSES=2 # Skeleton y zombie
PORCENTAJE_TEST=0.2
IMAGE_SIZE = 50

salir=False
training_images=[]
training_labels=[]
testing_images=[]
testing_labels=[]

while not salir:

    dire = input("Ruta de la carpeta con las imagenes: ")
    label= input("Clase de la carpeta? {0:Zombies 1:Skeleton 2:Creeper} (E para salir) :")

    if label in [str(0),str(1),str(2)]:
        label=int(label)

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
                """
                image_flatten=[]
                for y in range(IMAGE_SIZE):
                    for x in range(IMAGE_SIZE):
                        valor_pixel = image[y,x]
                        image_flatten.append(valor_pixel[0])
                        image_flatten.append(valor_pixel[1])
                        image_flatten.append(valor_pixel[2])
                """
                training_images.append(image)
                training_labels.append(label)
            else:
                """
                image_flatten=[]
                for y in range(IMAGE_SIZE):
                    for x in range(IMAGE_SIZE):
                        valor_pixel = image[y,x]
                        image_flatten.append(valor_pixel[0])
                        image_flatten.append(valor_pixel[1])
                        image_flatten.append(valor_pixel[2])
                """
                training_images.append(image)
                training_labels.append(label)     
    else:
        salir=True

"""
for i in range(len(training_images)):
    training_images[i] = [x / 255 for x in training_images[i]]

for i in range(len(testing_images)):
    testing_images[i] = [x / 255 for x in testing_images[i]]
"""


"""
training_images = np.array(training_images)
training_labels= np.array(training_labels)
"""

#print(np.shape(training_images))
print(len(training_labels))
print(len(training_images))

"""
testing_images = np.array(testing_images)
testing_labels= np.array(testing_labels)
"""

model = models.Sequential()
model.add(layers.Conv2D(50, (3,3), activation='relu', input_shape=(50,50,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(100, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(100, (3,3), activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(NUMBER_OF_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print("loss: ",loss)
print("accuracy: ",accuracy)

model.save("mdl_guardado.model")

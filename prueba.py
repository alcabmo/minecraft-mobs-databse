import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import  cv2 as cv
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential





# create datasheet
batch_size = 10
img_height = 50
img_width = 50

data_dir = "PRUEBA"

training_percentage = 0.8
validation_percentage = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=training_percentage, subset="training",seed=123,image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=validation_percentage,subset="validation",seed=123,image_size=(img_height, img_width),batch_size=batch_size)

normalization_layer = layers.Rescaling(1./255)


# nombre classes
class_names = train_ds.class_names


# Estandarizar los datos
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

print(len(image_batch))

# Crear el modelo
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Entrenamos el modelo
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



# Predecir
image = cv.imread("zombie.jpg")
predictions = model.predict(image)
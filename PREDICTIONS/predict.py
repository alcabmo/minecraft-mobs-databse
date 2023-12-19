import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv

model_class = ['creeper', 'skeleton', 'zombie']

def predict(image_path):
    loaded_model = tf.keras.models.load_model('modelo.keras')
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(50, 50))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Añadir dimensión de lote

    # Realizar la predicción
    predictions = loaded_model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1)

    return predicted_class.numpy()[0]


input_image = input("Ingresa image a clasificar: ")

image_path_to_predict = input_image
original = cv.imread(image_path_to_predict)
resized_image = cv.resize(original, (50, 50))
cv.imwrite(input_image, resized_image)

predicted_class = predict(image_path_to_predict)
print(f'La imagen es: {model_class[predicted_class]}')

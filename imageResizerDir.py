import cv2
import os

dire="hola"
def redimensionar_guardar(route):
    global dire
    nueva_route=dire+"/"+route

    image = cv2.imread(nueva_route)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        print("No se pudo cargar la imagen en ",route)
    else:
    	target_size=(50, 50)
    	resized_image = cv2.resize(image, target_size)

    	output_image_path = "./exit/"+route
    	# Asegurarse de que el directorio de salida exista, si no, crearlo
    	os.makedirs(os.path.dirname(output_image_path), exist_ok=True)


    	cv2.imwrite(output_image_path, resized_image)
    	print("Imagen redimensionada y guardada en: ",output_image_path)


dire = input("Ruta de la carpeta con las imagenes: ")

archivos = os.listdir(dire)

for archivo in archivos:
    redimensionar_guardar(archivo)

import cv2
import numpy as np

# Cargar imagen y template
imagen = cv2.imread("imagen.jpg")
template = cv2.imread("template.jpg")
imagen2 = cv2.imread("imagen.jpg")
template2 = cv2.imread("template.jpg")

# Convertir a escala de grises
imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# Convertir a escala de grises
imagen_gray2 = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
template_gray2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

##################################################Correlacion Normalizada#######################################################
def norm_corr(template, image):
    # Obtener dimensiones del template e imagen
    t_height, t_width = template.shape[:2]
    i_height, i_width = image.shape[:2]

    # Inicializar matriz de correlación
    correlation = np.zeros((i_height - t_height + 1, i_width - t_width + 1))

    # Calcular la media y la desviación estándar del template
    t_mean = np.mean(template)
    t_std = np.std(template)

    # Calcular la correlación en cada píxel
    for y in range(correlation.shape[0]):
        for x in range(correlation.shape[1]):
            # Extraer el parche actual de la imagen y calcular su media y desviación estándar
            i_patch = image[y:y+t_height, x:x+t_width]
            i_mean = np.mean(i_patch)
            i_std = np.std(i_patch)

            # Calcular el coeficiente de correlación normalizada
            if t_std * i_std == 0:
                correlation[y,x] = 0
            else:
                correlation[y,x] = np.sum((i_patch - i_mean) * (template - t_mean)) / ((t_std + 1e-9) * (i_std + 1e-9) * t_height * t_width)

    return correlation


#########################################################################Diferencia de cuadrados#########################################################3
t_height, t_width = template_gray2.shape[:2]
i_height, i_width = imagen_gray2.shape[:2]
correlation = np.zeros((i_height - t_height + 1, i_width - t_width + 1))
# Calcular la media del template
t_mean = np.mean(template_gray2)
for y in range(correlation.shape[0]):
    for x in range(correlation.shape[1]):
        # Extraer el parche actual de la imagen
        i_patch = imagen_gray2[y:y+t_height, x:x+t_width]
        
        # Calcular la media del parche
        i_mean = np.mean(i_patch)

        # Calcular la diferencia al cuadrado
        correlation[y,x] = np.sum((i_patch - i_mean - template_gray + t_mean)**2)

# Encontrar la posición del valor mínimo de correlación
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation)
top_left = min_loc

# Dibujar un rectángulo en la ubicación del template encontrado
bottom_right = (top_left[0] + t_width, top_left[1] + t_height)
cv2.rectangle(imagen, top_left, bottom_right, (0, 255, 0), 2)
###################################################    Imagen (Diferencia de cuadrados)       ###########################################3

#################################################################### Correlacion Normalizada #############################################
result = norm_corr(template_gray, imagen_gray)

# Normalizar la matriz de correlación para escalar los valores entre 0 y 1
result_norm = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Encontrar la posición del valor máximo de correlación
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc

# Dibujar un rectángulo en la ubicación del template encontrado
bottom_right = (top_left[0] + template_gray.shape[1], top_left[1] + template_gray.shape[0])
cv2.rectangle(imagen2, top_left, bottom_right, (0, 255, 0), 2)
#################################################################### Correlacion Normalizada #############################################


# Mostrar imagenes resultante
cv2.imshow("Resultado", imagen)#Diferencia de cuadrados
cv2.imshow("Resultado 2", imagen2)#Correlacion Normalizada
cv2.waitKey(0)
cv2.destroyAllWindows()
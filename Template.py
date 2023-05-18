import cv2 
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
# leer las imagenes y el template
img_rgb = cv2.imread('monedas.jpg')
template = cv2.imread('moneda.jpg')

# guarda las dimenciones de la imagen
W, H = template.shape[:2]
  
# Define el minimo threshold
thresh = 0.4

#imagenes a escala gris

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
temp_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

# metodo de template matching
match = cv2.matchTemplate(image=img_gray, templ=temp_gray,method=cv2.TM_CCOEFF_NORMED)

#Seleccionar rectángulos con mayor coincidencia que el umbral
(y_points, x_points) = np.where(match >= thresh)

# Inicializar nuestra lista de cuadros delimitadores (bounding boxes)
boxes = list()

# Almacenar las coordenadas de cada cuadro delimitador
#crearemos una nueva lista iterando a través de cada par de puntos
for (x, y) in zip(x_points, y_points):
	
	# Actualizar nuestra lista de cuadros delimitadores
	boxes.append((x, y, x + W, y + H))
#aplicar supresión de no-máximo a los cuadros delimitadores
#esto creará un solo cuadro delimitado
boxes = non_max_suppression(np.array(boxes))
  
# ciclo sobre los cuadros delimitadores finales
for (x1, y1, x2, y2) in boxes:
    
    # dibujar el cuadro delimitador en la imagen
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2),
                  (255, 0, 0), 3)

cv2.imshow("Template", template)
cv2.imshow("resultaddo", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()    
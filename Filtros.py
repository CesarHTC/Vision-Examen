import numpy as np
import cv2
import imutils
import random
########################################    Ruido Gaussiano         ###############################################
# Cargar imagen 
img = cv2.imread('Ropa2.jpg')
img = imutils.resize(img,400)

# Obtener dimensiones de la imagen
filas, columnas, canales = img.shape

# Agregar ruido gaussiano a cada canal de color con media 0 y desviación estándar de 35
ruido = np.random.normal(0, 35, (filas, columnas, canales))
img_Gaussiana = img + ruido

# Asegurarse que los valores de la imagen estén entre 0 y 255
img_Gaussiana = np.clip(img_Gaussiana, 0, 255)

# Convertir a tipo de datos uint8
img_Gaussiana = np.uint8(img_Gaussiana)

############################################ Ruido Sal y Pimienta   ############################################## 
img2=cv2.imread("Ropa.jpg")
img2= imutils.resize(img2,300)
num=int(0.2*img2.shape[0]*img2.shape[1])# 1-Saco el area de la imagen y determino el Número de puntos de ruido de sal y pimienta
random.randint(0, img2.shape[0])#me da un numero entero en rango de 0 al numero de filas 
img_Sal_Pimienta=img2.copy()
for i in range(num):
    X=random.randint(0,img_Sal_Pimienta.shape[0]-1)# Un número entero aleatorio desde 0 hasta la longitud de la imagen, porque es un intervalo cerrado, -1
    Y=random.randint(0,img_Sal_Pimienta.shape[1]-1)
    if random.randint(0,1) ==0: # Probabilidad en blanco y negro 50%
        img_Sal_Pimienta[X,Y] = (255,255,255)#blanco
    else:
        img_Sal_Pimienta[X,Y] =(0,0,0)#negro

######################################   Filtros    ############################# 
#Media
img_Gaussiana_Media = cv2.blur(img_Gaussiana, (3, 5))
img_Sal_Pimienta_Media = cv2.blur(img_Sal_Pimienta, (3, 3))
#Gaussiano
img_Gaussiana_Gau = cv2.GaussianBlur(img_Gaussiana,(5,5),0)
img_Sal_Pimienta_Gau = cv2.GaussianBlur(img_Sal_Pimienta,(5,5),0)
#Mediana
img_Gaussiana_Mediana = cv2.medianBlur(img_Gaussiana,3)
img_Sal_Pimienta_Mediana = cv2.medianBlur(img_Sal_Pimienta,3)
#Minimo
img_Gaussiana_Minimo = cv2.erode(img_Gaussiana, (3,3))
img_Sal_Pimienta_Minimo = cv2.erode(img_Sal_Pimienta, (5,5))
#Maximo
img_Gaussiana_Maximo = cv2.dilate(img_Gaussiana, (3,3))
img_Sal_Pimienta_Maximo = cv2.dilate(img_Sal_Pimienta, (6,5))


# Mostrar imagen original y con ruido
#gausiano
h1_1= cv2.hconcat([img,img_Gaussiana])
h1_2= cv2.hconcat([img_Gaussiana_Media,img_Gaussiana_Gau,img_Gaussiana_Mediana,img_Gaussiana_Minimo,img_Gaussiana_Maximo])
cv2.imshow("Gausiana", h1_1)
cv2.imshow("Filtros de la imagen Gaussiana", h1_2)

# Sal y Pimienta
h2_1=cv2.hconcat([img2,img_Sal_Pimienta])
h2_2= cv2.hconcat([img_Sal_Pimienta_Media,img_Sal_Pimienta_Gau,img_Sal_Pimienta_Mediana,img_Sal_Pimienta_Minimo,img_Sal_Pimienta_Maximo])
cv2.imshow("Sal y Pimienta",h2_1)
cv2.imshow("Filtros de la imagen Sal y Pimienta",h2_2)
###############     Filtros     ######################


cv2.waitKey(0)
cv2.destroyAllWindows()

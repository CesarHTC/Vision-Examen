# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:57:22 2023

@author: cesar
"""
import cv2
import numpy as np
import imutils

path = r"C:\Users\cesar\Vision\hand4.jpg"
img = cv2.imread(path,0)
img = imutils.resize(img,400)
blur = cv2.GaussianBlur(img,(5,5),0) # filtro para eliminar el ruido
#en Umbra se guarda la imagen ya umbralizada
_,umbra = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#Encontrar un las areas y etiquetarlas 
_,Regiones = cv2.connectedComponents(umbra)

"""
Aquí se etiquetan los componentes conectados de la imagen umbralizada 
utilizando cv2.connectedComponents(). Esto devuelve el número total de componentes etiquetados 
y una matriz de etiquetas, donde cada píxel está etiquetado con el número de componente al que pertenece.
"""
label_hue = np.uint8(179*Regiones/np.max(Regiones))
blanco = 255*np.ones_like(label_hue)#creo una imagen en blanco de mi imagen 
#blank_ch se utiliza para crear dos canales de color 
#adicionales, uno para la saturación y otro para el valor, para crear una imagen de tres canales
img_etiquetas = cv2.merge([label_hue, blanco, blanco])



#Aquí se mapean las etiquetas de los componentes a valores de tonalidad de color en el rango
#de 0 a 179, que es el rango de tonalidad en OpenCV. Luego, se crea una imagen vacía del mismo 
#tamaño que la imagen etiquetada y se fusionan las matrices de valores de tonalidad de color y la imagen
#vacía en una sola imagen utilizando cv2.merge() la imagen resultante esta en CVT.
img_etiquetas[label_hue==0] = 0
#convierte la imagen de cvt a rgb
img_etiquetas = cv2.cvtColor(img_etiquetas, cv2.COLOR_HSV2BGR)

#encuentro la cantidad de contornos
contours = cv2.findContours(umbra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
#Genero los contornos  o bounding boxes
for i in contours:
    x,y,w,h = cv2.boundingRect(i)
    cv2.rectangle(img_etiquetas, (x, y), (x + w, y + h), (255,255,255), 4)
#Se establece el fondo de la imagen etiquetada (es decir, los píxeles con etiqueta 0) a gris .



cv2.imshow("Umbralizada",umbra)
cv2.imshow('Recuadros', img_etiquetas)
cv2.waitKey(0)
cv2.destroyAllWindows()
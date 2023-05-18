# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:26:20 2023

@author: cesar
"""
import cv2
import numpy as np
import imutils
#propiedades de la imagen
high=602 #alto de la imagen
width=700 #ancho

path = r"C:\Users\cesar\Vision\mano.jpg"
img = cv2.imread(path)
#img = imutils.resize(img,600)

#buscal el valor mas grande  y chico de rgb de la imagen
rojo =11
rojoc=255
azul =22
azulc=255
verde =12
verdec=255
#le asigno a las variables los valores maximos y minimow
for h in range(high):
    for w in range(width):
        if img[h,w,0] >= azul:
            azul= img[h,w,0]
        if img[h,w,0] <= azulc and img[h,w,0]!=0:
            azulc = img[h,w,0];   
for h in range(high):
    for w in range(width):
        if img[h,w,1] >= verde:
            verde= img[h,w,1]
        if img[h,w,1] <= verdec and img[h,w,1]!=0:
            verdec= img[h,w,1] 
for h in range(high):
    for w in range(width):
        if img[h,w,2] >= rojo:
            rojo= img[h,w,2]
        if img[h,w,2] < rojoc and img[h,w,2]!=0:
            rojoc= img[h,w,2]             


#parte de la umbralizacion canal B
for h in range(high):
    for w in range(width):
        if img[h,w,0] >= 15 and img[h,w,0] <=azul:
            img[h,w,0]=255
        else:
            img[h,w,0]=0
#parte de la umbralizacion canal G
for h in range(high):
    for w in range(width):
        if img[h,w,1] >= verdec and img[h,w,1] <=verde:
            img[h,w,1]=255
        else:
            img[h,w,1]=0
#parte de la umbralizacion canal R           
for h in range(high):
    for w in range(width):
        if img[h,w,2] >=rojoc and img[h,w,2] <=rojo:
            img[h,w,2] = 255
        else:
            img[h,w,2]=0
       
for h in range(high):
    for w in range (width):
        if img[h,w,0]==255 and img[h,w,1]==255 and img[h,w,2]==255:
            img[h,w,0]=255 
            img[h,w,1]=255
            img[h,w,2]=255
        else:    
            img[h,w,0]=0
            img[h,w,1]=0
            img[h,w,2]=0
  
    

cv2.imshow('filtro',img)
cv2.waitKey(0)
cv2.destroyAllWindows()            
            
            
            
            
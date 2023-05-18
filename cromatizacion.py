# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:27:42 2023

@author: cesar la mera verga
"""

import cv2
import numpy as np
import imutils
path = r"C:\Users\cesar\Vision\hand.jpeg"
img = cv2.imread(path)
img = imutils.resize(img,200)
img_n = np.copy(img)
img_n1 = np.copy(img)
img_n2 = np.copy(img)
img_64=np.asarray(img,dtype=np.float64)
img_brillo=np.asarray(img,dtype=np.float64)
img_brillo1=np.asarray(img,dtype=np.float64)
filas=img.shape[0]
columnas=img.shape[1]
colores=3
imgcrom=np.zeros([filas,columnas,colores])
imgcrom1=np.zeros([filas,columnas,colores])
imgcrom2=np.zeros([filas,columnas,colores])
for x in range(filas):
    for y in range (columnas):
        
        img_brillo[x,y,0]=img_brillo[x,y,0]*0.4
        img_brillo[x,y,1]=img_brillo[x,y,1]*0.4
        img_brillo[x,y,2]=img_brillo[x,y,2]*0.4 

        img_brillo1[x,y,0]=img_brillo1[x,y,0]*0.2
        img_brillo1[x,y,1]=img_brillo1[x,y,1]*0.2
        img_brillo1[x,y,2]=img_brillo1[x,y,2]*0.2     
            
            

#Cromatizacion de la imgaen original
for i in range(img.shape[0]):
    for j in range(img.shape[1]):      
        for k in range(img.shape[2]):
            imgcrom[i,j,k]=((img_64[i,j,k])/(img_64[i,j,0]+img_64[i,j,1]+img_64[i,j,2]))
            
#cromatizacion de la imgagen con la imagen con imajen baja

for i in range(img.shape[0]):
    for j in range(img.shape[1]):      
        for k in range(img.shape[2]):
            imgcrom1[i,j,k]=((img_brillo[i,j,k])/(img_brillo[i,j,0]+img_brillo[i,j,1]+img_brillo[i,j,2]))
            

for i in range(img.shape[0]):
    for j in range(img.shape[1]):      
        for k in range(img.shape[2]):
            imgcrom2[i,j,k]=((img_brillo1[i,j,k])/(img_brillo1[i,j,0]+img_brillo1[i,j,1]+img_brillo1[i,j,2]))
                       
img_brillo=np.asarray(img_brillo,dtype=np.uint8)            
img_brillo1=np.asarray(img_brillo1,dtype=np.uint8)
  


#Multiplicar el array por 255 para poder umbralizarlo
 
imgcrom=np.asarray(np.multiply(imgcrom, 255),dtype=np.uint8)
imgcrom1=np.asarray(np.multiply(imgcrom1, 255),dtype=np.uint8)
imgcrom2=np.asarray(np.multiply(imgcrom2, 255),dtype=np.uint8)
#Umbralizarla
for x in range(img.shape[0]):
    for y in range(img.shape[1]): 
        if(79 <imgcrom[x,y,0]<87):
            img_n[x,y]=0
            img_n1[x,y]=0
            img_n2[x,y]=0
        else:
            img_n[x,y]=255
            img_n1[x,y]=255
            img_n2[x,y]=255

cv2.imshow("imagen",cv2.vconcat([cv2.hconcat([img,img_brillo,img_brillo1]),cv2.hconcat([imgcrom,imgcrom1,imgcrom2]),cv2.hconcat([img_n,img_n1,img_n2])]))
cv2.waitKey(0)
cv2.destroyAllWindows()
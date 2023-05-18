
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:40:42 2023

@author: cesar
"""

import cv2
import numpy as np
import imutils
path = r"C:\Users\cesar\Vision\hand.jpeg"

img = cv2.imread(path)
img = imutils.resize(img,150)
img_n = np.copy(img)
img_azul = np.copy(img)
img_rojo = np.copy(img)
img_verde = np.copy(img)
#whitepatch
img_W = np.copy(img)
img_azulW = np.copy(img_azul)
img_rojoW = np.copy(img_rojo)
img_verdeW = np.copy(img_verde)
#img_baja= np.round(np.multiply(img[:,:,0],1.2))

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_n[x,y]=255
        else:
            img_n[x,y]=0
            #crear una imagen con una saturacion en el canal del azul
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
            img_azul[x,y,1]=img_azul[x,y,1]*0.7
            img_azul[x,y,2]=img_azul[x,y,2]*0.7

            img_rojo[x,y,0]=img_rojo[x,y,0]*0.7
            img_rojo[x,y,1]=img_rojo[x,y,1]*0.7
            
            img_verde[x,y,0]=img_verde[x,y,0]*0.7
            img_verde[x,y,2]=img_verde[x,y,2]*0.7

#algoritmo de White-patch
#azul

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #Imagen normal
        img_W[x,y,0]=(255/(np.max(img[:,:,0])))*(img[x,y,0])
        img_W[x,y,1]=(255/(np.max(img[:,:,1])))*(img[x,y,1])
        img_W[x,y,2]=(255/(np.max(img[:,:,2])))*(img[x,y,2])
        #Azul
        img_azulW[x,y,0]=(255/(np.max(img_azul[:,:,0])))*(img_azul[x,y,0])
        img_azulW[x,y,1]=(255/(np.max(img_azul[:,:,1])))*(img_azul[x,y,1])
        img_azulW[x,y,2]=(255/(np.max(img_azul[:,:,2])))*(img_azul[x,y,2])
        #Verda
        img_verdeW[x,y,0]=(255/(np.max(img_verde[:,:,0])))*(img_verde[x,y,0])
        img_verdeW[x,y,1]=(255/(np.max(img_verde[:,:,1])))*(img_verde[x,y,1])
        img_verdeW[x,y,2]=(255/(np.max(img_verde[:,:,2])))*(img_verde[x,y,2])
        #Rojo
        img_rojoW[x,y,0]=(255/(np.max(img_rojo[:,:,0])))*(img_rojo[x,y,0])
        img_rojoW[x,y,1]=(255/(np.max(img_rojo[:,:,1])))*(img_rojo[x,y,1])
        img_rojoW[x,y,2]=(255/(np.max(img_rojo[:,:,2])))*(img_rojo[x,y,2])
#Umbralizacion a todas las imagenes
img_azul_U = np.copy(img_azul)
img_rojo_U = np.copy(img_rojo)
img_verde_U = np.copy(img_verde)
#whitepatch
img_W_U = np.copy(img_W)
img_azulW_U = np.copy(img_azulW)
img_rojoW_U = np.copy(img_rojoW)
img_verdeW_U = np.copy(img_verdeW)
#imagen original
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_n[x,y]=255
        else:
            img_n[x,y]=0
#imagen Withe-Patch
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img_W[x,y,0]<150) and (81<img_W[x,y,1]<195) and (53<img_W[x,y,2]<228)):
            img_W_U[x,y]=255
        else:
            img_W_U[x,y]=0
#imagen azul
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_azul_U[x,y]=255
        else:
            img_azul_U[x,y]=0
#imagen azul White-Patch
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_azulW_U[x,y]=255
        else:
            img_azulW_U[x,y]=0
#imagen verde
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_verde_U[x,y]=255
        else:
            img_verde_U[x,y]=0 
#imagen verde White-Patch
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_verdeW_U[x,y]=255
        else:
            img_verdeW_U[x,y]=0
#imagen rojo
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_rojo_U[x,y]=255
        else:
            img_rojo_U[x,y]=0
#imagen rojo White-Patch
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        #       B                       G                           R        
        if((45 <img[x,y,0]<120) and (81<img[x,y,1]<240) and (53<img[x,y,2]<198)):
            img_rojoW_U[x,y]=255
        else:
            img_rojoW_U[x,y]=0
#imagen cromatizada
img_h1 = cv2.hconcat([img,img_azul,img_rojo,img_verde])
img_h2 = cv2.hconcat([img_W,img_azulW,img_verdeW,img_rojoW])
img_v = cv2.vconcat([img_h1,img_h2])

img_h3 = cv2.hconcat([img_n,img_azul_U,img_rojo_U,img_verde_U])
img_h4 = cv2.hconcat([img_W_U,img_azulW_U,img_verdeW_U,img_rojoW_U])
img_v2 = cv2.vconcat([img_h3,img_h4])
img_v3 = cv2.vconcat([img_v,img_v2])                   
cv2.imshow('imgenes',img_v3)
#cv2.imshow('imgenes Umbralizadas',img_v2)
cv2.waitKey(0)
    
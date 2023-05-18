# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:44:52 2023

@author: cesar
"""

import cv2

capture = cv2.VideoCapture(0)
#objeto salida, contiene los parámetros para crear el video
salida = cv2.VideoWriter('webCam.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))


while (True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    #Usar write para GUARDAR el video
    salida.write(frame)
    if (cv2.waitKey(1) == ord('s')):
        break

salida.release()
capture.release()
cv2.destroyAllWindows()
capture = cv2.VideoCapture('webCam.avi')

while (capture.isOpened()):
    ret, frame = capture.read()
    if (ret == True):
        cv2.imshow("gato0", frame)
        if (cv2.waitKey(24) == ord('s')):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()

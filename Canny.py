import numpy as np
import cv2
import imutils

#Necesario para que funcione el trackbar
def nothing(x):
	pass

#Creo una ventana para almacenar los dos tracksbar
cv2.namedWindow('Parametros')
#creo dos trackbars en la ventana 'Parametros'  con el nombre Threshold 1 y Threshold 2'

cv2.createTrackbar('Threshold 1','Parametros',1,255,nothing)
cv2.createTrackbar('Threshold 2','Parametros',0,255,nothing)

#creo un loop para que los trackbars funcionen
while(1):

    #le asingno el valor a la variable dependiendo de los trackbar 
    threshold1 = cv2.getTrackbarPos('Threshold 1','Parametros')
    
    threshold2 = cv2.getTrackbarPos('Threshold 2','Parametros')

    img = cv2.imread('Fichas.jpg', cv2.IMREAD_GRAYSCALE)
    imge = cv2.imread('Fichas.jpg')
    img = imutils.resize(img,400)
    imge = imutils.resize(imge,400)
    #Aplico la funcion de canny con los valores de los trackbar
    bordes = cv2.Canny(img,threshold1,threshold2)
    #concateno las dos imagenes la imagen original y la que detecta los bordes
    imagen=cv2.hconcat([img,bordes]) 
    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)  
    #cv2.imshow('canny',imagen)

    ##############################################################Contador#####################################################################
    contor, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imge, contor, -1, (255,0,5), 2)
    texto = 'Contornos encontrados: '+ str(len(contor))
    cv2.putText(imge, texto, (10,20), cv2.QT_FONT_NORMAL, 0.7,(255, 0, 0), 2)
    imge=cv2.hconcat([imge,imagen]) 
    cv2.imshow('Contador',imge)

# Condicional para romper el ciclo del While
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  #Si la tecla es "esc" rompo el ciclo 
        break
cv2.destroyAllWindows()   # cierra toda las ventanas
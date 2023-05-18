
from deepface import DeepFace
"""
# Cargar imagen de referencia
img_ref = cv2.imread("Cara2.jpg")

#print(DeepFace.verify(img1_path = "Cara.jpg", img2_path = "Cara2.jpg"))
# Función para detectar y extraer el rostro de una imagen

img_ref = DeepFace.extract_faces(img_ref, detector_backend='opencv',align=False)
img_ref = cv2.cvtColor(img_ref[0]["face"], cv2.COLOR_BGR2RGB)

cv2.imshow("HOLA",img_ref)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Iniciar la webcam
cap = cv2.VideoCapture(0)


while True:
    # Leer un frame de la webcam
    ret, frame = cap.read()

    # Extraer el rostro del frame
    rostro = DeepFace.extract_faces(frame, detector_backend='opencv',align=False,enforce_detection=True)
    x, y, w, h = rostro[0]["facial_area"]

    rostro1 = cv2.cvtColor(rostro[0]["face"], cv2.COLOR_BGR2RGB)
    #cv2.imshow("HsOLA",rostro)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
   


    if rostro is not None:
        # Comparar el rostro extraído con la imagen de referencia
        resultado = DeepFace.verify(rostro1, img_ref)

        # Obtener el resultado de la comparación
        verificado = resultado["verified"]

        # Si el rostro es verificado, dibujar un cuadro verde alrededor del rostro
        color = (0, 255, 0) if verificado else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Mostrar el frame en una ventana
    cv2.imshow("Reconocimiento facial", frame)

    # Esperar por una tecla para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la webcam y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
"""
DeepFace.stream(db_path = "C:\\Users\\cesar\\Vision\\database",time_threshold=4,frame_threshold=100)

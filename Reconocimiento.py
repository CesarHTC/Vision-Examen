import cv2
from deepface import DeepFace

# Cargar imagen de referencia
referencia = cv2.imread("Cara.jpg")

# Función para detectar y marcar rostros
def detectar_rostros(imagen):
    demography = DeepFace.analyze(img_path=imagen, actions=['demography'], enforce_detection=False)
    resultados = demography["instance"]
    if not resultados:
        print("No se encontraron rostros en la imagen.")
        return imagen
    for resultado in resultados:
        x, y, w, h = resultado["box"]
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return imagen


# Iniciar cámara
captura = cv2.VideoCapture(0)

while True:
    # Leer imagen de la cámara
    ret, frame = captura.read()
    if not ret:
        break

    # Redimensionar imagen para un análisis más rápido
    frame = cv2.resize(frame, (640, 360))

    # Detectar y marcar rostros
    frame = detectar_rostros(frame)

    # Comparar rostro detectado con base de datos de referencia
    resultados = DeepFace.verify(referencia, frame)["verified"]
    if resultados:
        cv2.putText(frame, "Rostro reconocido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Rostro no reconocido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar imagen en una ventana
    cv2.imshow("Reconocimiento facial", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()

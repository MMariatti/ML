import cv2

capturar_video = cv2.VideoCapture(0)

if not capturar_video.isOpened():
    print('No se encontro una camara')
    exit()

while capturar_video.isOpened():
    tipoCamara, camara = capturar_video.read()  # Hacemos que captura la imagen en la camara
    camara_grises = cv2.cvtColor(camara, cv2.COLOR_BGR2GRAY)  # Convertimos la imagen capturada a grises
    cv2.imshow('En vivo', camara_grises)   # Mostramos la imagen en grises
    if cv2.waitKey(1) == ord('/'):
        break
capturar_video.release()
cv2.destroyAllWindows()

from cv2 import cv2
import numpy as np


def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]


def alineamiento(imagen, ancho, alto):
    imagen_alineada = None
    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    tipo_umbral, umbral = cv2.threshold(grises, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('Umbral', umbral)
    contorno = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key=cv2.contourArea, reverse=True)[:1]
    for c in contorno:
        epsilon = 0.01 * cv2.arcLength(c, True)
        aproximacion = cv2.approxPolyDP(c, epsilon, closed=True)
        if len(aproximacion) == 4:
            puntos = ordenar_puntos(aproximacion)
            punto_centrado_1 = np.float32(puntos)
            punto_centrado_2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(punto_centrado_1, punto_centrado_2)
            imagen_alineada = cv2.warpPerspective(imagen, M, (ancho, alto))
    return imagen_alineada


captura_video = cv2.VideoCapture(0)
while True:
    tipo_camara, camara = captura_video.read()
    if not tipo_camara:
        break
    imagen_A6 = alineamiento(camara, ancho=677, alto=480)
    if imagen_A6 is not None:
        puntos = []
        imagen_grises = cv2.cvtColor(imagen_A6, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_grises, (5, 5), 1)
        _, umbral2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cv2.imshow('Umbral', umbral2)
        contorno2 = cv2.findContours(umbral2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6, contorno2, -1, (255, 0, 0), 2)
        suma_monedas_2p = 0.0  # Cantidad de monedas 2 pesos
        suma_monedas_5p = 0.0  # Cantidad de monedas 5 pesos
        for c_2 in contorno2:
            area = cv2.contourArea(c_2)  # de acuerdo al area que tenga el circulo vamos a saber que moneda es
            Momentos = cv2.moments(c_2)
            if Momentos['m00'] == 0:
                Momentos['m00'] = 1.0
            x = int(Momentos['m10']/Momentos['m00'])
            y = int(Momentos['m01']/Momentos['m00'])

            if 9300 > area > 8500:  # Depende de la camara, no va a ser exacto. V.E 9082 px
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, '$5 ARS', (x, y), font, 0.75, (0, 0, 255), 2)
                suma_monedas_5p += 5

            if 8400 > area > 6500:  # Depende de la camara, no va a ser exacto. V.E 7936 px
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, '$2 ARS', (x, y), font, 0.75, (0, 0, 255), 2)
                suma_monedas_2p += 2

        total = suma_monedas_5p+suma_monedas_2p
        print('Sumatoria total en pesos:', total)

        cv2.imshow('Imagen A6', imagen_A6)
        cv2.imshow('Camara', camara)
        if cv2.waitKey(1) == ord('/'):
            break
captura_video.release()
cv2.destroyAllWindows()


from cv2 import cv2

imagen = cv2.imread('contorno.jpg')
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_, umbral = cv2.threshold(grises, 50, 255, cv2.THRESH_BINARY)
contorno, jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen, contorno, -1, (23, 43, 124), 3)

# Mostrar Imagenes
cv2.imshow('Imagen original', imagen)

'''
cv2.imshow('Imagen en grises', grises)
cv2.imshow('Imagen como umbral', umbral)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()

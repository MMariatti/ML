import cv2
import numpy as np

valor_gauss = 3
valor_kernel = 3
original = cv2.imread('monedas.jpg')
grises = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(grises, (valor_gauss, valor_gauss), 0)  # Tiene que ser una matriz cuadrada
canny = cv2.Canny(gauss, 60, 100)  # Mas grande la matriz, mas borrosa la foto

kernel = np.ones((valor_kernel, valor_kernel), np.uint8)
cierre_contornos = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
contornos, jerarquias = cv2.findContours(cierre_contornos.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('Monedas encontradas : {}'.format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0, 24, 233), 3)

# mostrar resultados
'''
cv2.imshow('Grises', grises)
cv2.imshow('Gauss', gauss)
cv2.imshow('Canny', canny)
cv2.imshow('Cierre de contornos', cierre_contornos)
'''
cv2.imshow('Original', original)
cv2.waitKey(0)

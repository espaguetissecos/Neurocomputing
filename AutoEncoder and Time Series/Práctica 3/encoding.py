from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorAdaline
from Clasificador import AutoEncoder
import numpy as np
from parsing import *
import sys

import pylab as pl


parserAlfabeto("alfabeto_dat.txt")
parserAlfabetoRuidoTest("alfabeto_dat.txt", 1)
############################################################################
## CLASIFICACION DE PROBLEMA 2 NO ETIQUETADOS
############################################################################
dat1 = Datos("alfabeto.txt", True)

#dat2 = Datos("alfabeto.txt", True)
dat2 = Datos("alfabeto.txt", True)


datos_train = dat1.datos
datos_test = dat2.datos

############################################################################
## ENCODER
############################################################################

clas2 = AutoEncoder(10, 0.15)
clas2.entrenamiento(datos_train, 35, "pesos_no_etiquetados.txt")
pred = clas2.clasifica(datos_test)
print str((clas2.error(datos_test[:, (70- 35):], pred))) + ": Promedio de pixeles fallados por letra"
letras = clas2.letras(datos_test[:, (70- 35):], pred)
print str(letras.__len__()) + " letras falladas"
print str(datos_test.__len__() - letras.__len__()) + " letras acertadas"
print str(datos_test.__len__()) + " letras totales"



# if (letras.__len__() > 0):
#     print "Las letras falladas son: "
#     print letras

# file = open('predicciones_nnet.txt', 'w')
# file.write("PrediccionNoEtiquetada\n")
# for i in range(0, pred.__len__()):
#     for j in range(0, 2):
#         file.write(str(int(pred[i][j])))
#         file.write(" ")
#     file.write("\n")
# file.close()

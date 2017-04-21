from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorAdaline
from Clasificador import ClasificadorPerceptron
import pylab as pl

############################################################################
## CLASIFICACION DE PROBLEMA 2 NO ETIQUETADOS
############################################################################


dat1 = Datos("problema_real2.txt", True)
dat2 = Datos("problema_real2_no_etiquetados.txt", True)

datos_train = dat1.datos
datos_test = dat2.datos


############################################################################
## PERCEPTRON
############################################################################

clas1 = ClasificadorPerceptron()
clas1.entrenamiento(datos_train)
pred = clas1.clasifica(datos_test)

file = open('predicciones_perceptron_no_clasificados.txt', 'w')
file.write("PrediccionNoEtiquetada")
for i in range(0, pred.__len__()):
    file.write(str(pred[i]))
file.close()

############################################################################
## ADALINE
############################################################################

clas2 = ClasificadorAdaline()
clas2.entrenamiento(datos_train)
pred = clas2.clasifica(datos_test)

file = open('predicciones_adaline_no_clasificados.txt', 'w')
file.write("PrediccionNoEtiquetada")
for i in range(0, pred.__len__()):
    file.write(str(pred[i]))
file.close()


from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorAdaline
from Clasificador import ClasificadorPerceptronMulticapa
import pylab as pl
import sys


if ( len(sys.argv) < 6):
    print("Instrucciones de ejecucion:")
    print("python multicapa.py <ficheroEntrada> <ficheroSalida> <porcentajeEntrenamiento> <tasaAprendizaje> <numNeuronasOculta>")
    pass

fichero_entrada = str(sys.argv[1])
fichero_salida = str(sys.argv[2])
porcentaje = float(sys.argv[3])
tasaAprendizaje = float(sys.argv[4])
numNeuronas = int(sys.argv[5])

dat = Datos(fichero_entrada, True)
val =  ValidacionSimple(porcentaje)


clas1 = ClasificadorPerceptronMulticapa(numNeuronas, tasaAprendizaje)
errores1 = clas1.validacion(val, dat, clas1, fichero_salida)

print "La tasa de acierto en la particion de test en el Perceptron Multicapa es: "
print str((1 - errores1[0]) * 100) + " %"
print ""

#
# pl.figure(0)
# for d in dat.datos:
#     if d[2]==-1:
#         pl.scatter(d[0], d[1], color='r')
#     else:
#         pl.scatter(d[0], d[1], color='g')
# pl.xlabel('Eje X')
# pl.ylabel('Eje Y')
# pl.show()
#

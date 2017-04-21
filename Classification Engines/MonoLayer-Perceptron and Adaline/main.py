from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorAdaline
from Clasificador import ClasificadorPerceptron
import pylab as pl



print "Introduzca el nombre del fichero de entrada (ej: problema_real1.txt):"
nombre_fichero = raw_input()

print "Introduzca el porcentaje de particionado para el entrenamiento (para 1-1, introducir 1, para 2/3-1/3 introducir 0.66):"
porcentaje = raw_input()

while (float(porcentaje) <= 0.09 or float(porcentaje) > 1):
    print "Porcentaje de particionado invalido."
    print "Introduzca de nuevo el porcentaje de particionado para el entrenamiento (para 1-1, introducir 1, para 2/3-1/3 introducir 0.66):"
    porcentaje = raw_input()

print ""
print "Generando salida del Perceptron ..."
print ""

dat = Datos(nombre_fichero, True)
val =  ValidacionSimple(float(porcentaje))


clas1 = ClasificadorPerceptron()
errores1 = clas1.validacion(val, dat, clas1)

print "El error en la particion de test en el Perceptron es: "
print errores1[0]
print ""

print ""
print "Generando salida del Adaline ..."
print ""

clas2 = ClasificadorAdaline()
errores2 = clas2.validacion(val, dat, clas2)

print "El error en la particion de test en el Adaline es: "
print errores2[0]
print ""



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
#

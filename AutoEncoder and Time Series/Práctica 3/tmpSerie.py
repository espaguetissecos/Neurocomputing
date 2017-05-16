from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorAdaline
from Clasificador import SerieTemporal
from parsing import adaptaficheroserie
import pylab as pl
import sys


if ( len(sys.argv) < 6):
    print("Instrucciones de ejecucion:")
    print("python main.py <ficheroEntrada> <ficheroSalida> <porcentajeEntrenamiento> <tasaAprendizaje> <numNeuronasOculta>")
    sys.exit()


fichero_entrada = str(sys.argv[1])
fichero_salida = str(sys.argv[2])
porcentaje = float(sys.argv[3])
tasaAprendizaje = float(sys.argv[4])
numNeuronas = int(sys.argv[5])
Na = 5
Ns = 1



adaptaficheroserie(fichero_entrada,fichero_salida, Na, Ns)

dat = Datos(fichero_salida, True)
val =  ValidacionSimple(porcentaje)


clas1 = SerieTemporal(numNeuronas, tasaAprendizaje)
errores1 = clas1.validacion(val, dat, clas1, random=False,salida="pesos.txt", recursivo=False)

print "El ECM en la particion test del clasificador es: "
print str((errores1[0]))
print ""
import numpy as np

###
# Practica realizada por Francisco Andreu y Javier Cela
###

class Datos(object):
    datos = np.array(())
    clase_reg = np.array(())
    # Lista de diccionarios. Uno por cada atributo.

    def __init__(self, nombreFichero, sup):
        with open(nombreFichero, "r") as f:

            lista_aux = []
            iterlen = lambda it: sum(1 for _ in it)
            lineas = iterlen(file(nombreFichero)) - 1
            num_atributos, num_clase = f.readline().split(' ')
            array_aux = np.array(()).astype(int)

            for j in range(0, lineas):
                array_aux = np.array(())
                linea_aux = f.readline().split()#[s.strip() for s in f.readline().split(' ') if s]
                for i in range(0, int(num_atributos)):
                    array_aux = np.append(array_aux, float(linea_aux[i]))
                array_aux = np.append(array_aux, self.retBipolarClass(linea_aux[int(num_atributos):(int(num_atributos)+int(num_clase))]))
                if (j==0):
                    self.datos = array_aux
                else:
                    self.datos = np.vstack([self.datos, np.copy(array_aux)])


    def retBipolarClass(self, clase):
        if (clase.__eq__(['0', '1'])):
            return 1
        else:
            return -1

    def extraeDatos(self, idx):
        return self.datos[idx]

import numpy as np
from random import randint

def parserAlfabeto(nombreFichero):
    with open(nombreFichero, "r") as f:
        lista_aux = []
        iterlen = lambda it: sum(1 for _ in it)
        lineas = iterlen(file(nombreFichero)) - 1
        array_aux = np.array(()).astype(int)
        f_out = open("alfabeto.txt", "w")
        contador = 0
        array = []
        f_out.write("35 35\n")
        contador2 = 0
        for j in range(0, lineas):
            array_aux = np.array(())
            linea_aux = f.readline().split()
            for l in range(0, len(linea_aux)):
                if linea_aux[l] == "0":
                    linea_aux[l] = "-1"

            if (linea_aux.__len__() > 0):
                if (linea_aux[0].__eq__("//")):
                    contador = 0
                    continue

                for i in range(0, linea_aux.__len__()):
                    f_out.write(linea_aux[i])
                    array.append(linea_aux[i])
                    f_out.write(" ")
                contador+=1
                if (7 == contador):
                    for i in range(0, 35):
                        f_out.write(array[i])
                        if (i != 35):
                            f_out.write(" ")
                    contador2 += 1
                    array = []
                    f_out.write("\n")

def parserAlfabetoRuidoTest(nombreFichero, n):
    with open(nombreFichero, "r") as f:
        lista_aux = []
        iterlen = lambda it: sum(1 for _ in it)
        lineas = iterlen(file(nombreFichero)) - 1
        array_aux = np.array(()).astype(int)
        f_out = open("alfabeto_ruido.txt", "w")
        contador = 0
        array = []
        f_out.write("35 35\n")

        for j in range(0, lineas):
            array_aux = np.array(())
            linea_aux = f.readline().split()
            for l in range(0, len(linea_aux)):
                if linea_aux[l] == "0":
                    linea_aux[l] = "-1"

            if (linea_aux.__len__() > 0):
                if (linea_aux[0].__eq__("//")):
                    contador = 0
                    continue

                for i in range(0, linea_aux.__len__()):
                    array.append(linea_aux[i])
                contador+=1
                if (7 == contador):
                    for l in range(0, 10):
                        array_aux = array[:]
                        random_list = []
                        for i in range(0, n):
                            random_list.append(randint(0,34))

                        for i in range(0, random_list.__len__()):
                            array_aux[random_list[i]] = switchValue(array_aux[random_list[i]])

                        for i in range(0, 35):
                            f_out.write(array_aux[i])
                            if (i != 35):
                                f_out.write(" ")
                        for i in range(0, 35):
                            f_out.write(array[i])
                            if (i != 35):
                                f_out.write(" ")
                        array_aux = []
                        f_out.write("\n")
                    array = []

def parserAlfabetoRuidoTrain(nombreFichero, n):
    with open(nombreFichero, "r") as f:
        lista_aux = []
        iterlen = lambda it: sum(1 for _ in it)
        lineas = iterlen(file(nombreFichero)) - 1
        array_aux = np.array(()).astype(int)
        f_out = open("alfabeto_ruido_train.txt", "w")
        contador = 0
        array = []
        f_out.write("35 35\n")

        for j in range(0, lineas):
            array_aux = np.array(())
            linea_aux = f.readline().split()
            for l in range(0, len(linea_aux)):
                if linea_aux[l] == "0":
                    linea_aux[l] = "-1"

            if (linea_aux.__len__() > 0):
                if (linea_aux[0].__eq__("//")):
                    contador = 0
                    continue

                for i in range(0, linea_aux.__len__()):
                    array.append(linea_aux[i])
                contador+=1
                if (7 == contador):

                    for i in range(0, 35):
                        f_out.write(array[i])
                        if (i != 35):
                            f_out.write(" ")
                    for i in range(0, 35):
                        f_out.write(array[i])
                        if (i != 35):
                            f_out.write(" ")
                    f_out.write("\n")

                    for l in range(0, 10):
                        array_aux = array[:]
                        random_list = []
                        for i in range(0, n):
                            random_list.append(randint(0,34))

                        for i in range(0, random_list.__len__()):
                            array_aux[random_list[i]] = switchValue(array_aux[random_list[i]])

                        for i in range(0, 35):
                            f_out.write(array_aux[i])
                            if (i != 35):
                                f_out.write(" ")
                        for i in range(0, 35):
                            f_out.write(array[i])
                            if (i != 35):
                                f_out.write(" ")
                        array_aux = []
                        f_out.write("\n")
                    array = []


def switchValue(dato):
    if dato == '1':
        dato = '-1'
    else:
        dato = '1'
    return dato

def adaptaficheroserie(entrada, salida, Na, Ns):
    f_in = open(entrada, "r")
    datos = f_in.read().split('\n')

    f_out = open(salida, "w")
    f_out.write(str(Na) + " " + str(Ns))
    for i in range(0, len(datos) - Na - Ns):
        f_out.write('\n')
        for j in range(0, Na):
            f_out.write(datos[i + j] + " ")
        for j in range(0, Ns - 1):
            f_out.write(datos[i + Na + j] + " ")
        f_out.write(datos[i + Na + Ns - 1])

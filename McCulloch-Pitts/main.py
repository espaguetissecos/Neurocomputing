import sys

from Neurona import Neurona

"""Anyadimos neuronas"""

""" Primera capa """
neurona_entrada1 = Neurona()
neurona_entrada2 = Neurona()
neurona_entrada3 = Neurona()
""" Segunda capa """
neuronaZ1 = Neurona([neurona_entrada1], [2], 2)
neuronaZ2 = Neurona([neurona_entrada2], [2], 2)
neuronaZ3 = Neurona([neurona_entrada3], [2], 2)
""" Tercera capa """
neuronaV1 = Neurona([neuronaZ1, neurona_entrada2], [1, 1], 2)
neuronaV2 = Neurona([neuronaZ2, neurona_entrada1], [1, 1], 2)
neuronaV3 = Neurona([neuronaZ2, neurona_entrada3], [1, 1], 2)
neuronaV4 = Neurona([neuronaZ3, neurona_entrada2], [1, 1], 2)
neuronaV5 = Neurona([neuronaZ3, neurona_entrada1], [1, 1], 2)
neuronaV6 = Neurona([neuronaZ1, neurona_entrada3], [1, 1], 2)

""" Capa final """
neuronaY1 = Neurona([neuronaV2, neuronaV4, neuronaV6], [2, 2, 2], 2)
neuronaY2 = Neurona([neuronaV1, neuronaV3, neuronaV5], [2, 2, 2], 2)

ficheroEntrada = open("input.txt", 'r')
ficheroSalida = open("output.txt", 'w')

for linea in ficheroEntrada:
    i=0
    input = linea.split(" ")
    """Leemos la linea i del fichero"""
    for neurona_ini in [neurona_entrada1, neurona_entrada2, neurona_entrada3]:
        neurona_ini.activada = (input[i])
        i += 1

    neuronaY2.propagar()
    neuronaY1.propagar()
    for neurona in [neuronaV1, neuronaV2, neuronaV3, neuronaV4, neuronaV5, neuronaV6]:
        neurona.propagar()

    for neurona in [neuronaZ1, neuronaZ2, neuronaZ3]:
        neurona.propagar()
    ficheroSalida.write(str(neuronaY1.activada) + " " + str(neuronaY2.activada) + "\n")

"""Propagamos otra vez para la ultima salida"""
neuronaY2.propagar()
neuronaY1.propagar()
ficheroSalida.write(str(neuronaY1.activada) + " " + str(neuronaY2.activada) + "\n")
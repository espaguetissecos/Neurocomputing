from abc import ABCMeta,abstractmethod
from numpy import random
import numpy as np
from Datos import Datos


class Particion():
  
  indicesTrain=[]
  indicesTest=[]
  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  nombreEstrategia="null"
  numeroParticiones=0
  particiones=[]
  
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None, porcentaje=0.5):



    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)

  def __init__(self, particionado=0.5):
    self.particionado = particionado

  def creaParticiones(self,datos,seed=None):

    random.seed(seed)

    self.nombreEstrategia = "Simple"
    self.numeroParticiones = 2
    particion_aux = Particion()
    array_aux = np.array(())

    array_aux = random.permutation((len(datos.datos)))
    if (self.particionado == 1):
      particion_aux.indicesTrain = array_aux[0:int((len(array_aux)))]
      particion_aux.indicesTest = array_aux[0:int((len(array_aux)))]

    else:
      particion_aux.indicesTrain = array_aux[0:int((len(array_aux)*self.particionado))]
      particion_aux.indicesTest = array_aux[int((len(array_aux)*self.particionado)):len(array_aux)]


    self.particiones.append(particion_aux)
    pass

      

#####################################################################################################
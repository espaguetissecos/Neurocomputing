from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm
from math import log10

class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta

  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass


  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
      nAciertos = 0
      nFallos = 0
      lineas = datos.shape[0]
      lineas2 = pred.shape[0]
      if (lineas!=lineas2): return -1
      for i in range(0, lineas- 1):
          if (datos[i] == pred[i]):
              nAciertos = nAciertos + 1
          else:
              nFallos = nFallos + 1
      return nFallos / float(nFallos + nAciertos)

    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador, seed=None):
       particionado.creaParticiones(dataset)

       datos_train = np.array(())
       datos_test = np.array(())

       errores = []

       if particionado.nombreEstrategia == 'Simple':
         datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)
         datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)

         clasificador.entrenamiento(datos_train)
         pred = clasificador.clasifica(datos_test)

         errores.append(self.error(datos_test[:, datos_test.shape[1]-1], pred))

       ## Esto no hace falta
       elif particionado.nombreEstrategia == 'Cruzada':
         for i in range(0, particionado.numeroParticiones):
           datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
           datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)
           clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

           pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

           errores.append(self.error(datos_test[:, datos_test.shape[1]-1], pred))

       return errores



#############################################################################
############################### PERCEPTRON ##################################
#############################################################################


class ClasificadorPerceptron(Clasificador):

    mayoritaria = 0
    n_epocas = 1000
    vector_final = np.array(())
    peso_bias_final=0
    # TODO: implementar
    bias = 1
    tasa_aprendizaje = 0.2
    frontera = 0

    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):


        nrows, ncols = datostrain.shape
        matriz_pesos = np.array(())
        incremento_pesos_vector_actual = np.array(())
        pesos_vector_actual = np.array(())
        vector_error = np.array(())

        incremento_bias = 0
        peso_bias = 0

        y_in = 0
        y = 0

        for i in range(0, ncols - 1):
            incremento_pesos_vector_actual = np.append(incremento_pesos_vector_actual, 0)
            pesos_vector_actual = np.append(pesos_vector_actual, 0)

        matriz_pesos = np.append(matriz_pesos, np.copy(pesos_vector_actual))

        for k in range(0, self.n_epocas):
            contador = 0
            for i in range(0, nrows):
                y_in = 0
                for j in range(0, ncols - 1):
                    y_in += datostrain[i, j] * pesos_vector_actual[j]
                y_in += peso_bias

                if (y_in < -self.frontera):
                    y = -1
                elif (y_in > self.frontera):
                    y = 1
                else:
                    y = 0

                for j in range(0, ncols - 1):
                    if (y != datostrain[i, ncols - 1]):
                        incremento_pesos_vector_actual[j] = datostrain[i, j] * datostrain[i, ncols - 1] * self.tasa_aprendizaje  # Incremento de Wi = Xi*T
                        incremento_bias = datostrain[i, ncols - 1] * self.tasa_aprendizaje * self.bias
                    else:
                        incremento_pesos_vector_actual[j] = 0
                        incremento_bias = 0
                    pesos_vector_actual[j] += incremento_pesos_vector_actual[j]  # Wi = Wi + Incremento de Wi

                peso_bias += incremento_bias

                if (y != datostrain[i, ncols - 1]):
                    contador += 1

            vector_error = np.append(vector_error, float(contador) / float(nrows))

        print "Numero de iteraciones de entrenamiento en el Perceptron:"
        print self.n_epocas
        print ""

        print "La media de error en entrenamiento en el Perceptron es:"
        print np.average(vector_error)
        print ""

        self.vector_final = pesos_vector_actual
        self.peso_bias_final = peso_bias

        pass



        # TODO: implementar

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):
        # Asignar la clase mayoritaria a todos los datos

        datospred = np.copy(datostest)
        num_filas, num_cols = datostest.shape
        for i in range(0, num_filas):
            y_in = 0
            for j in range(0, num_cols - 1):
                y_in += datospred[i, j] * self.vector_final[j]
            y_in += self.peso_bias_final

            if (y_in < -self.frontera):
                y = -1
            elif (y_in > self.frontera):
                y = 1
            else:
                y = 0
            datospred[i, num_cols - 1] = y


        return datospred[:, num_cols - 1]

    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas = datos.shape[0]
        lineas2 = pred.shape[0]

        file = open('predicciones_perceptron.txt', 'w')
        file.write("Dato\tPrediccion\n")

        if (lineas != lineas2): return -1
        for i in range(0, lineas):
            file.write(str(datos[i]) + "\t" + str(pred[i]) + "\n")
            if (datos[i] == pred[i]):
                nAciertos = nAciertos + 1
            else:
                nFallos = nFallos + 1
        file.close()
        return nFallos / float(nFallos + nAciertos)


#############################################################################
################################ ADALINE ####################################
#############################################################################


class ClasificadorAdaline(Clasificador):

    mayoritaria = 0
    n_epocas = 1000
    vector_final = np.array(())
    peso_bias_final=0
    # TODO: implementar
    bias = 1
    tasa_aprendizaje = 0.2
    frontera = 0

    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):
        diferencia_y_t = 0
        nrows, ncols = datostrain.shape
        datostrain_aux = np.copy(datostrain[:, : - 1])
        matriz_pesos = np.array(())
        incremento_pesos_vector_actual = np.array(())
        pesos_vector_actual = np.array(())
        contador = 0
        vector_error = np.array(())

        incremento_bias = 0
        peso_bias = 0.1

        y_in = 0
        y = 0

        for i in range(0, ncols - 1):
            incremento_pesos_vector_actual = np.append(incremento_pesos_vector_actual, 0)
            pesos_vector_actual = np.append(pesos_vector_actual, 0.1)

        matriz_pesos = np.append(matriz_pesos, np.copy(pesos_vector_actual))

        for k in range(0, self.n_epocas):
            contador = 0
            for i in range(0, nrows):
                y_in = 0
                for j in range(0, ncols - 1):
                    y_in += datostrain[i, j] * pesos_vector_actual[j]
                y_in += peso_bias

                if (y_in < self.frontera):
                    y = -1
                else:
                    y = 1


                for j in range(0, ncols - 1):
                    self.diferencia_y_t = datostrain[i, ncols - 1] - y_in
                    incremento_pesos_vector_actual[j] = datostrain[i, j] * self.diferencia_y_t * self.tasa_aprendizaje  # Incremento de Wi = Xi*T
                    incremento_bias = self.diferencia_y_t * self.tasa_aprendizaje * self.bias

                    pesos_vector_actual[j] += incremento_pesos_vector_actual[j]  # Wi = Wi + Incremento de Wi

                peso_bias += incremento_bias
                # matriz_pesos = np.vstack([matriz_pesos, np.copy(pesos_vector_actual)])


                if (y != datostrain[i, ncols - 1]):
                    contador += 1

            vector_error = np.append(vector_error, float(contador) / float(nrows))

        print "Numero de iteraciones de entrenamiento en el Adaline:"
        print self.n_epocas
        print ""

        print "La media de error en entrenamiento en el Adaline es:"
        print np.average(vector_error)
        print ""

        self.vector_final = pesos_vector_actual
        self.peso_bias_final = peso_bias

        pass



        # TODO: implementar

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):
        # Asignar la clase mayoritaria a todos los datos

        datospred = np.copy(datostest)
        num_filas, num_cols = datostest.shape
        for i in range(0, num_filas):
            y_in = 0
            for j in range(0, num_cols - 1):
                y_in += datospred[i, j] * self.vector_final[j]
            y_in += self.peso_bias_final

            if (y_in < self.frontera):
                y = -1
            else:
                y = 1

            datospred[i, num_cols - 1] = y


        return datospred[:, num_cols - 1]

    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas = datos.shape[0]
        lineas2 = pred.shape[0]

        file = open('predicciones_adaline.txt', 'w')
        file.write("Dato\tPrediccion\n")

        if (lineas != lineas2): return -1
        for i in range(0, lineas):
            file.write(str(datos[i]) + "\t" + str(pred[i]) + "\n")
            if (datos[i] == pred[i]):
                nAciertos = nAciertos + 1
            else:
                nFallos = nFallos + 1
        file.close()
        return nFallos / float(nFallos + nAciertos)


  
##############################################################################


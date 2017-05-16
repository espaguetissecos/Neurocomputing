from abc import ABCMeta, abstractmethod
import numpy as np
import random
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
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas = datos.shape[0]
        lineas2 = pred.shape[0]
        if (lineas != lineas2): return -1
        for i in range(0, lineas - 1):
            if (datos[i] == pred[i]):
                nAciertos = nAciertos + 1
            else:
                nFallos = nFallos + 1
        return nFallos / float(nFallos + nAciertos)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, salida=None, seed=None):
        particionado.creaParticiones(dataset)

        datos_train = np.array(())
        datos_test = np.array(())
        errores = []

        if particionado.nombreEstrategia == 'Simple':
            datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)
            num_clase = dataset.extraeNumeroClases()
            datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)

            clasificador.entrenamiento(datos_train, num_clase, salida)
            pred = clasificador.clasifica(datos_test)
            nrows, ncols = datos_test.shape
            nclases = int(num_clase)
            errores.append(self.error(datos_test[:, (ncols - nclases):], pred))

        ## Esto no hace falta
        elif particionado.nombreEstrategia == 'Cruzada':
            for i in range(0, particionado.numeroParticiones):
                datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
                datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)
                clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

                pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

                errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], pred))

        return errores


#############################################################################
############################### PERCEPTRON ##################################
#############################################################################


class ClasificadorPerceptron(Clasificador):
    mayoritaria = 0
    n_epocas = 1000
    vector_final = np.array(())
    peso_bias_final = 0
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
                        incremento_pesos_vector_actual[j] = datostrain[i, j] * datostrain[
                            i, ncols - 1] * self.tasa_aprendizaje  # Incremento de Wi = Xi*T
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
    peso_bias_final = 0
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
                    incremento_pesos_vector_actual[j] = datostrain[
                                                            i, j] * self.diferencia_y_t * self.tasa_aprendizaje  # Incremento de Wi = Xi*T
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

#############################################################################
######################### PERCEPTRON MULTICAPA ##############################
#############################################################################





class ClasificadorPerceptronMulticapa(Clasificador):
    mayoritaria = 0
    n_epocas = 500
    vector_final = np.array(())
    peso_bias_final = 0
    num_neuronas_oculta = 9
    num_neuronas_salida = 0
    nclases = 0
    ##
    ## VECTORES DE SESGOS
    ##
    vector_bias_z = np.array(())
    vector_bias_y = np.array(())

    ##
    ## VECTORES DE PESOS
    ##
    vector_pesos_z = np.array(())
    vector_pesos_y = np.array(())

    ##
    ## VECTOR DE CAPA OCULTA
    ##
    vector_zin = np.array(())
    vector_zout = np.array(())

    ##
    ## VECTOR DE CAPA SALIDA
    ##
    vector_yin = np.array(())
    vector_yout = np.array(())

    ##
    ## DELTAS
    ##
    vector_delta_y = np.array(())
    vector_delta_z = np.array(())
    vector_delta_zin = np.array(())

    ##
    ## INCREMENTOS
    ##
    vector_incremento_pesos_y = np.array(())
    vector_incremento_pesos_z = np.array(())

    vector_incremento_bias_y = np.array(())
    vector_incremento_bias_z = np.array(())

    bias = 1
    tasa_aprendizaje = 0.15
    frontera = 0

    def __init__(self, numNeuronas, tasaAprendizaje):
        self.num_neuronas_oculta = numNeuronas
        self.tasa_aprendizaje = tasaAprendizaje

    def entrenamiento(self, datos, f_nclases, salida, normalizar=False, atributosDiscretos=None, diccionario=None):

        datostrain = np.copy(datos)

        nrows, ncols = datostrain.shape

        matriz_pesos = np.array(())
        # incremento_pesos_vector_actual = np.array(())
        # pesos_vector_actual = np.array(())
        vector_error = np.array(())

        incremento_bias = 0
        peso_bias = 0
        y_in = 0
        y = 0
        error_cuadratico = 0
        error_cuadratico_aux = 0
        error_cuadratico_medio = 0

        nclases = int(f_nclases)
        self.nclases = nclases

        self.num_neuronas_salida = nclases

        self.normalizar = normalizar

        # Si el flag de normalizacion esta activado
        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - nclases):
                media = np.array(datostrain[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))

                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - nclases):
                    datostrain[i][j] = (datostrain[i][j] - float(mediasaux[j])) / float(desviacionesaux[j])

            print "NumAtributo\tMedia\tDesviacion"
            for i in range(0, ncols - nclases):
                print (str(i) + "\t" + str(mediasaux[i]) + "\t" + str(desviacionesaux[i]))

        ##
        ## Inicializacion capa oculta
        ##
        for j in range(0, self.num_neuronas_oculta):
            self.vector_zin = np.append(self.vector_zin, 0)
            self.vector_delta_z = np.append(self.vector_delta_z, 0)
            self.vector_delta_zin = np.append(self.vector_delta_zin, 0)

            if (j == 0):
                for i in range(0, ncols - nclases):
                    self.vector_incremento_pesos_z = np.append(self.vector_incremento_pesos_z, 0)
                    self.vector_pesos_z = np.append(self.vector_pesos_z, (random.random() - 0.5))
            else:
                vector_aux = np.array(())
                vector_aux2 = np.array(())

                for i in range(0, ncols - nclases):
                    vector_aux = np.append(vector_aux, (random.random() - 0.5))
                    vector_aux2 = np.append(vector_aux2, 0)
                self.vector_pesos_z = np.vstack([self.vector_pesos_z, np.copy(vector_aux)])
                self.vector_incremento_pesos_z = np.vstack([self.vector_incremento_pesos_z, np.copy(vector_aux2)])

            self.vector_zout = np.append(self.vector_zout, 0)
            self.vector_bias_z = np.append(self.vector_bias_z, (random.random() - 0.5))
            self.vector_incremento_bias_z = np.append(self.vector_incremento_bias_z, 0)
        vector_aux1 = np.array(())
        for i in range(0, ncols - nclases):
            vector_aux1 = np.append(vector_aux1, 0)
        self.vector_pesos_z = np.vstack([self.vector_pesos_z, vector_aux1])
        self.vector_incremento_pesos_z = np.vstack([self.vector_incremento_pesos_z, vector_aux1])

        ##
        ## Inicializacion capa salida
        ##
        for j in range(0, self.num_neuronas_salida):
            self.vector_yin = np.append(self.vector_yin, 0)
            self.vector_delta_y = np.append(self.vector_delta_y, 0)
            # self.vector_incremento_pesos_y = np.append(self.vector_incremento_pesos_y, 0)

            if (j == 0):
                for i in range(0, self.num_neuronas_oculta):
                    self.vector_incremento_pesos_y = np.append(self.vector_incremento_pesos_y, 0)
                    self.vector_pesos_y = np.append(self.vector_pesos_y, (random.random() - 0.5))
            else:
                vector_aux = np.array(())
                vector_aux2 = np.array(())

                for i in range(0, self.num_neuronas_oculta):
                    vector_aux = np.append(vector_aux, (random.random() - 0.5))
                    vector_aux2 = np.append(vector_aux2, 0)

                self.vector_pesos_y = np.vstack([self.vector_pesos_y, np.copy(vector_aux)])
                self.vector_incremento_pesos_y = np.vstack([self.vector_incremento_pesos_y, np.copy(vector_aux2)])

            self.vector_yout = np.append(self.vector_yout, 0)
            self.vector_bias_y = np.append(self.vector_bias_y, (random.random() - 0.5))
            self.vector_incremento_bias_y = np.append(self.vector_incremento_bias_y, 0)

        vector_aux2 = np.array(())
        for i in range(0, self.num_neuronas_oculta):
            vector_aux2 = np.append(vector_aux2, 0)
        self.vector_pesos_y = np.vstack([self.vector_pesos_y, vector_aux2])
        self.vector_incremento_pesos_y = np.vstack([self.vector_incremento_pesos_y, vector_aux2])

        ##
        ## ESCRITURA FICHERO SALIDA
        ##
        f = open(salida, "w")
        for j in range(0, self.num_neuronas_oculta):
            f.write("V0" + str(j + 1) + " ")
            for i in range(0, (ncols - nclases)):
                f.write("V" + str(i + 1) + str(j + 1) + " ")

        f.write("\t\t")

        for j in range(0, self.num_neuronas_salida):
            f.write("W0" + str(j + 1) + " ")
            for i in range(0, self.num_neuronas_oculta):
                f.write("W" + str(i + 1) + str(j + 1) + " ")

        f.write("\t\t")
        f.write("Error\n")

        for j in range(0, self.num_neuronas_oculta):
            f.write(str(self.vector_bias_z[j]) + " ")
            for i in range(0, (ncols - nclases)):
                f.write(str(self.vector_pesos_z[j][i]) + " ")

        f.write("         ")

        for j in range(0, self.num_neuronas_salida):
            f.write(str(self.vector_bias_y[j]) + " ")
            for i in range(0, self.num_neuronas_oculta):
                f.write(str(self.vector_pesos_y[j][i]) + " ")

        f.write("\t\t")

        f.write("0\n")

        ##
        ## ENTRENAMIENTO
        ##
        for k in range(0, self.n_epocas):
            contador = 0
            error_cuadratico_aux = 0
            for i in range(0, nrows):
                error_cuadratico = 0
                for l in range(0, self.num_neuronas_oculta):
                    self.vector_incremento_bias_z[l] = 0
                    self.vector_incremento_pesos_z[l] = 0
                    self.vector_delta_zin[l] = 0
                    self.vector_delta_z[l] = 0
                    self.vector_zin[l] = 0
                    self.vector_zout[l] = 0

                for l in range(0, self.num_neuronas_salida):
                    self.vector_incremento_bias_y[l] = 0
                    self.vector_incremento_pesos_y[l] = 0
                    self.vector_delta_y[l] = 0
                    self.vector_yin[l] = 0
                    self.vector_yout[l] = 0

                ##
                ## FEEDFORWARD CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):

                    for j in range(0, ncols - nclases):
                        self.vector_zin[l] += datostrain[i][j] * self.vector_pesos_z[l][j]
                    self.vector_zin[l] += self.vector_bias_z[l]

                    self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

                ##
                ## FEEDFORWARD CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):

                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                    self.vector_yin[l] += self.vector_bias_y[l]

                    self.vector_yout[l] = self.sigmoide_bipolar(self.vector_yin[l])

                ##
                ## BACKPROPAGATION CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):
                    self.vector_delta_y[l] = (datostrain[i][ncols - nclases + l] - self.vector_yout[
                        l]) * self.derivada_sigmoide_bipolar(self.vector_yin[l])

                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_incremento_pesos_y[l][j] = self.tasa_aprendizaje * self.vector_delta_y[l] * \
                                                               self.vector_zout[j]
                    self.vector_incremento_bias_y[l] = self.tasa_aprendizaje * self.vector_delta_y[l]
                ##
                ## BACKPROPAGATION CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):
                    for j in range(0, self.num_neuronas_salida):
                        self.vector_delta_zin[l] += self.vector_delta_y[j] * self.vector_pesos_y[j][l]

                    self.vector_delta_z[l] = self.vector_delta_zin[l] * self.derivada_sigmoide_bipolar(
                        self.vector_zin[l])

                    for j in range(0, ncols - nclases):
                        self.vector_incremento_pesos_z[l][j] = self.tasa_aprendizaje * self.vector_delta_z[l] * \
                                                               datostrain[i][j]

                    self.vector_incremento_bias_z[l] = self.tasa_aprendizaje * self.vector_delta_z[l]

                ##
                ## AJUSTE DE LOS PESOS CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):
                    for j in range(0, ncols - nclases):
                        self.vector_pesos_z[l][j] += self.vector_incremento_pesos_z[l][j]
                    self.vector_bias_z[l] += self.vector_incremento_bias_z[l]

                ##
                ## AJUSTE DE LOS PESOS CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):
                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_pesos_y[l][j] += self.vector_incremento_pesos_y[l][j]
                    self.vector_bias_y[l] += self.vector_incremento_bias_y[l]

                ##
                ## ERROR CUADRATICO POR FILA
                ##
                for l in range(0, self.num_neuronas_salida):
                    error_cuadratico += (datostrain[i][ncols - nclases + l] - self.vector_yout[l]) ** 2
                error_cuadratico_aux += 0.5 * error_cuadratico

                contador += self.error_entrena(self.vector_yout, datostrain[i][ncols - nclases:])

            vector_error = np.append(vector_error, float(contador) / float(nrows))
            for j in range(0, self.num_neuronas_oculta):
                f.write(str(self.vector_bias_z[j]) + " ")
                for i in range(0, (ncols - nclases)):
                    f.write(str(self.vector_pesos_z[j][i]) + " ")

            f.write("\t\t")

            for j in range(0, self.num_neuronas_salida):
                f.write(str(self.vector_bias_y[j]) + " ")
                for i in range(0, self.num_neuronas_oculta):
                    f.write(str(self.vector_pesos_y[j][i]) + " ")

            f.write("\t\t")
            error_cuadratico_medio = error_cuadratico_aux / nrows
            f.write(str(error_cuadratico_medio) + "\n")

        print "La tasa de acierto en la particion de entrenamiento en el Perceptron Multicapa es: "
        print str((1 - np.average(vector_error)) * 100) + " %"

        f.close()

        f = open("error_frente_a_epocas.txt", "w")
        for i in range(0, len(vector_error)):
            f.write(str(i))
            f.write(" ")
            f.write(str(vector_error[i]))
            f.write("\n")

        pass

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):

        datospred = np.copy(datostest)
        nrows, ncols = datostest.shape

        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - self.nclases):
                media = np.array(datospred[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))
                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - self.nclases):
                    datospred[i][j] = (datospred[i][j] - float(mediasaux[j])) / float(desviacionesaux[j])

        for i in range(0, nrows):
            clase = 0
            for l in range(0, self.num_neuronas_oculta):
                self.vector_incremento_bias_z[l] = 0
                self.vector_incremento_pesos_z[l] = 0
                self.vector_delta_zin[l] = 0
                self.vector_delta_z[l] = 0
                self.vector_zin[l] = 0
                self.vector_zout[l] = 0
            for l in range(0, self.num_neuronas_salida):
                self.vector_incremento_bias_y[l] = 0
                self.vector_incremento_pesos_y[l] = 0
                self.vector_delta_y[l] = 0
                self.vector_yin[l] = 0
                self.vector_yout[l] = 0

            ##
            ## FEEDFORWARD CAPA OCULTA
            ##
            for l in range(0, self.num_neuronas_oculta):

                for j in range(0, ncols - self.nclases):
                    self.vector_zin[l] += datospred[i][j] * self.vector_pesos_z[l][j]
                self.vector_zin[l] += self.vector_bias_z[l]

                self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

            ##
            ## FEEDFORWARD CAPA SALIDA
            ##
            for l in range(0, self.num_neuronas_salida):

                for j in range(0, self.num_neuronas_oculta):
                    self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                self.vector_yin[l] += self.vector_bias_y[l]

                self.vector_yout[l] = self.sigmoide_bipolar(self.vector_yin[l])

            maximo = 0
            pos = 0
            for l in range(0, self.num_neuronas_salida):
                if (self.vector_yout[l] > maximo):
                    maximo = self.vector_yout[l]
                    pos = l

            vector_prediccion = np.array(())

            for l in range(0, self.num_neuronas_salida):
                if (self.vector_yout[l] == maximo):
                    vector_prediccion = np.append(vector_prediccion, 1)
                else:
                    vector_prediccion = np.append(vector_prediccion, 0)

            datospred[i][(ncols - self.nclases):] = vector_prediccion

        return datospred[:, (ncols - self.num_neuronas_salida):]
        pass

    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas, cols = datos.shape
        lineas2 = pred.shape[0]

        file = open('predicciones_nnet.txt', 'w')
        file.write("Prediccion\n")

        if (lineas != lineas2):
            return -1
        for i in range(0, lineas):
            for j in range(0, cols):
                file.write(str(pred[i][j]) + "\n")

            for j in range(0, cols):
                if (datos[i][j] != pred[i][j]):
                    nFallos = nFallos + 1
                    break

        file.close()

        return float(nFallos) / float(lineas)

    def sigmoide_bipolar(self, zin):

        zout = (2.0 / (1.0 + np.exp(float(-zin)))) - 1.0
        return zout

    def derivada_sigmoide_bipolar(self, entry):
        sigmoide = self.sigmoide_bipolar(entry)
        out = 0.5 * (1.0 + sigmoide) * (1.0 - sigmoide)
        return out

    def validacion(self, particionado, dataset, clasificador, salida=None, seed=None):
        particionado.creaParticiones(dataset)

        datos_train = np.array(())
        datos_test = np.array(())
        errores = []

        if particionado.nombreEstrategia == 'Simple':
            datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)
            num_clase = dataset.extraeNumeroClases()
            datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)

            clasificador.entrenamiento(datos_train, num_clase, salida)
            pred = clasificador.clasifica(datos_test)
            nrows, ncols = datos_test.shape
            nclases = int(num_clase)
            errores.append(self.error(datos_test[:, (ncols - nclases):], pred))

        ## Esto no hace falta
        elif particionado.nombreEstrategia == 'Cruzada':
            for i in range(0, particionado.numeroParticiones):
                datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
                datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)
                clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

                pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

                errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], pred))

        return errores

    def error_entrena(self, vector_yout, datostrain):

        maximo = 0

        for l in range(0, self.num_neuronas_salida):
            if (vector_yout[l] > maximo):
                maximo = self.vector_yout[l]
                pos = l

        vector_prediccion = np.array(())

        for l in range(0, self.num_neuronas_salida):
            if (self.vector_yout[l] == maximo):
                vector_prediccion = np.append(vector_prediccion, 1)
            else:
                vector_prediccion = np.append(vector_prediccion, 0)

        for l in range(0, self.num_neuronas_salida):
            if (vector_prediccion[l] != datostrain[l]):
                return 1

        return 0
        pass

        ##################################################################################

    #############################################################################
    ######################### AUTOENCODER #######################################
    #############################################################################





class AutoEncoder(Clasificador):
    mayoritaria = 0
    n_epocas = 1000
    vector_final = np.array(())
    peso_bias_final = 0
    num_neuronas_oculta = 9
    num_neuronas_salida = 0
    nclases = 0
    ##
    ## VECTORES DE SESGOS
    ##
    vector_bias_z = np.array(())
    vector_bias_y = np.array(())

    ##
    ## VECTORES DE PESOS
    ##
    vector_pesos_z = np.array(())
    vector_pesos_y = np.array(())

    ##
    ## VECTOR DE CAPA OCULTA
    ##
    vector_zin = np.array(())
    vector_zout = np.array(())

    ##
    ## VECTOR DE CAPA SALIDA
    ##
    vector_yin = np.array(())
    vector_yout = np.array(())

    ##
    ## DELTAS
    ##
    vector_delta_y = np.array(())
    vector_delta_z = np.array(())
    vector_delta_zin = np.array(())

    ##
    ## INCREMENTOS
    ##
    vector_incremento_pesos_y = np.array(())
    vector_incremento_pesos_z = np.array(())

    vector_incremento_bias_y = np.array(())
    vector_incremento_bias_z = np.array(())

    bias = 1
    tasa_aprendizaje = 0.15
    frontera = 0

    def __init__(self, numNeuronas, tasaAprendizaje):
        self.num_neuronas_oculta = numNeuronas
        self.tasa_aprendizaje = tasaAprendizaje

    def entrenamiento(self, datos, f_nclases, salida, normalizar=False, atributosDiscretos=None, diccionario=None):

        datostrain = np.copy(datos)

        nrows, ncols = datostrain.shape

        matriz_pesos = np.array(())
        # incremento_pesos_vector_actual = np.array(())
        # pesos_vector_actual = np.array(())
        vector_error = np.array(())

        incremento_bias = 0
        peso_bias = 0
        y_in = 0
        y = 0
        error_cuadratico = 0
        error_cuadratico_aux = 0
        error_cuadratico_medio = 0

        nclases = int(f_nclases)
        self.nclases = nclases

        self.num_neuronas_salida = nclases

        self.normalizar = normalizar

        # Si el flag de normalizacion esta activado
        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - nclases):
                media = np.array(datostrain[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))

                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - nclases):
                    datostrain[i][j] = (datostrain[i][j] - float(mediasaux[j])) / float(desviacionesaux[j])

            print "NumAtributo\tMedia\tDesviacion"
            for i in range(0, ncols - nclases):
                print (str(i) + "\t" + str(mediasaux[i]) + "\t" + str(desviacionesaux[i]))

        ##
        ## Inicializacion capa oculta
        ##
        for j in range(0, self.num_neuronas_oculta):
            self.vector_zin = np.append(self.vector_zin, 0)
            self.vector_delta_z = np.append(self.vector_delta_z, 0)
            self.vector_delta_zin = np.append(self.vector_delta_zin, 0)

            if (j == 0):
                for i in range(0, ncols - nclases):
                    self.vector_incremento_pesos_z = np.append(self.vector_incremento_pesos_z, 0)
                    self.vector_pesos_z = np.append(self.vector_pesos_z, (random.random() - 0.5))
            else:
                vector_aux = np.array(())
                vector_aux2 = np.array(())

                for i in range(0, ncols - nclases):
                    vector_aux = np.append(vector_aux, (random.random() - 0.5))
                    vector_aux2 = np.append(vector_aux2, 0)
                self.vector_pesos_z = np.vstack([self.vector_pesos_z, np.copy(vector_aux)])
                self.vector_incremento_pesos_z = np.vstack([self.vector_incremento_pesos_z, np.copy(vector_aux2)])

            self.vector_zout = np.append(self.vector_zout, 0)
            self.vector_bias_z = np.append(self.vector_bias_z, (random.random() - 0.5))
            self.vector_incremento_bias_z = np.append(self.vector_incremento_bias_z, 0)
        vector_aux1 = np.array(())
        for i in range(0, ncols - nclases):
            vector_aux1 = np.append(vector_aux1, 0)
        self.vector_pesos_z = np.vstack([self.vector_pesos_z, vector_aux1])
        self.vector_incremento_pesos_z = np.vstack([self.vector_incremento_pesos_z, vector_aux1])

        ##
        ## Inicializacion capa salida
        ##
        for j in range(0, self.num_neuronas_salida):
            self.vector_yin = np.append(self.vector_yin, 0)
            self.vector_delta_y = np.append(self.vector_delta_y, 0)
            # self.vector_incremento_pesos_y = np.append(self.vector_incremento_pesos_y, 0)

            if (j == 0):
                for i in range(0, self.num_neuronas_oculta):
                    self.vector_incremento_pesos_y = np.append(self.vector_incremento_pesos_y, 0)
                    self.vector_pesos_y = np.append(self.vector_pesos_y, (random.random() - 0.5))
            else:
                vector_aux = np.array(())
                vector_aux2 = np.array(())

                for i in range(0, self.num_neuronas_oculta):
                    vector_aux = np.append(vector_aux, (random.random() - 0.5))
                    vector_aux2 = np.append(vector_aux2, 0)

                self.vector_pesos_y = np.vstack([self.vector_pesos_y, np.copy(vector_aux)])
                self.vector_incremento_pesos_y = np.vstack([self.vector_incremento_pesos_y, np.copy(vector_aux2)])

            self.vector_yout = np.append(self.vector_yout, 0)
            self.vector_bias_y = np.append(self.vector_bias_y, (random.random() - 0.5))
            self.vector_incremento_bias_y = np.append(self.vector_incremento_bias_y, 0)

        vector_aux2 = np.array(())
        for i in range(0, self.num_neuronas_oculta):
            vector_aux2 = np.append(vector_aux2, 0)
        self.vector_pesos_y = np.vstack([self.vector_pesos_y, vector_aux2])
        self.vector_incremento_pesos_y = np.vstack([self.vector_incremento_pesos_y, vector_aux2])

        ##
        ## ESCRITURA FICHERO SALIDA
        ##
        f = open(salida, "w")
        for j in range(0, self.num_neuronas_oculta):
            f.write("V0" + str(j + 1) + " ")
            for i in range(0, (ncols - nclases)):
                f.write("V" + str(i + 1) + str(j + 1) + " ")

        f.write("\t\t")

        for j in range(0, self.num_neuronas_salida):
            f.write("W0" + str(j + 1) + " ")
            for i in range(0, self.num_neuronas_oculta):
                f.write("W" + str(i + 1) + str(j + 1) + " ")

        f.write("\t\t")
        f.write("Error\n")

        for j in range(0, self.num_neuronas_oculta):
            f.write(str(self.vector_bias_z[j]) + " ")
            for i in range(0, (ncols - nclases)):
                f.write(str(self.vector_pesos_z[j][i]) + " ")

        f.write("         ")

        for j in range(0, self.num_neuronas_salida):
            f.write(str(self.vector_bias_y[j]) + " ")
            for i in range(0, self.num_neuronas_oculta):
                f.write(str(self.vector_pesos_y[j][i]) + " ")

        f.write("\t\t")

        f.write("0\n")

        ##
        ## ENTRENAMIENTO
        ##
        for k in range(0, self.n_epocas):
            contador = 0
            error_cuadratico_aux = 0
            for i in range(0, nrows):
                error_cuadratico = 0
                for l in range(0, self.num_neuronas_oculta):
                    self.vector_incremento_bias_z[l] = 0
                    self.vector_incremento_pesos_z[l] = 0
                    self.vector_delta_zin[l] = 0
                    self.vector_delta_z[l] = 0
                    self.vector_zin[l] = 0
                    self.vector_zout[l] = 0

                for l in range(0, self.num_neuronas_salida):
                    self.vector_incremento_bias_y[l] = 0
                    self.vector_incremento_pesos_y[l] = 0
                    self.vector_delta_y[l] = 0
                    self.vector_yin[l] = 0
                    self.vector_yout[l] = 0

                ##
                ## FEEDFORWARD CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):

                    for j in range(0, ncols - nclases):
                        self.vector_zin[l] += float(datostrain[i][j]) * self.vector_pesos_z[l][j]
                    self.vector_zin[l] += self.vector_bias_z[l]

                    self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

                ##
                ## FEEDFORWARD CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):

                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                    self.vector_yin[l] += self.vector_bias_y[l]

                    self.vector_yout[l] = self.sigmoide_bipolar(self.vector_yin[l])

                ##
                ## BACKPROPAGATION CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):
                    self.vector_delta_y[l] = (float(datostrain[i][ncols - nclases + l]) - self.vector_yout[
                        l]) * self.derivada_sigmoide_bipolar(self.vector_yin[l])

                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_incremento_pesos_y[l][j] = self.tasa_aprendizaje * self.vector_delta_y[l] * \
                                                               self.vector_zout[j]
                    self.vector_incremento_bias_y[l] = self.tasa_aprendizaje * self.vector_delta_y[l]
                ##
                ## BACKPROPAGATION CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):
                    for j in range(0, self.num_neuronas_salida):
                        self.vector_delta_zin[l] += self.vector_delta_y[j] * self.vector_pesos_y[j][l]

                    self.vector_delta_z[l] = self.vector_delta_zin[l] * self.derivada_sigmoide_bipolar(
                        self.vector_zin[l])

                    for j in range(0, ncols - nclases):
                        self.vector_incremento_pesos_z[l][j] = self.tasa_aprendizaje * self.vector_delta_z[l] * \
                                                               float(datostrain[i][j])

                    self.vector_incremento_bias_z[l] = self.tasa_aprendizaje * self.vector_delta_z[l]

                ##
                ## AJUSTE DE LOS PESOS CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):
                    for j in range(0, ncols - nclases):
                        self.vector_pesos_z[l][j] += self.vector_incremento_pesos_z[l][j]
                    self.vector_bias_z[l] += self.vector_incremento_bias_z[l]

                ##
                ## AJUSTE DE LOS PESOS CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):
                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_pesos_y[l][j] += self.vector_incremento_pesos_y[l][j]
                    self.vector_bias_y[l] += self.vector_incremento_bias_y[l]

                ##
                ## ERROR CUADRATICO POR FILA
                ##
                for l in range(0, self.num_neuronas_salida):
                    error_cuadratico += (float(datostrain[i][ncols - nclases + l]) - self.vector_yout[l]) ** 2
                error_cuadratico_aux += 0.5 * error_cuadratico

                contador += self.error_entrena(self.vector_yout, datostrain[i][ncols - nclases:])

            vector_error = np.append(vector_error, float(contador) / float(nrows))
            for j in range(0, self.num_neuronas_oculta):
                f.write(str(self.vector_bias_z[j]) + " ")
                for i in range(0, (ncols - nclases)):
                    f.write(str(self.vector_pesos_z[j][i]) + " ")

            f.write("\t\t")

            for j in range(0, self.num_neuronas_salida):
                f.write(str(self.vector_bias_y[j]) + " ")
                for i in range(0, self.num_neuronas_oculta):
                    f.write(str(self.vector_pesos_y[j][i]) + " ")

            f.write("\t\t")
            error_cuadratico_medio = error_cuadratico_aux / nrows
            f.write(str(error_cuadratico_medio) + "\n")

        print "La tasa de acierto promedio de epocas en la particion de entrenamiento en el AutoEncoder es: "
        print str((1 - np.average(vector_error)) * 100) + " %"

        f.close()

        f = open("error_frente_a_epocas.txt", "w")
        for i in range(0, len(vector_error)):
            f.write(str(i))
            f.write(" ")
            f.write(str(vector_error[i]))
            f.write("\n")

        pass

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):

        datospred = np.copy(datostest)
        nrows, ncols = datostest.shape

        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - self.nclases):
                media = np.array(datospred[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))
                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - self.nclases):
                    datospred[i][j] = (datospred[i][j] - float(mediasaux[j])) / float(desviacionesaux[j])

        for i in range(0, nrows):
            clase = 0
            for l in range(0, self.num_neuronas_oculta):
                self.vector_incremento_bias_z[l] = 0
                self.vector_incremento_pesos_z[l] = 0
                self.vector_delta_zin[l] = 0
                self.vector_delta_z[l] = 0
                self.vector_zin[l] = 0
                self.vector_zout[l] = 0
            for l in range(0, self.num_neuronas_salida):
                self.vector_incremento_bias_y[l] = 0
                self.vector_incremento_pesos_y[l] = 0
                self.vector_delta_y[l] = 0
                self.vector_yin[l] = 0
                self.vector_yout[l] = 0

            ##
            ## FEEDFORWARD CAPA OCULTA
            ##
            for l in range(0, self.num_neuronas_oculta):

                for j in range(0, ncols - self.nclases):
                    self.vector_zin[l] += float(datospred[i][j]) * self.vector_pesos_z[l][j]
                self.vector_zin[l] += self.vector_bias_z[l]

                self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

            ##
            ## FEEDFORWARD CAPA SALIDA
            ##
            for l in range(0, self.num_neuronas_salida):

                for j in range(0, self.num_neuronas_oculta):
                    self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                self.vector_yin[l] += self.vector_bias_y[l]

                self.vector_yout[l] = self.sigmoide_bipolar(self.vector_yin[l])
            #
            # maximo = 0
            # pos = 0
            # for l in range(0, self.num_neuronas_salida):
            #     if (self.vector_yout[l] > maximo):
            #         maximo = self.vector_yout[l]
            #         pos = l

            vector_prediccion = np.array(())

            for l in range(0, self.num_neuronas_salida):
                if (self.vector_yout[l] >= 0):
                    vector_prediccion = np.append(vector_prediccion, 1)
                else:
                    vector_prediccion = np.append(vector_prediccion, -1)

            datospred[i][(ncols - self.nclases):] = vector_prediccion

        return datospred[:, (ncols - self.num_neuronas_salida):]
        pass

    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas, cols = datos.shape
        lineas2 = pred.shape[0]

        file = open('predicciones_nnet.txt', 'w')
        file.write("Prediccion\n")

        if (lineas != lineas2):
            return -1
        for i in range(0, lineas):
            nFallos_linea = 0
            # for j in range(0, cols):
            #     file.write(str(pred[i][j]) + "\n")

            for j in range(0, cols):
                if (float(datos[i][j]) != float(pred[i][j])):
                    nFallos_linea = nFallos_linea + 1
            nFallos += float(nFallos_linea)

        file.close()

        return float(nFallos) / float(lineas)

    def letras(self, datos, pred):

        lineas, cols = datos.shape
        lineas2 = pred.shape[0]
        letras = []

        if (lineas != lineas2):
            return -1
        for i in range(0, lineas):
            for j in range(0, cols):
                if (float(datos[i][j]) != float(pred[i][j])):
                    letras.append(chr(65 + (i%26)))
                    break


        return letras

    def sigmoide_bipolar(self, zin):

        zout = (2.0 / (1.0 + np.exp(float(-zin)))) - 1.0
        return zout

    def derivada_sigmoide_bipolar(self, entry):
        sigmoide = self.sigmoide_bipolar(entry)
        out = 0.5 * (1.0 + sigmoide) * (1.0 - sigmoide)
        return out

    def validacion(self, particionado, dataset, clasificador, salida=None, seed=None):
        particionado.creaParticiones(dataset)

        datos_train = np.array(())
        datos_test = np.array(())
        errores = []

        if particionado.nombreEstrategia == 'Simple':
            datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)
            num_clase = dataset.extraeNumeroClases()
            datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)

            clasificador.entrenamiento(datos_train, num_clase, salida)
            pred = clasificador.clasifica(datos_test)
            nrows, ncols = datos_test.shape
            nclases = int(num_clase)
            errores.append(self.error(datos_test[:, (ncols - nclases):], pred))

        ## Esto no hace falta
        elif particionado.nombreEstrategia == 'Cruzada':
            for i in range(0, particionado.numeroParticiones):
                datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
                datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)
                clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

                pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

                errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], pred))

        return errores

    def error_entrena(self, vector_yout, datostrain):

        # maximo = 0
        #
        # for l in range(0, self.num_neuronas_salida):
        #     if (vector_yout[l] > maximo):
        #         maximo = self.vector_yout[l]
        #         pos = l

        vector_prediccion = np.array(())

        for l in range(0, self.num_neuronas_salida):
            if (self.vector_yout[l] >= 0):
                vector_prediccion = np.append(vector_prediccion, 1)
            else:
                vector_prediccion = np.append(vector_prediccion, -1)

        for l in range(0, self.num_neuronas_salida):
            if (float(vector_prediccion[l]) != float(datostrain[l])):
                return 1

        return 0
        pass

        ##################################################################################


class SerieTemporal(Clasificador):
    mayoritaria = 0


    n_epocas = 250
    vector_final = np.array(())
    peso_bias_final = 0
    num_neuronas_oculta = 20
    num_neuronas_salida = 0
    nclases = 0
    ##
    ## VECTORES DE SESGOS
    ##
    vector_bias_z = np.array(())
    vector_bias_y = np.array(())

    ##
    ## VECTORES DE PESOS
    ##
    vector_pesos_z = np.array(())
    vector_pesos_y = np.array(())

    ##
    ## VECTOR DE CAPA OCULTA
    ##
    vector_zin = np.array(())
    vector_zout = np.array(())

    ##
    ## VECTOR DE CAPA SALIDA
    ##
    vector_yin = np.array(())
    vector_yout = np.array(())

    ##
    ## DELTAS
    ##
    vector_delta_y = np.array(())
    vector_delta_z = np.array(())
    vector_delta_zin = np.array(())

    ##
    ## INCREMENTOS
    ##
    vector_incremento_pesos_y = np.array(())
    vector_incremento_pesos_z = np.array(())

    vector_incremento_bias_y = np.array(())
    vector_incremento_bias_z = np.array(())

    bias = 1
    tasa_aprendizaje = 0.15
    frontera = 0


    def __init__(self, numNeuronas, tasaAprendizaje):
        self.num_neuronas_oculta = numNeuronas
        self.tasa_aprendizaje = tasaAprendizaje


    def entrenamiento(self, datos, f_nclases, salida, normalizar=True, atributosDiscretos=None, diccionario=None):
        datostrain = np.copy(datos)

        nrows, ncols = datostrain.shape

        matriz_pesos = np.array(())
        # incremento_pesos_vector_actual = np.array(())
        # pesos_vector_actual = np.array(())
        vector_error = np.array(())

        incremento_bias = 0
        peso_bias = 0
        y_in = 0
        y = 0
        error_cuadratico = 0
        error_cuadratico_aux = 0
        error_cuadratico_medio = 0

        nclases = int(f_nclases)
        self.nclases = nclases

        self.num_neuronas_salida = nclases

        self.normalizar = normalizar

        # Si el flag de normalizacion esta activado
        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - nclases):
                media = np.array(datostrain[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))

                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - nclases):
                    datostrain[i][j] = (float(datostrain[i][j]) - float(mediasaux[j])) / float(desviacionesaux[j])

            print "NumAtributo\tMedia\tDesviacion"
            for i in range(0, ncols - nclases):
                print (str(i) + "\t" + str(mediasaux[i]) + "\t" + str(desviacionesaux[i]))

        ##
        ## Inicializacion capa oculta
        ##
        for j in range(0, self.num_neuronas_oculta):
            self.vector_zin = np.append(self.vector_zin, 0)
            self.vector_delta_z = np.append(self.vector_delta_z, 0)
            self.vector_delta_zin = np.append(self.vector_delta_zin, 0)

            if (j == 0):
                for i in range(0, ncols - nclases):
                    self.vector_incremento_pesos_z = np.append(self.vector_incremento_pesos_z, 0)
                    self.vector_pesos_z = np.append(self.vector_pesos_z, (random.random() - 0.5))
            else:
                vector_aux = np.array(())
                vector_aux2 = np.array(())

                for i in range(0, ncols - nclases):
                    vector_aux = np.append(vector_aux, (random.random() - 0.5))
                    vector_aux2 = np.append(vector_aux2, 0)
                self.vector_pesos_z = np.vstack([self.vector_pesos_z, np.copy(vector_aux)])
                self.vector_incremento_pesos_z = np.vstack([self.vector_incremento_pesos_z, np.copy(vector_aux2)])

            self.vector_zout = np.append(self.vector_zout, 0)
            self.vector_bias_z = np.append(self.vector_bias_z, (random.random() - 0.5))
            self.vector_incremento_bias_z = np.append(self.vector_incremento_bias_z, 0)
        vector_aux1 = np.array(())
        for i in range(0, ncols - nclases):
            vector_aux1 = np.append(vector_aux1, 0)
        self.vector_pesos_z = np.vstack([self.vector_pesos_z, vector_aux1])
        self.vector_incremento_pesos_z = np.vstack([self.vector_incremento_pesos_z, vector_aux1])

        ##
        ## Inicializacion capa salida
        ##
        for j in range(0, self.num_neuronas_salida):
            self.vector_yin = np.append(self.vector_yin, 0)
            self.vector_delta_y = np.append(self.vector_delta_y, 0)
            # self.vector_incremento_pesos_y = np.append(self.vector_incremento_pesos_y, 0)

            if (j == 0):
                for i in range(0, self.num_neuronas_oculta):
                    self.vector_incremento_pesos_y = np.append(self.vector_incremento_pesos_y, 0)
                    self.vector_pesos_y = np.append(self.vector_pesos_y, (random.random() - 0.5))
            else:
                vector_aux = np.array(())
                vector_aux2 = np.array(())

                for i in range(0, self.num_neuronas_oculta):
                    vector_aux = np.append(vector_aux, (random.random() - 0.5))
                    vector_aux2 = np.append(vector_aux2, 0)

                self.vector_pesos_y = np.vstack([self.vector_pesos_y, np.copy(vector_aux)])
                self.vector_incremento_pesos_y = np.vstack([self.vector_incremento_pesos_y, np.copy(vector_aux2)])

            self.vector_yout = np.append(self.vector_yout, 0)
            self.vector_bias_y = np.append(self.vector_bias_y, (random.random() - 0.5))
            self.vector_incremento_bias_y = np.append(self.vector_incremento_bias_y, 0)

        vector_aux2 = np.array(())
        for i in range(0, self.num_neuronas_oculta):
            vector_aux2 = np.append(vector_aux2, 0)
        self.vector_pesos_y = np.vstack([self.vector_pesos_y, vector_aux2])
        self.vector_incremento_pesos_y = np.vstack([self.vector_incremento_pesos_y, vector_aux2])

        ##
        ## ESCRITURA FICHERO SALIDA
        ##
        f = open(salida, "w")
        for j in range(0, self.num_neuronas_oculta):
            f.write("V0" + str(j + 1) + " ")
            for i in range(0, (ncols - nclases)):
                f.write("V" + str(i + 1) + str(j + 1) + " ")

        f.write("\t\t")

        for j in range(0, self.num_neuronas_salida):
            f.write("W0" + str(j + 1) + " ")
            for i in range(0, self.num_neuronas_oculta):
                f.write("W" + str(i + 1) + str(j + 1) + " ")

        f.write("\t\t")
        f.write("Error\n")

        for j in range(0, self.num_neuronas_oculta):
            f.write(str(self.vector_bias_z[j]) + " ")
            for i in range(0, (ncols - nclases)):
                f.write(str(self.vector_pesos_z[j][i]) + " ")

        f.write("         ")

        for j in range(0, self.num_neuronas_salida):
            f.write(str(self.vector_bias_y[j]) + " ")
            for i in range(0, self.num_neuronas_oculta):
                f.write(str(self.vector_pesos_y[j][i]) + " ")

        f.write("\t\t")

        f.write("0\n")

        ##
        ## ENTRENAMIENTO
        ##
        for k in range(0, self.n_epocas):
            contador = 0
            error_cuadratico_aux = 0
            for i in range(0, nrows):
                error_cuadratico = 0
                for l in range(0, self.num_neuronas_oculta):
                    self.vector_incremento_bias_z[l] = 0
                    self.vector_incremento_pesos_z[l] = 0
                    self.vector_delta_zin[l] = 0
                    self.vector_delta_z[l] = 0
                    self.vector_zin[l] = 0
                    self.vector_zout[l] = 0

                for l in range(0, self.num_neuronas_salida):
                    self.vector_incremento_bias_y[l] = 0
                    self.vector_incremento_pesos_y[l] = 0
                    self.vector_delta_y[l] = 0
                    self.vector_yin[l] = 0
                    self.vector_yout[l] = 0

                ##
                ## FEEDFORWARD CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):

                    for j in range(0, ncols - nclases):
                        self.vector_zin[l] += float(datostrain[i][j]) * self.vector_pesos_z[l][j]
                    self.vector_zin[l] += self.vector_bias_z[l]

                    self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

                ##
                ## FEEDFORWARD CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):

                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                    self.vector_yin[l] += self.vector_bias_y[l]

                    self.vector_yout[l] = (self.vector_yin[l]) # CAMBIO IMPORTANTEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

                ##
                ## BACKPROPAGATION CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):
                    self.vector_delta_y[l] = (float(datostrain[i][ncols - nclases + l]) - self.vector_yout[
                        l]) ## CAMBIO IMPORTANTEEEE* self.derivada_sigmoide_bipolar(self.vector_yin[l])

                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_incremento_pesos_y[l][j] = self.tasa_aprendizaje * self.vector_delta_y[l] * \
                                                               self.vector_zout[j]
                    self.vector_incremento_bias_y[l] = self.tasa_aprendizaje * self.vector_delta_y[l]
                ##
                ## BACKPROPAGATION CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):
                    for j in range(0, self.num_neuronas_salida):
                        self.vector_delta_zin[l] += self.vector_delta_y[j] * self.vector_pesos_y[j][l]

                    self.vector_delta_z[l] = self.vector_delta_zin[l] * self.derivada_sigmoide_bipolar(
                        self.vector_zin[l])

                    for j in range(0, ncols - nclases):
                        self.vector_incremento_pesos_z[l][j] = self.tasa_aprendizaje * self.vector_delta_z[l] * \
                                                               float(datostrain[i][j])

                    self.vector_incremento_bias_z[l] = self.tasa_aprendizaje * self.vector_delta_z[l]

                ##
                ## AJUSTE DE LOS PESOS CAPA OCULTA
                ##
                for l in range(0, self.num_neuronas_oculta):
                    for j in range(0, ncols - nclases):
                        self.vector_pesos_z[l][j] += self.vector_incremento_pesos_z[l][j]
                    self.vector_bias_z[l] += self.vector_incremento_bias_z[l]

                ##
                ## AJUSTE DE LOS PESOS CAPA SALIDA
                ##
                for l in range(0, self.num_neuronas_salida):
                    for j in range(0, self.num_neuronas_oculta):
                        self.vector_pesos_y[l][j] += self.vector_incremento_pesos_y[l][j]
                    self.vector_bias_y[l] += self.vector_incremento_bias_y[l]

                ##
                ## ERROR CUADRATICO POR FILA
                ##
                for l in range(0, self.num_neuronas_salida):
                    error_cuadratico += (float(datostrain[i][ncols - nclases + l]) - self.vector_yout[l]) ** 2
                error_cuadratico_aux += 0.5 * error_cuadratico

                contador += self.error_entrena(self.vector_yout, (datostrain[i][ncols - nclases:]))

            vector_error = np.append(vector_error, float(contador) / float(nrows))
            for j in range(0, self.num_neuronas_oculta):
                f.write(str(self.vector_bias_z[j]) + " ")
                for i in range(0, (ncols - nclases)):
                    f.write(str(self.vector_pesos_z[j][i]) + " ")

            f.write("\t\t")

            for j in range(0, self.num_neuronas_salida):
                f.write(str(self.vector_bias_y[j]) + " ")
                for i in range(0, self.num_neuronas_oculta):
                    f.write(str(self.vector_pesos_y[j][i]) + " ")

            f.write("\t\t")
            error_cuadratico_medio = error_cuadratico_aux / nrows
            f.write(str(error_cuadratico_medio) + "\n")

        print "El ECM en el entrenamiento es: "
        print str((np.average(vector_error)))

        f.close()

        f = open("error_frente_a_epocas.txt", "w")
        for i in range(0, len(vector_error)):
            f.write(str(i))
            f.write(" ")
            f.write(str(vector_error[i]))
            f.write("\n")

        pass


    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):
        datospred = np.copy(datostest)
        nrows, ncols = datostest.shape

        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - self.nclases):
                media = np.array(datospred[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))
                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - self.nclases):
                    datospred[i][j] = (float(datospred[i][j]) - float(mediasaux[j])) / float(desviacionesaux[j])

        for i in range(0, nrows):
            clase = 0
            for l in range(0, self.num_neuronas_oculta):
                self.vector_incremento_bias_z[l] = 0
                self.vector_incremento_pesos_z[l] = 0
                self.vector_delta_zin[l] = 0
                self.vector_delta_z[l] = 0
                self.vector_zin[l] = 0
                self.vector_zout[l] = 0
            for l in range(0, self.num_neuronas_salida):
                self.vector_incremento_bias_y[l] = 0
                self.vector_incremento_pesos_y[l] = 0
                self.vector_delta_y[l] = 0
                self.vector_yin[l] = 0
                self.vector_yout[l] = 0

            ##
            ## FEEDFORWARD CAPA OCULTA
            ##
            for l in range(0, self.num_neuronas_oculta):

                for j in range(0, ncols - self.nclases):
                    self.vector_zin[l] += float(datospred[i][j]) * self.vector_pesos_z[l][j]
                self.vector_zin[l] += self.vector_bias_z[l]

                self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

            ##
            ## FEEDFORWARD CAPA SALIDA
            ##
            for l in range(0, self.num_neuronas_salida):

                for j in range(0, self.num_neuronas_oculta):
                    self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                self.vector_yin[l] += self.vector_bias_y[l]

                self.vector_yout[l] = (self.vector_yin[l])

            datospred[i][(ncols - self.nclases):] = self.vector_yout # vector_prediccion

        return datospred[:, (ncols - self.num_neuronas_salida):]
        pass

    def clasifica_recursivo(self, datostest, atributosDiscretos=None, diccionario=None):
        datospred = np.copy(datostest)
        nrows, ncols = datostest.shape
        Na = 5
        Nf = 500
        if (self.normalizar):
            mediasaux = []
            desviacionesaux = []
            for i in range(0, ncols - self.nclases):
                media = np.array(datospred[:, i]).astype(np.float)
                desv = np.std(media)
                mediasaux.append(np.mean(media))
                desviacionesaux.append(desv)

            for i in range(0, nrows):
                for j in range(0, ncols - self.nclases):
                    datospred[i][j] = (float(datospred[i][j]) - float(mediasaux[j])) / float(desviacionesaux[j])

        ###################################################################################
        ##                              CASO BASE
        ###################################################################################

        for l in range(0, self.num_neuronas_oculta):
            self.vector_incremento_bias_z[l] = 0
            self.vector_incremento_pesos_z[l] = 0
            self.vector_delta_zin[l] = 0
            self.vector_delta_z[l] = 0
            self.vector_zin[l] = 0
            self.vector_zout[l] = 0
        for l in range(0, self.num_neuronas_salida):
            self.vector_incremento_bias_y[l] = 0
            self.vector_incremento_pesos_y[l] = 0
            self.vector_delta_y[l] = 0
            self.vector_yin[l] = 0
            self.vector_yout[l] = 0

        ##
        ## FEEDFORWARD CAPA OCULTA
        ##
        for l in range(0, self.num_neuronas_oculta):

            for j in range(0, ncols - self.nclases):
                self.vector_zin[l] += float(datospred[0][j]) * self.vector_pesos_z[l][j]
            self.vector_zin[l] += self.vector_bias_z[l]

            self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

        ##
        ## FEEDFORWARD CAPA SALIDA
        ##
        for l in range(0, self.num_neuronas_salida):

            for j in range(0, self.num_neuronas_oculta):
                self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
            self.vector_yin[l] += self.vector_bias_y[l]

            self.vector_yout[l] = (self.vector_yin[l])

        datospred[0][(ncols - self.nclases):] = self.vector_yout

        ###################################################################################
        ##                    FIN DE CASO BASE
        ###################################################################################


        for i in range(1, Nf):

            for j in range(0, ncols - self.nclases):
                datospred[i][j] = datospred[i - 1][ncols - Na + j]

            clase = 0
            for l in range(0, self.num_neuronas_oculta):
                self.vector_incremento_bias_z[l] = 0
                self.vector_incremento_pesos_z[l] = 0
                self.vector_delta_zin[l] = 0
                self.vector_delta_z[l] = 0
                self.vector_zin[l] = 0
                self.vector_zout[l] = 0
            for l in range(0, self.num_neuronas_salida):
                self.vector_incremento_bias_y[l] = 0
                self.vector_incremento_pesos_y[l] = 0
                self.vector_delta_y[l] = 0
                self.vector_yin[l] = 0
                self.vector_yout[l] = 0

            ##
            ## FEEDFORWARD CAPA OCULTA
            ##
            for l in range(0, self.num_neuronas_oculta):

                for j in range(0, ncols - self.nclases):
                    dato = float(datospred[i][j])
                    self.vector_zin[l] += dato * self.vector_pesos_z[l][j]
                self.vector_zin[l] += self.vector_bias_z[l]

                self.vector_zout[l] = self.sigmoide_bipolar(self.vector_zin[l])

            ##
            ## FEEDFORWARD CAPA SALIDA
            ##
            for l in range(0, self.num_neuronas_salida):

                for j in range(0, self.num_neuronas_oculta):
                    self.vector_yin[l] += self.vector_zout[j] * self.vector_pesos_y[l][j]
                self.vector_yin[l] += self.vector_bias_y[l]

                self.vector_yout[l] = (self.vector_yin[l])

            datospred[i][(ncols - self.nclases):] = self.vector_yout
        ret = datospred[0:Nf, (ncols - self.num_neuronas_salida):]
        return ret
        pass

    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas, cols = datos.shape
        lineas2 = pred.shape[0]
        ecm = 0
        if (lineas != lineas2):
            return -1
        for i in range(0, lineas):
            for j in range(0, cols):
                ecm += (float(datos[i][j]) - float(pred[i][j]))**2


        return float(ecm) / float(lineas)


    def sigmoide_bipolar(self, zin):
        zout = (2.0 / (1.0 + np.exp(float(-zin)))) - 1.0
        return zout


    def derivada_sigmoide_bipolar(self, entry):
        sigmoide = self.sigmoide_bipolar(entry)
        out = 0.5 * (1.0 + sigmoide) * (1.0 - sigmoide)
        return out


    def validacion(self, particionado, dataset, clasificador, random=True, salida=None, recursivo=False, seed=None):
        particionado.creaParticiones(dataset, False)

        datos_train = np.array(())
        datos_test = np.array(())
        errores = []

        if particionado.nombreEstrategia == 'Simple':
            datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)
            num_clase = dataset.extraeNumeroClases()
            datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)

            clasificador.entrenamiento(datos_train, num_clase, salida)

            if (recursivo):
                pred = clasificador.clasifica_recursivo(datos_test)
            else:
                pred = clasificador.clasifica(datos_test)

            nrows, ncols = datos_test.shape
            nclases = int(num_clase)
            real = datos_test[:, (ncols - nclases):]
            f1 = open("Original", "w")
            f2 = open("Prediccion", "w")
            for i in range(0, len(pred)):
                f2.write(str(i+1) + " " + str(pred[i][0]) + "\n")
            for i in range(0, len(real)):
                f1.write(str(i+1) + " " + str(real[i][0]) + "\n")
            f2.close()
            f1.close()

            errores.append(self.error(real, pred))

        ## Esto no hace falta
        elif particionado.nombreEstrategia == 'Cruzada':
            for i in range(0, particionado.numeroParticiones):
                datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
                datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)
                clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

                pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

                errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], pred))

        return errores


    def error_entrena(self, vector_yout, datostrain):

        ecm = 0
        # for l in range(0, self.num_neuronas_salida):
        #     ecm += (float(vector_yout[l]) - float(datostrain[l]))**2

        return ecm
        pass

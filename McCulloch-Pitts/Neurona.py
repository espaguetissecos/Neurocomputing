class Neurona(object):
    umbral = 0
    activada = 0
    pesos = []
    enlaces = []

    def __init__(self, enlaces=None, pesos=None, umbral=0, activada=0):
        super(Neurona, self).__init__()
        self.enlaces = enlaces
        self.pesos = pesos
        self.umbral = umbral
        self.activada = activada

    def propagar(self):
        enlace = 0
        for i in range(0, len(self.enlaces)):
            enlace = enlace + int(self.pesos[i]) * int(self.enlaces[i].activada)

        if enlace < self.umbral:
            self.activada = 0
        else:
            self.activada = 1
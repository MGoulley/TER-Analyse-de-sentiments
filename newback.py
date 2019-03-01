# From: https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
import math
import random

class Noeud:
    valeur = 0
    somme = 0
    arcs = []
    nom = ""
    def __init__(self, val, som, nom):
        self.valeur = val
        self.somme = som
        self.arcs = []
        self.nom = nom

    def add_arc(self, arc):
        self.arcs.append(arc)

    def sigmoid(self):
        try:
            ans = math.exp(-self.somme)
        except OverflowError:
            ans = float('inf')
        self.valeur = 1/(1+ans)

    def __repr__(self):
        return "<%s -- Valeur: %s Somme: %s %s>" % (self.nom, self.valeur, self.somme, self.arcs)
    def __str__(self):
        return "%s \n   Valeur: %s \n   Somme: %s \n   Arcs: %s" % (self.nom, self.valeur, self.somme, self.arcs)

class Arc:
    def __init__(self, val, noeud1, noeud2):
        self.valeur = val
        self.noeudD = noeud1
        self.noeudA = noeud2

    def __repr__(self):
        return "<Valeur arc: %s Arrivee: %s>" % (self.valeur, self.noeudA.nom)
    def __str__(self):
        return "Valeur arc: %s \nArrivee: %s" % (self.valeur, self.noeudA.nom)

class Reseau:
    def __init__(self, inputs, hiddens, outputs):
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.noeuds = inputs + hiddens + outputs

    def forward(self):
        self.reset()
        for noeud in self.inputs:
            for arc in noeud.arcs:
                arc.noeudA.somme += noeud.valeur * arc.valeur

        for noeud in self.hiddens:
            noeud.sigmoid()
            for arc in noeud.arcs:
                arc.noeudA.somme += noeud.valeur * arc.valeur

        for noeud in self.outputs:
            noeud.sigmoid()

    def delta_output_sum(somme, marge_erreur):
        return  sigmoid_derivee(somme) * marge_erreur

    def backward(self, expected):
        try:
            ans = math.exp(-self.outputs[0].somme)
        except OverflowError:
            ans = float('inf')
        delta = (1/(1+ans)) * (1-(1/(1+ans))) * (expected - ans)
        for noeud in self.hiddens:
            for arc in noeud.arcs:
                try:
                    ans = math.exp(-noeud.somme)
                except OverflowError:
                    ans = float('inf')
                noeud.delta_hidden_sum = delta / arc.valeur * (1/(1+ans)) * (1-(1/(1+ans)))
                arc.valeur += (delta / noeud.valeur)

        for noeud in self.inputs:
            for arc in noeud.arcs:
                if arc.noeudD.valeur != 0:
                    arc.valeur += arc.noeudA.delta_hidden_sum / arc.noeudD.valeur

    def reset(self):
        for noeud in self.noeuds:
            noeud.somme = 0

    def print_reseau(self):
        for noeud in self.noeuds:
            print(noeud)


def predict(reseau, input1, input2):
    reseau.inputs[0].valeur = 0
    reseau.inputs[1].valeur = 0
    reseau.forward()
    if reseau.outputs[0].valeur > 0.5:
        return 1
    else:
        return 0

def train(reseau, nbtrain):
    compteur = 0
    while compteur < nbtrain:
        # if compteur % 2:
        #     #training 0,0 -> 0
        #     reseau.inputs[0].valeur = 1
        #     reseau.inputs[1].valeur = 1
        #     output = 0
        # else:
        #     #training 1,1 -> 1
        #     reseau.inputs[0].valeur = 1
        #     reseau.inputs[1].valeur = 0
        #     output = 1
        # reseau.inputs[0].valeur = 1
        # reseau.inputs[1].valeur = 0
        # output = 0
        # reseau.forward()
        # reseau.backward(output)
        reseau.inputs[0].valeur = 1
        reseau.inputs[1].valeur = 0
        output = 1
        reseau.forward()
        reseau.backward(output)
        reseau.inputs[0].valeur = 0
        reseau.inputs[1].valeur = 1
        output = 1
        reseau.forward()
        reseau.backward(output)
        reseau.inputs[0].valeur = 1
        reseau.inputs[1].valeur = 1
        output = 0
        reseau.forward()
        reseau.backward(output)
        reseau.inputs[0].valeur = 0
        reseau.inputs[1].valeur = 0
        output = 0
        reseau.forward()
        reseau.backward(output)
        compteur+= 1

# Input Layer
n1 = Noeud(1,0,"n1")
n2 = Noeud(1,0,"n2")
# Hidden Layer
n3 = Noeud(0,0,"n3")
n4 = Noeud(0,0,"n4")
n5 = Noeud(0,0,"n5")
# Output Layer
n6 = Noeud(0,0,"n6")

# creation des arcs (exemple)
# a13 = Arc(0.8, n1, n3)
# a14 = Arc(0.4, n1, n4)
# a15 = Arc(0.3, n1, n5)
# a23 = Arc(0.2, n2, n3)
# a24 = Arc(0.9, n2, n4)
# a25 = Arc(0.5, n2, n5)
# a36 = Arc(0.3, n3, n6)
# a46 = Arc(0.5, n4, n6)
# a56 = Arc(0.9, n5, n6)

# creation des arcs random
a13 = Arc(random.random(), n1, n3)
a14 = Arc(random.random(), n1, n4)
a15 = Arc(random.random(), n1, n5)
a23 = Arc(random.random(), n2, n3)
a24 = Arc(random.random(), n2, n4)
a25 = Arc(random.random(), n2, n5)
a36 = Arc(random.random(), n3, n6)
a46 = Arc(random.random(), n4, n6)
a56 = Arc(random.random(), n5, n6)

# ajout des arcs aux noeuds
n1.add_arc(a13)
n1.add_arc(a14)
n1.add_arc(a15)
n2.add_arc(a23)
n2.add_arc(a24)
n2.add_arc(a25)
n3.add_arc(a36)
n4.add_arc(a46)
n5.add_arc(a56)


reseau = Reseau([n1,n2],[n3,n4,n5],[n6])
train(reseau, 10)
print(predict(reseau, 0,0))
print(predict(reseau, 1,0))
# compteur = 0
# i = 0
# nberreur = 0
# while i < 20:
#     temp = predict(reseau, 0,0)
#     if temp != 0:
#         nberreur += 1
#     i += 1
# print(nberreur)

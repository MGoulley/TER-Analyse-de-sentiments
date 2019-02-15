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
        self.valeur = 1/(1+math.exp(-self.somme))

    def __repr__(self):
        return "<%s -- Valeur: %s Somme: %s>" % (self.nom, self.valeur, self.somme)
    def __str__(self):
        return "%s \n   Valeur: %s \n   Somme: %s" % (self.nom, self.valeur, self.somme)


class Arc:
    def __init__(self, val, noeud1, noeud2):
        self.valeur = val
        self.noeudD = noeud1
        self.noeudA = noeud2

    def __repr__(self):
        return "<Valeur arc: %s Depart: %s Arrivee: %s>" % (self.valeur, self.noeudD, self.noeudA)
    def __str__(self):
        return "Valeur arc: %s \nDepart: %s \nArrivee: %s" % (self.valeur, self.noeudD.nom, self.noeudA.nom)


# Fonctions du réseau
def calcul_somme(lst, noeudoutput):
    for noeud in lst:
        for arc in noeud.arcs:
            arc.noeudA.somme += arc.valeur
    noeudoutput.somme = 0

def calcul_sigmoid(lst, noeudoutput):
    somme = 0
    for noeud in lst:
        if(noeud.valeur != 1 and noeud != noeudoutput):
            noeud.sigmoid()
            for arc in noeud.arcs:
                if(arc.noeudA == noeudoutput):
                    somme += noeud.valeur * arc.valeur
    noeudoutput.somme = somme
    noeudoutput.sigmoid()

def sigmoid_derivee(val):
    return (1/(1+math.exp(-val))) * (1-(1/(1+math.exp(-val))))

def delta_output_sum(somme, marge_erreur):
    return  sigmoid_derivee(somme) * marge_erreur

def backward(lst, arcs, noeudoutput, outputTarget):
    delta = delta_output_sum(noeudoutput.somme, outputTarget - noeudoutput.valeur)
    hidden_nodes = []
    arcs_to_output = []
    for arc in arcs:
        if(arc.noeudA == noeudoutput):
            a = Arc(arc.valeur, arc.noeudD, arc.noeudA) # sauvegarde l'ancienne valeure de l'arc
            arc.valeur += delta / arc.noeudD.valeur
            hidden_nodes.append(arc.noeudD)
            arcs_to_output.append(a)
    for arc in arcs:
        for hidden_arc in arcs_to_output:
            if(arc.noeudA in hidden_nodes and hidden_arc.noeudD == arc.noeudA):
                arc.valeur += (((delta / hidden_arc.valeur) * sigmoid_derivee(arc.noeudA.somme)) / arc.noeudD.valeur)

def reset(lst):
    for noeud in lst:
        noeud.somme = 0

def print_reseau(lst):
    for noeud in lst:
        print(noeud)

def print_arcs(lst):
    for arc in lst:
        print(arc)

input1 = 1
input2 = 1
output = 0

# Input Layer
n1 = Noeud(input1,0,"n1")
n2 = Noeud(input2,0,"n2")
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

# Le reseau de neurones
neurones = []
neurones.append(n1)
neurones.append(n2)
neurones.append(n3)
neurones.append(n4)
neurones.append(n5)
neurones.append(n6)
arcs = []
arcs.append(a13)
arcs.append(a14)
arcs.append(a15)
arcs.append(a23)
arcs.append(a24)
arcs.append(a25)
arcs.append(a36)
arcs.append(a46)
arcs.append(a56)

compteur = 0
min = 1.0
while compteur < 10000:
    calcul_somme(neurones, n6)
    calcul_sigmoid(neurones, n6)
    backward(neurones, arcs, n6, output)
    reset(neurones)
    compteur+= 1
print(n6.valeur)

# print_reseau(neurones)
# n1.valeur = 0
# n2.valeur = 0
# print_reseau(neurones)
# calcul_somme(neurones, n6)
# calcul_sigmoid(neurones, n6)
# valeur_output = n6.valeur
# print(valeur_output)
#print_arcs(arcs)
#print_reseau(neurones)

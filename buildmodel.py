import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

features = ['sentsize', 'nbwords', 'mentions', 'hashtags', 'emojis']
analyse = 'sentiment'

train = pd.read_excel(r'michel.xlsx')
train.head()
X_train = train.loc[:,features]
X_train.shape
Y_train = train.loc[:, analyse]
Y_train.shape

test = pd.read_excel(r'test.xlsx')
test.head()
X_test = test.loc[:,features]
X_test.shape
Y_test = test.loc[:, analyse]
Y_test.shape

# construction du modele
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

#Evaluation du modele (vrai, prediction)
print("Precision: " + str(precision_score(Y_test, knn.predict(X_test), average='macro')))
print("Rappel: " + str(recall_score(Y_test, knn.predict(X_test), average='macro')))
print("F1-Score: " + str(f1_score(Y_test, knn.predict(X_test), average='macro')))

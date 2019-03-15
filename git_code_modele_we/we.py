import re
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_phrase(file):
    fi=open(file,encoding="utf8")
    res_temp=[]
    for line in fi:
        res_temp.append(line)
    res=[]
    for elem in res_temp:
        x=elem.split(",")
        res.append(x)
    return res

def token(tab):
    aq=[]
    for elem,jy in tab:
        #temp=nltk.word_tokenize(elem)
        temp=re.split(" |'",elem)
        aq.append(temp)
    return aq


def label(tab):
    aq=[]
    for elem,jy in tab:
        aq.append(int(jy))
    return aq
def phra(tab):
    aq=[]
    for elem,y in tab:
        aq.append(elem)
    return aq

#recuperation des phrases
file=get_phrase("propre2.csv")
#tokenisation de la phrase
rf=token(file)
sentences=phra(file)

max_length=max([len(elem) for elem in rf ])

#shape de l'array de la phrase
k=np.shape(rf)

# train model
model = Word2Vec(rf, min_count=1)

# somme des vecteurs d'une phrase
matrice =[]
for i in range(0,k[0]):
    sum=model[rf[i][0]]+model[rf[i][1]]
    for j in range(2,len(rf[i])):
        sum=sum+model[rf[i][j]]
    matrice.append(sum)

#affichage de graphe
pca = PCA(n_components=2)
result = pca.fit_transform(matrice)
plt.scatter(result[:, 0], result[:, 1])
#Annonation mais illisible pour une phrase
# for i, word in enumerate(file):
#     plt.annotate(word, xy=(result[i, 0], result[i, 1]))
#plt.show()

labels=label(file)

from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test=train_test_split(matrice, labels, test_size=0.20, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding

EMBEDDING_DIM=100
print(" Build Model ............")

model=Sequential()
model.add(Embedding(len(matrice),EMBEDDING_DIM,input_length=max_length))
model.add(GRU(units=32,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

#model.fit(X_train,y_train,batch_size=128,epochs=25,verbose=2)



from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(sentences)

max_long=max([len(s.split()) for s in sentences])
vocab_size=len(tokenizer_obj.word_index)+1

#separation pour les phrase
x,y=train_test_split(sentences,test_size=0.2,random_state=42)
x_token=tokenizer_obj.texts_to_sequences(x)
y_token=tokenizer_obj.texts_to_sequences(y)

x_pad=pad_sequences(x_token,maxlen=max_long,padding='post')
y_pad=pad_sequences(y_token,maxlen=max_long,padding='post')

modell=Sequential()
modell.add(Embedding(vocab_size,EMBEDDING_DIM,input_length=max_long))
modell.add(GRU(units=32,dropout=0.2,recurrent_dropout=0.2))
modell.add(Dense(1,activation='sigmoid'))

modell.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(modell.summary())

#separation pour les labels
x_lab,y_lab=train_test_split(labels,test_size=0.2,random_state=42)
#print(x_lab)

print('Train.......................')
modell.fit(x_pad,x_lab,batch_size=128,epochs=6,validation_data=(y_pad,y_lab),verbose=2)

test_sample1="La syrie ca vous interresse ou pas ? "
test_sample2="La patronne du FMI Christine Lagarde mise en examen pour négligence en France via    "
test_sample3="Merci baba pour ce  et vivement  &"

test_sample=[test_sample1,test_sample2,test_sample3]

test_sample_token=tokenizer_obj.texts_to_sequences(test_sample)
test_sample_token_pad=pad_sequences(test_sample_token,maxlen=max_long)
print(modell.predict(x=test_sample_token_pad))

print(x_pad[1])
print(x[1])
print(x_lab[1])
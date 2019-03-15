import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def evaluation_tweets(fichier, knn):
    test = pd.read_excel(fichier)
    test.head()
    X_test = test.loc[:,features]
    X_test.shape
    Y_test = test.loc[:, analyse]
    Y_test.shape

    #Evaluation du modele (vrai, prediction)
    print("Precision: " + str(precision_score(Y_test, knn.predict(X_test), average='macro')))
    print("Rappel: " + str(recall_score(Y_test, knn.predict(X_test), average='macro')))
    print("F1-Score: " + str(f1_score(Y_test, knn.predict(X_test), average='macro')))

def evaluation_tweet(tweet, knn):
    # Recupere le tweet
    sentence = re.search(r"\t.*\t", tweet, flags=re.IGNORECASE)
    if sentence:
        sentence = sentence.groups()
    else:
        sentence = tweet
    #retire les urls
    sentence = re.sub(r"http\S+", '', sentence, flags=re.IGNORECASE)
    # Recupere les emoticones et les retire
    emote = re.findall(r"([&][#][\w]*[;])", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"([&][#][\w]*[;])", '', sentence, flags=re.IGNORECASE)
    #retire les mentions @.....
    mention = re.findall(r"@\S+", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"@\S+", '', sentence, flags=re.IGNORECASE)
    #retire les hashtags
    hashtag = re.findall(r"#\S+", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"#\S+", '', sentence, flags=re.IGNORECASE)
    #tokenise la phrase
    sentence_tokens = word_tokenize(sentence)

    X_test = [[len(sentence),len(sentence_tokens), len(mention), len(hashtag), len(emote)]]
    sentiment = ""
    predict = knn.predict(X_test)
    if predict == 0:
        sentiment = "negative"
    elif sentiment == 1:
        sentiment = "positive"
    elif sentiment == 2:
        sentiment = "mixed"
    else:
        sentiment = "objectif"
    print("Le tweet: " + sentence + " est " + sentiment)

def build_latex(X,Y,fichier_test):
    test = pd.read_excel(fichier_test)
    test.head()
    X_test = test.loc[:,features]
    X_test.shape
    Y_test = test.loc[:, analyse]
    Y_test.shape

    methodes_names = ['K-nearest neighbors', 'Logistic Regression', 'Decision Tree',
    'Linear Discriminant', 'Gaussian Naive Bayes', 'Support Vector Machine',
    'MLPClassifier', 'Random Forest', 'AdaBoost', 'Quadratic Discriminant']

    methodes = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(),
    LinearDiscriminantAnalysis(), GaussianNB(), SVC(), MLPClassifier(),
    RandomForestClassifier(), AdaBoostClassifier(), QuadraticDiscriminantAnalysis()]

    print(r"\begin{center}")
    print(r"\begin{tabular}{ |l | c | c | r |}")
    print(r"\hline")
    print(r"Méthode & Precision & Rappel & F1-Score \\ \hline")
    for mtd_name, mtd in zip(methodes_names, methodes):
        mtd.fit(X,Y)
        print_score_latex(mtd_name, mtd, X_test, Y_test)
    print(r"\end{tabular}")
    print(r"\end{center}")

def print_score_latex(methode_name, methode, X_test, Y_test):
    print(methode_name + " & " + str(round(precision_score(Y_test, methode.predict(X_test), average='macro'), 4)) +
            " & " + str(round(recall_score(Y_test, methode.predict(X_test), average='macro'), 4)) + " & " +
            str(round(f1_score(Y_test, methode.predict(X_test), average='macro'), 4)) + r" \\ \hline")



features = ['sentsize', 'nbwords', 'mentions', 'hashtags', 'emojis']
analyse = 'sentiment'

train = pd.read_excel(r'michel.xlsx')
train.head()
X_train = train.loc[:,features]
X_train.shape

Y_train = train.loc[:, analyse]
Y_train.shape

build_latex(X_train,Y_train,'test.xlsx')


# construction du modele
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)

#Evaluation
#evaluation_tweets(r'test.xlsx', knn)
#evaluation_tweet("#Immigration Est-ce qu'on a besoin de 400 000 étrangers de plus par an alors que nous avons d innombrables territoires qui ne sont plus en France et qui sont islamisés intégralement ? dit Eric Zemmour sur @BFMTV #NewsAndCo", knn)

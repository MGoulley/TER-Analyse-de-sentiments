#Auteur : DUVAL Cedric M1 ATAL
#Date : 22 mars 2019

from nltk.tokenize import TweetTokenizer
#from nltk.stem.snowball import FrenchStemmer
import spacy
import re
# python3 -m spacy download fr
stop=open("stopwords_fr.txt","r")
tknzr = TweetTokenizer()
#stemmer = FrenchStemmer()
nlp = spacy.load("fr")

#--------------------Creation d'un tweet a notre facon--------------------#
#--------------------   Brut : le tweet sans le sentiment ----------------#
#--------------------   Sentiment : le sentiment annoncer pour le tweet --#
#--------------------   Token : le tweet tokenise ------------------------#
#--------------------   Clean : le tweet sans # @ emote ------------------#
#--------------------   Emoji : les emojis du tweet ----------------------#
#--------------------   Mention : les mentions du tweet ------------------#
#--------------------   Hashtag : les hashtags du twwet ------------------#
#--------------------   Lem : le tweet lemmatise -------------------------#
#--------------------   Stem : le tweet racine ---------------------------#
class tweet:
    def __init__(self, Tweet, Sentiment):
        self.Brut =Tweet
        self.Sentiment = Sentiment
        self.Sentimentnb = numb(Sentiment)
        [self.Token, self.Clean, self.Emoji, self.Mention, self.Hashtag] = tokeniseur(Tweet,tknzr)
        self.Lem = lemma(self.Token.copy(),nlp)
        #self.Stem = stemma(self.Token.copy(),stemmer)

#----------------------       Retirer les stop words  --------------------#
#-----------------------------  ENTREE  ----------------------------------#
#---------------------- k : tableau de token d'un tweet ------------------#
#-----------------------------  SORTIE  ----------------------------------#
#---------------------- k : tweet dont aucun mot est commun avec stopword #
def stopw(k):
    stopw=[]
    for line in stop:
        stopw.append(line.rstrip('\n'))
    for i in k:
        if i in stopw:
            k.remove(i)
    return k

#----------------------       Lemming d'un tweet      --------------------#
#-----------------------------  ENTREE  ----------------------------------#
#---------------------- k : tableau de token d'un tweet ------------------#
#---------------------- nlp : fonction spacy chargeant un dictionnaire fr-#
#-----------------------------  SORTIE  ----------------------------------#
#---------------------- lemtemp : tableau de lemme des tokens ------------#
def lemma(k,nlp):
    lemtemp=[]
    stopw(k)
    for i in range(0,len(k)):
        doc = nlp(k[i])
        mot=[(word.lemma_) for word in doc]
        lemtemp.append(mot[0])
    return lemtemp

#----------------------       Stemming d'un tweet     --------------------#
#-----------------------------  ENTREE  ----------------------------------#
#---------------------- k : tableau de token d'un tweet ------------------#
#---------------------- stemmer : fonction de nltk ayant les racines fr---#
#-----------------------------  SORTIE  ----------------------------------#
#---------------------- stemtemp : tableau de racine des tokens ----------#
def stemma(k,stemmer):
    stemtemp =[]
    stopw(k)
    for i in range(0,len(k)):
        stemtemp.append(stemmer.stem(k[i]))
    return stemtemp

#----------------------       Tokenizing d'un tweet     ------------------#
#-----------------------------  ENTREE  ----------------------------------#
#---------------------- line : tableau de token d'un tweet ---------------#
#---------------------- tknzr : fonction de nltk ayant les racines fr-----#
#-----------------------------  SORTIE  ----------------------------------#
#---------------------- k : tweet tokenized ------------------------------#
def tokeniseur(line,tknzr):
    line = re.search(r"\t.*\t", line, flags=re.IGNORECASE).group()
    #----------------retire les urls----------------#
    line = re.sub(r"http\S+", '', line, flags=re.IGNORECASE)
    #----------------Recupere les emoticones----------------#
    emote = re.findall(r"([&][#][\w]*[;])", line, flags=re.IGNORECASE)
    line = re.sub(r"([&][#][\w]*[;])", '', line, flags=re.IGNORECASE)
    #----------------retire les mentions @.....----------------#
    mention = re.findall(r"@\S+", line, flags=re.IGNORECASE)
    line = re.sub(r"@\S+", '', line, flags=re.IGNORECASE)
    #----------------retire les hashtags----------------#
    hashtag = re.findall(r"#\S+", line, flags=re.IGNORECASE)
    line = re.sub(r"#\S+", '', line, flags=re.IGNORECASE)
    #---------------------Tokenize---------------------#
    k=tknzr.tokenize(line)
    return [k,line,emote,mention,hashtag]

#----------------------       Conversion d'un sentiment   ----------------#
#-----------------------------  ENTREE  ----------------------------------#
#---------------------- str : sentiment du tweet en charactere------------#
#-----------------------------  SORTIE  ----------------------------------#
#---------------------- n : valeur correspondante au sentiment -----------#
def numb(str):
    n=-1
    if str == "positive":
        n=1
    elif str == "negative":
        n=0
    elif str == "mixed":
        n=2
    elif str == "objective":
        n=3
    return n

#------------------------------       Main    ----------------------------#
#-----------------------------  SORTIE  ----------------------------------#
#---------------------- tweetos : liste de tweet -------------------------#
def main():
    fichier= open("train-deft2017.txt","r")
    #fichier = open("train.txt","r")

    tweetos=[]
    o=1
    for line in fichier:
        print(o)
        #----------------Recupere le sentiment du tweet----------------#
        sentiment = re.sub(r"^.+\t", '', line, flags=re.IGNORECASE)
        line = re.sub(r"sentiment", '', line, flags=re.IGNORECASE)
        sentiment = sentiment.rstrip()
        tweets = tweet(line,sentiment)
        tweetos.append(tweets)
        o=o+1
    return tweetos


# for i in range(0,len(tweetos)):
#     print(tweetos[i].Token, tweetos[i].Sentiment)

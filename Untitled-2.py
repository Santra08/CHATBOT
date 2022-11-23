import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from nltk import flatten
class T:
    def __init__(self):
        self.words=[]
        self.classes=[]
        self.doc=[]
        self.ignore_words = ['?', '!','(',')']
        data_file = open(r"C:\Users\santr\Desktop\santra kujnni\CHATBOT\intents-1.json").read()
        self.intents = json.loads(data_file)['intents']
        self.pre_processing()
    def pre_processing(self):
        for i in self.intents: #or self.intente["intents"] but then change line 18
            for p in i['patterns']:
                w = nltk.word_tokenize(p)
                self.words.extend(w)
                self.doc.append((w,i["tag"]))
                if i["tag"] not in self.classes:
                    self.classes.append(i["tag"])
        
        self.words=[lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        print(self.words)
t=T()
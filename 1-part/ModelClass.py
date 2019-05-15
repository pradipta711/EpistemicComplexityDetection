# -*- coding: utf-8 -*-
import json
import pandas as pd
from abc import ABC, abstractmethod
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import spacy
nlp = spacy.load("en_core_web_sm")
#Model class preprocesses and loads the Training data
class Model(ABC):
    
    def __init__(self):
        pass
    
    @classmethod
    def getWordnetPos(self,word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        # print(nltk.pos_tag([word]))
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    @classmethod
    def preprocessRawText(self,raw_text):
        #lowercase
        raw_text = raw_text.lower()
        #remove punctuation and numbers and split into seperate words
        raw_text  = re.sub('[^a-zA-Z]',' ',raw_text)
        words = re.findall(r'\w+', raw_text,flags = re.UNICODE)
        #removal of stopwords
        important_words = filter(lambda x: x not in stopwords.words('english'), words)
        # Init Lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize important_words with the appropriate POS tag
        lemmatized_raw_word_list = [lemmatizer.lemmatize(w, self.getWordnetPos(w)) for w in important_words]
        return lemmatized_raw_word_list

    @classmethod
    def GetNounPhrases(self,text_answer):
        text_answer = self.preprocessRawText(text_answer)
        text_answer = ' '.join(text_answer)
        doc = nlp(text_answer)
        answer=[chunk.text for chunk in doc.noun_chunks]
        answer = ' '.join(answer)
        return answer

    @classmethod     
    def SetTrainingData(self):
        #Load the specific dataset of the unit
        #dataset = pd.read_csv("Soil_training.csv")
        dataset = pd.read_csv("light_data.csv")
        dataset['label_complexity']=dataset.complexity_level.map({'L-EF':0,'L-F':1,'L-EE':2,'L-E':3})
        corpus = []

        for i in range(0,len(dataset)):
            answer= dataset['answer'][i]
            answer=self.GetNounPhrases(answer)
            corpus.append(answer)
        
        vect =CountVectorizer(max_features=600)
        X = vect.fit_transform(corpus).toarray()  #convert the text into numeric data through vectorization
        y= dataset.label_complexity
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
        listTrain= [X_train,X_test,y_train,y_test,X,y]
        return listTrain
    
    @classmethod
    def GetMetrics(self,y_test,y_pred):
        acc =metrics.accuracy_score(y_test, y_pred)             #calculate the accuracy
        return acc 
         
    @abstractmethod
    def GetPredictions():
        pass
    
    @abstractmethod
    def GetCrossValidation():
        pass
        
   
    



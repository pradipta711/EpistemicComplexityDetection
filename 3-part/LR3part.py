# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
#nltk.download('wordnet')
from nltk.corpus import wordnet
import spacy
import re
import nltk
nlp = spacy.load("en_core_web_sm")


class LR3partC:
    
    def __init__(self):
        print("Inside constructor LR3part")

    def get_wordnet_pos(self,word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess_raw_text(self,raw_text):
        raw_text = raw_text.lower()
        raw_text  = re.sub('[^a-zA-Z]',' ',raw_text)
        words = re.findall(r'\w+', raw_text,flags = re.UNICODE) 
        important_words = filter(lambda x: x not in stopwords.words('english'), words)
        lemmatizer = WordNetLemmatizer()
        lammatized_raw_word_list = [lemmatizer.lemmatize(w,self.get_wordnet_pos(w)) for w in important_words]
        return lammatized_raw_word_list

    
    def noun_phrases(self,text_answer):
        text_answer =self.preprocess_raw_text(text_answer)
        text_answer = ' '.join(text_answer)
        doc = nlp(text_answer)
        answer=[chunk.text for chunk in doc.noun_chunks]
        answer = ' '.join(answer)
        return answer
    
    def execute3part(self):
        #Part 1 of 3 Classification of students notes to Fact vs Explanation
        trainingDF = pd.read_csv("light_data_training.csv")
        testingDF =pd.read_csv("light_data_testing.csv")
        trainingDF['complexity_Type']=trainingDF.complexity_type.map({'fact':0,'explanation':1})
        testingDF['complexity_Type']=testingDF.complexity_type.map({'fact':0,'explanation':1})
    
        corpus = []
        for i in range(0,len(trainingDF)):
            answer =trainingDF['answer'][i]
            answer=self.noun_phrases(answer)
            corpus.append(answer)
      
        vect =CountVectorizer(max_features=600)
        X = vect.fit_transform(corpus).toarray()
        y= trainingDF.complexity_Type    


        X_test= testingDF.answer
        X_testFE= vect.transform(X_test)
        y_test= testingDF.complexity_Type   
    

        logisticRegr = LogisticRegression()
        logisticRegr.fit(X, y)

        y_predFE = logisticRegr.predict(X_testFE)
        acc =metrics.accuracy_score(y_test, y_predFE)
        cm = confusion_matrix(y_test, y_predFE)    
    
        #Part 2 of 3 L-EF vs L-F 
        testingDF['y_predFE']=""
        testingDF['y_predFE']=y_predFE

        trainingDF2=trainingDF.loc[trainingDF.complexity_type=='fact',['complexity_type','answer','complexity_level']]
        testingDF2=testingDF.loc[testingDF.y_predFE==0,['answer','complexity_level']]

        trainingDF2 = trainingDF2.reset_index()
        del trainingDF2['index']

        trainingDF2['complexity_Level']=trainingDF2.complexity_level.map({'L-EF':0,'L-F':1})
        testingDF2['complexity_Level']=testingDF2.complexity_level.map({'L-EF':0,'L-F':1})

        df=testingDF2.dropna(how='any')

        corpus1 = []
        for i in range(0,len(trainingDF2)):
            #print("Inside for loop")
            answer1 =trainingDF2['answer'][i]
            answer1=self.noun_phrases(answer1)
            corpus1.append(answer1)

 
        vect1 =CountVectorizer()
        X1 = vect1.fit_transform(corpus1).toarray()
        y1= trainingDF2.complexity_Level    

        X_test2= df.answer
        X_testEFE= vect1.transform(X_test2)
        y_test2= df.complexity_Level   
    
        logisticRegr1 = LogisticRegression()
        logisticRegr1.fit(X1, y1)

        y_predEFE = logisticRegr1.predict(X_testEFE)
        acc1 =metrics.accuracy_score(y_test2, y_predEFE)
        cm1 = confusion_matrix(y_test2, y_predEFE) 

        #Part3 of 3 L-EE vs L-E
        trainingDF3=trainingDF.loc[trainingDF.complexity_type=='explanation',['complexity_type','answer','complexity_level']]
        testingDF3=testingDF.loc[testingDF.y_predFE==1,['answer','complexity_level']]

        trainingDF3 = trainingDF3.reset_index()
        del trainingDF3['index']

        trainingDF3['complexity_Level']=trainingDF3.complexity_level.map({'L-EE':0,'L-E':1})
        testingDF3['complexity_Level']=testingDF3.complexity_level.map({'L-EE':0,'L-E':1})
        df2=testingDF3.dropna(how='any')
        corpus2 = []
        for i in range(0,len(trainingDF3)):
            answer2 =trainingDF3['answer'][i]
            answer2=self.noun_phrases(answer2)
            corpus2.append(answer2)

        vect2 =CountVectorizer()
        X2 = vect2.fit_transform(corpus2).toarray()
        y2= trainingDF3.complexity_Level    

        X_test3= df2.answer
        X_testEEE= vect2.transform(X_test3)
        y_test3= df2.complexity_Level   

        logisticRegr2 = LogisticRegression()
        logisticRegr2.fit(X2, y2)

        y_predEEE = logisticRegr2.predict(X_testEEE)
        acc2 =metrics.accuracy_score(y_test3, y_predEEE)

        cm2 = confusion_matrix(y_test3, y_predEEE) 
        list_accuracy=[acc,acc1,acc2,cm,cm1,cm2]
        return list_accuracy
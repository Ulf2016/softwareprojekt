'''
CorpusReader(class) to read the germeval corpus. 

 
--------------------------------------------
    | w_1   |   w_2  |   w_3 |  ...  |  w_n  |
--------------------------------------------
t1  |#w_1/t1|        |       |       |       |
--------------------------------------------
... |       |        | ...   |       |  ...  |
--------------------------------------------
tn  |#w_1/tn|        |       |       |#w_n/tn|
--------------------------------------------
'''

import csv
import sys
import string
import json

import numpy as np
from nltk.corpus import stopwords
from autosarkasmus.preprocessor.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

class Baseline_SVM:

    def __init__(self, filename):
        '''
		Constructor of Baseline Class

		Keyword arguments:
		filename (str): path to the corpus file
        '''
        self.this_file = filename
        

    def preprocess(self):
        """
        Preprocessing based on Scheffler et. al. German Twitter Preprocessing
        """
        tokenizedTweets_writer = open('./daten/tokenized_tweets.txt', 'w')
        preprocTweets_writer = open('./daten/preprocessed_tweets.txt', 'w')
        
        pp = Pipeline(self.this_file, "./autosarkasmus/rsrc/de-tiger.map" )
        tweets_tkn, tweets_proc, labels = pp.process()
        assert(len(tweets_tkn) == len(tweets_proc) == len(labels))
        
        # write preprocessing results to file 
        for x in range(len(tweets_proc)):
            t_tweet = (" ").join(tweets_tkn[x])
            p_tweet = (" ").join([str(x) + "/" + str(y) for x,y in tweets_proc[x]])
            label = labels[x]
            tokenizedTweets_writer.write(t_tweet + "\t" + label + "\n")
            preprocTweets_writer.write(p_tweet + "\t" + label + "\n")
        

    def createBaselineClassifier(self):
        """
        Erstellt simplen baseline tf-idf-basierten unigram SVM classifier mit 10-fold cross-validation auf preprocessed germeval.train dataset
        
        """
        tweets = []
        labels = []
        with open('./daten/tokenized_tweets.txt', 'r') as inputfile:
            for line in inputfile:
                l = line.split("\t")
                tweets.append(l[0])
                labels.append(l[1])
        assert(len(tweets) == len(labels))

        vectorizer = CountVectorizer(analyzer='word')
        document_term_matrix = vectorizer.fit_transform(tweets)
        
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(document_term_matrix)
        X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, np.array(labels), test_size=0.2)

        # print(X_train.shape, y_train.shape)
        # print(X_test.shape, y_test.shape)

        svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None)
        svm.fit(X_train, y_train)
        predicted = svm.predict(X_test)
      
        scores = cross_val_score(svm, X_train, y_train, cv=10)
        # print(metrics.classification_report(y_test, predicted))
        print("Cross-validation avg-score:", np.mean(scores))


        # ----------------Generation------------------------  
        # clf = MultinomialNB().fit(X_train_tfidf, labels)
        # X_new_counts = vectorizer.transform(docs_new)
        # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        # predicted = clf.predict(X_new_tfidf)
        # for doc, category in zip(docs_new, predicted):
        #     print('%r => %s' % (doc, category))
         
if __name__ == '__main__':
    c = Baseline_SVM('./daten/germeval/germeval2018.training.txt')
    # c.preprocess()
    c.createBaselineClassifier()

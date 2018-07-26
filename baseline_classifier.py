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
import math
from pathlib import Path
from operator import itemgetter
from random import randrange
from random import seed

import numpy as np
from nltk.corpus import stopwords
from autosarkasmus.preprocessor.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import KFold
from iwnlp.iwnlp_wrapper import IWNLPWrapper

class Baseline_SVM:

    def __init__(self, filename, csv_filename):
        '''
		Constructor of Baseline Class

		Keyword arguments:
		filename (str): path to the corpus file
        '''
        self.this_file = filename
        self.csv_file = csv_filename
        

    def preprocess(self):
        
        tokenizedTweets_writer = open('./daten/tokenized_tweets_normalized.txt', 'w')
        preprocTweets_writer = open('./daten/preprocessed_tweets_normalized.txt', 'w')
        
        pp = Pipeline(self.this_file, "./autosarkasmus/rsrc/de-tiger.map" )
        tweets_tkn, tweets_proc, labels = pp.process()
        assert(len(tweets_tkn) == len(tweets_proc) == len(labels))
        
        # filter stopwords + normalize tokens
        lemmatizer = IWNLPWrapper(lemmatizer_path='daten/IWNLP.Lemmatizer_20170501.json')
        lemmatized_tokens = []
        for x in range(len(tweets_tkn)):
            tweet = []
            for token in tweets_tkn[x]:
                if token.lower() in stopwords.words('german'):
                    continue 
                try:
                    lemma = lemmatizer.lemmatize_plain(token, ignore_case=True)
                    if(lemma):
                        tweet.append(lemma[0])
                    else:
                        tweet.append(token)

                except Exception as e:
                    print(e)
            
            lemmatized_tokens.append(tweet)
           
        assert(len(lemmatized_tokens) == len(tweets_proc) == len(labels))

        # write preprocessing results to file 
        for x in range(len(lemmatized_tokens)):
            t_tweet = (" ").join(lemmatized_tokens[x])
            p_tweet = (" ").join([str(x) + "/" + str(y) for x,y in tweets_proc[x]])
            label = labels[x]
            tokenizedTweets_writer.write(t_tweet + "\t" + label + "\n")
            preprocTweets_writer.write(p_tweet + "\t" + label + "\n")
    
    def read_corpus(self):
        tweets = []
        labels = []
        with open('./daten/tokenized_tweets_normalized.txt', 'r') as inputfile:
            for line in inputfile:
                l = line.split("\t")
                tweets.append(l[0].strip())
                labels.append(l[1].strip())
        assert(len(tweets) == len(labels))
        return tweets, labels

        # tweets = []
        # labels = []
        # with open('./daten/normalized_tweets_final.txt', 'r') as inputfile:
            
        #     for line in inputfile:
        #         tweet = ""
        #         l = line.split("\t")
        #         tweets.append(l[0].strip())
        #         labels.append(l[1].strip())
        #         tweet_raw = l[0].strip()
        #         for t in tweet_raw.split(" "):
        #             w = t.split("/")[0].strip().lower()
        #             if w not in stopwords.words('german'): 
        #                 tweet += " " + w 
        #         tweets.append(tweet.strip())
        #         labels.append(l[1].strip())
        # assert(len(tweets) == len(labels))
        # return tweets, labels

    def get_n_folds(self, X, Y, n=10):
        """
        Soll sicherstellen, dass immer das gleiche Datenset verwendet wird (random seed)
        """
        seed(2)
        X_copy = list(X)
        Y_copy = list(Y)
        fold_size = int(len(X) / n)
        # Tweets + Labels in Tupel zusammenführen
        # Datenset in n folds splitten
        d_split = []
        for i in range(n):
            foldX = []
            foldY = []
            while(len(foldX) < fold_size):
                index = randrange(len(X_copy))
                foldX.append(X_copy.pop(index))
                foldY.append(Y_copy.pop(index))
            d_split.append([foldX, foldY])
        assert(len(d_split) == n)
        return d_split


    def createBaselineClassifier(self, bigram=False):
        """
        Erstellt simplen baseline tf-idf-basierten unigram SVM classifier mit 10-fold cross-validation auf preprocessed germeval.train dataset
        
        """
        
        tweets, labels = self.read_corpus()
        ten_folds = self.get_n_folds(tweets, labels)

        if(bigram):
            print("Baseline: tf-idf bigram SVM")
            self.write_label_to_csv("Baseline: tf-idf bigram SVM")
            bow_transformer = CountVectorizer(analyzer="word", min_df=1, lowercase=False, ngram_range=(1,2))
        else:
            print("Baseline: tf-idf unigram SVM")
            self.write_label_to_csv("Baseline: tf-idf unigram SVM")
            bow_transformer = CountVectorizer(analyzer="word", min_df=2, lowercase=False)
        
        tfidf_transformer = TfidfTransformer()
        document_term_matrix = bow_transformer.fit(tweets)
        vocab = bow_transformer.vocabulary_
        if(bigram):
            print(str(len(vocab)) + " Uni- and Bigrams found")
        else:
            print(str(len(vocab)) + " Unigrams found")

        scores = []
        # Cross validation: 
        for i in range(len(ten_folds)):
            
            test_fold = ten_folds[i]
            train_folds = [fold for x, fold in enumerate(ten_folds) if x != i]
            X_test, Y_test = test_fold[0], test_fold[1]
            X_train, Y_train = [], []
            for fold in range(len(train_folds)):
                X_train.extend(train_folds[fold][0])
                Y_train.extend(train_folds[fold][1])

            assert(len(X_test) == len(Y_test))
            assert(len(X_train) == len(Y_train))
            
            if(bigram):
                vectorizer = CountVectorizer(analyzer="word", min_df=1, vocabulary=vocab, lowercase=False, ngram_range=(1,2))
            else:
                vectorizer = CountVectorizer(analyzer="word", min_df=2, lowercase=False, vocabulary=vocab)
            
            document_term_matrix_tr = vectorizer.fit_transform(X_train).toarray()
            document_term_matrix_te = vectorizer.fit_transform(X_test).toarray()

            X_train_tfidf = tfidf_transformer.fit_transform(document_term_matrix_tr)
            X_test_tfidf = tfidf_transformer.fit_transform(document_term_matrix_te)

            le = preprocessing.LabelEncoder()
            Y_train_enc = le.fit_transform(Y_train)
            Y_test_enc = le.fit_transform(Y_test)
        
            clf = svm.SVC(kernel='linear', C=1.0,random_state=1).fit(X_train_tfidf, Y_train_enc)
            
            predicted = clf.predict(X_test_tfidf)
            # scores.append(clf.score(X_test_tfidf, Y_test_enc))
            # print(metrics.classification_report(Y_test_enc, predicted))  
            score = metrics.f1_score(Y_test_enc, predicted, average='weighted')
            print("Cross Validation #{0} --> avg. weighted F1-Score: {1}".format(i+1, score))
            self.write_output_to_csv(i, X_test, Y_test, X_train, Y_train, score)
            scores.append(score) 

        scores = np.array(scores)
        self.write_score_to_csv(scores)
        print("Total Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    def baseline_mi(self, path, n=2000):
        vocab = []
        counter = 0
        with open('daten/mi_vocab.txt', 'r', encoding='utf-8') as input_file:
            for line in input_file:
                if counter < n:
                    token = line.split("\t")[0].strip().lower()
                    vocab.append(token.split("/")[0].strip())  
                    counter += 1

        print(len(vocab))

        with open(path, "r") as f:
            for line in f:
                sw = line.split()[0].strip()
                if sw not in stopwords.words('german'):
                    vocab.append(line.split()[0].strip())

        print(len(vocab))

        tweets, labels = self.read_corpus()
        ten_folds = self.get_n_folds(tweets, labels)
        print(str(path))
        self.write_label_to_csv("Baseline: feature selection w. mutual information + extended lexicon")
        self.write_label_to_csv(str(path))
        bow_transformer = CountVectorizer(analyzer="word", min_df=1, lowercase=True, vocabulary=set(vocab))
        document_term_matrix = bow_transformer.fit(tweets)
        # print(bow_transformer.vocabulary_)
    
        scores = []
        for i in range(len(ten_folds)):
            
            test_fold = ten_folds[i]
            train_folds = [fold for x, fold in enumerate(ten_folds) if x != i]
            X_test, Y_test = test_fold[0], test_fold[1]
            X_train, Y_train = [], []
            for fold in range(len(train_folds)):
                X_train.extend(train_folds[fold][0])
                Y_train.extend(train_folds[fold][1])

            assert(len(X_test) == len(Y_test))
            assert(len(X_train) == len(Y_train))
            document_term_matrix_tr = bow_transformer.fit_transform(X_train)
            document_term_matrix_te = bow_transformer.fit_transform(X_test)

            le = preprocessing.LabelEncoder()
            Y_train_enc = le.fit_transform(Y_train)
            Y_test_enc = le.fit_transform(Y_test)
        
            clf = svm.SVC(kernel='linear', C=1.5, random_state=1).fit(document_term_matrix_tr, Y_train_enc)
            
            predicted = clf.predict(document_term_matrix_te)
            score = metrics.f1_score(Y_test_enc, predicted, average='weighted')
            scores.append(score)
            self.write_output_to_csv(i, X_test, Y_test, X_train, Y_train, score)
            print("Cross Validation #{0} --> avg. weighted F1-Score: {1}".format(i+1, score))
        self.write_score_to_csv(scores, "Total accuracy")

    def check_token(self, t, lex, label):
        if t.split("/")[0].strip().lower() not in stopwords.words('german') and (t.split("/")[0].strip().lower()) in set(lex):
            if(label == "OFFENSE"):
                print("tp because of " + t)
                return 1        
            elif label == "OTHER":
                print("fp because of " + t)
                return 2
        else:
            return 0
            

    def feature_extended_lexicon(self, path):
        tweets, labels = self.read_corpus()
        
        extended_lexicon = []
        print(len(extended_lexicon))
        with open(path, "r") as f:
            for line in f:
                sw = line.split()[0].strip()
                if sw not in stopwords.words('german'):
                    extended_lexicon.append(line.split()[0].strip().lower())
        print(str(path))
        print(len(extended_lexicon))
        print(len(set(extended_lexicon)))
        tn = 0
        fn = 0
        tp = 0
        fp = 0
        dic = {}
        with open('./daten/normalized_tweets_final.txt', 'r', encoding='utf-8') as preprocessed_tweets:
            counter = 1
            cc = 0
            for line in preprocessed_tweets:
                
                print("tweet " + str(counter))
                tweet = line.split("\t")[0].strip()
                label = line.split("\t")[1].strip()
                if label == "OFFENSE":
                    cc += 1

                try:
                    counter += 1
                    tokenlist = tweet.split(" ")
                    # total_tokens += len(tokenlist)
                    # count how many documents contain the token (avoid duplicate counts)
                    for token in (tokenlist):
                        r = self.check_token(token, extended_lexicon, label)
                        if r == 1:
                            tp += 1
                            break
                        elif r == 2:
                            fp += 1
                            break                    
                                
                                    
                except IndexError as e:
                    print("could not process token " + token)
                    counter += 1
        
        print("TP: " + str(tp))
        print("FP: " + str(fp))
        fn = 1688 - tp
        tn = 3320 - fp
        print("FN: " + str(fn))
        print("TN: " + str(tn))

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*((precision*recall)/(precision+recall))
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("f1: " + str(f1))
        print("accuracy: " + str(accuracy))
        # for k in sorted(dic, key=lambda k: len(dic[k]), reverse=True):
        #     if len(dic[k]) > 1:
        #         print(k, len(dic[k]))

        # with open('./daten/extended_classification_output.txt', 'a') as f:
        #     f.write(str(path) + "\n")
        #     f.write("----------"+ "\n")
        #     f.write("|" + str(tp) + "|" + str(fn) + "|" + "\n")
        #     f.write("----------"+ "\n")
        #     f.write("|" + str(fp) + "|" + str(tn) + "|" + "\n")
        #     f.write("----------"+ "\n")
        #     f.write("Precision: " + str(precision) + "\n")
        #     f.write("Recall: " + str(recall) + "\n")
        #     f.write("F1: " + str(f1) + "\n")
        #     f.write("Accuracy: " + str(accuracy) + "\n")
        #     f.write("#######################################")

        # ten_folds = self.get_n_folds(tweets, labels)
        # print("Extended lexicon: feature selection")
        # self.write_label_to_csv(str(path))

        # bow_transformer = CountVectorizer(analyzer="word", lowercase=True, vocabulary=set(extended_lexicon))
        # document_term_matrix = bow_transformer.fit(tweets)
        # scores = []
        # scores2 = []
        # scores3 = []

        # for i in range(len(ten_folds)):
            
        #     test_fold = ten_folds[i]
        #     train_folds = [fold for x, fold in enumerate(ten_folds) if x != i]
        #     X_test, Y_test = test_fold[0], test_fold[1]
        #     X_train, Y_train = [], []
        #     for fold in range(len(train_folds)):
        #         X_train.extend(train_folds[fold][0])
        #         Y_train.extend(train_folds[fold][1])


        #     assert(len(X_test) == len(Y_test))
        #     assert(len(X_train) == len(Y_train))
        #     document_term_matrix_tr = bow_transformer.fit_transform(X_train)
        #     document_term_matrix_te = bow_transformer.fit_transform(X_test)

        #     le = preprocessing.LabelEncoder()
        #     Y_train_enc = le.fit_transform(Y_train)
        #     Y_test_enc = le.fit_transform(Y_test)
        
        #     clf = svm.SVC(kernel='linear', C=1.0,random_state=1).fit(document_term_matrix_tr, Y_train_enc)
            
        #     predicted = clf.predict(document_term_matrix_te)
        #     score = metrics.f1_score(Y_test_enc, predicted, average='weighted')
        #     score2 = metrics.recall_score(Y_test_enc, predicted, average="weighted")
        #     score3 = metrics.precision_score(Y_test_enc, predicted, average="weighted")
        #     scores.append(score)
        #     scores2.append(score2)
        #     scores3.append(score3)
        #     self.write_output_to_csv(i, X_test, Y_test, X_train, Y_train, score)
        #     print("Cross Validation #{0} --> avg. weighted F1-Score: {1}".format(i+1, score))
        #     # print("Recall --> {0}".format(score2))
        #     # print("Precision --> {0}".format(score3))
        # self.write_score_to_csv(scores, "Total accuracy")
        # self.write_score_to_csv(scores2, "Avg. Recall")
        # self.write_score_to_csv(scores3, "Avg. Precision")


    def feature_selector(self, tweets=[], labels=[]):
        """
        Erstellt feature Vektoren mit mutual information scores für positive und negative Klasse
        """
        
        self.vocab = {}
        self.neg_tokens = {}
        self.pos_tokens = {}

        # iterate over preprocessed/normalized tweets and create vocabulary (include punctuation)
        with open('./daten/preprocessed_tweets.txt', 'r', encoding='utf-8') as preprocessed_tweets:
            for line in preprocessed_tweets:
                tweet = line.split("\t")[0].strip()
                label = line.split("\t")[1].strip()
                try:
                    tokenlist = tweet.split(" ")
                    # total_tokens += len(tokenlist)
                    # count how many documents contain the token (avoid duplicate counts)
                    for token in set(tokenlist):
                        self.set_vocab(token, self.vocab)
                        if label == 'OFFENSE':
                            self.set_vocab(token, self.neg_tokens)
                        else:
                            self.set_vocab(token, self.pos_tokens)
                                    
                except IndexError as e:
                    print("could not process token " + token)
        
        token_by_mi = []
        for v in self.vocab.keys():
            N_11 = self.get_freq(v, self.neg_tokens)
            N_01 = self.get_freq(v, self.pos_tokens)
            N_10 = self.get_neg_freq(v, self.neg_tokens)
            N_00 = self.get_neg_freq(v, self.pos_tokens)
            mi = self.calculate_mi(N_11, N_01, N_10, N_00)
            token_by_mi.append((v, mi))

        sorted_by_mi = sorted(token_by_mi, key=lambda tup: tup[1], reverse=True)
        
        # with open('mi_vocab.txt', 'w', encoding='utf-8') as out:
        #     for x in sorted_by_mi:
        #         out.write(x[0] + "\t" + str(x[1]) + '\n')

    
    def get_freq(self, t, dic):
        try:
            N = len(dic[t])
        except KeyError:
            N = 1
        return N
    
    def get_neg_freq(self, t, dic):
        N = 0
        for key, item in dic.items():
            if(key != t):
                N += len(item)
        return N

    def calculate_mi(self, N_11, N_01, N_10, N_00):
        # berechnet Mutual Information gem. Manning et. al. "Introduction to Information Retrieval" S. 152 f.  
        N = N_00 + N_01 + N_10 + N_11
        mi = (N_11/N)*math.log2((N*N_11)/((N_11 + N_10)*(N_11 + N_01))) + \
        (N_01/N)*math.log2((N*N_01)/((N_01 + N_00)*(N_11 + N_01))) + \
        (N_10/N)*math.log2((N*N_10)/((N_11 + N_10)*(N_10 + N_00))) + \
        (N_00/N)*math.log2((N*N_00)/((N_01 + N_00)*(N_10 + N_00)))
        return mi 

    def set_vocab(self, token, dic):

        if("HASHTAG" in token.split("/")[1]):
            dic.setdefault("HASHTAG", []).append(1)
        elif("MENTION" in token.split("/")[1]):
            dic.setdefault("MENTION", []).append(1)
        elif("SMILEYPOS" in token.split("/")[1]):
            dic.setdefault("SMILEYPOS", []).append(1)
        elif("SMILEYNEG" in token.split("/")[1]):
            dic.setdefault("SMILEYNEG", []).append(1)
        else:
            dic.setdefault(token, []).append(1)
  
    def write_output_to_csv(self, foldNo, x_test, y_test, x_train, y_train, score):
        # calculate basic statistics
        test_instances = 0
        train_instances = 0
        test_label_pos = 0
        test_label_neg = 0
        train_label_pos = 0
        train_label_neg = 0
        train_tokens = 0
        test_tokens = 0
        for i, tweet in enumerate(x_test):
            test_instances += 1
            test_tokens += len(tweet)
            if y_test[i] == 'OFFENSE':
                test_label_pos += 1
            else:
                test_label_neg += 1     

        for i, tweet in enumerate(x_train):
            train_instances += 1
            train_tokens += len(tweet)
            if y_train[i] == 'OFFENSE':
                train_label_pos += 1
            else:
                train_label_neg += 1
            
        # write to csv-file
        with open(self.csv_file, 'a', encoding='utf-8') as output_csv:
            csv_writer = csv.writer(output_csv, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([foldNo+1, train_instances, train_label_pos, train_label_neg, round((train_tokens/train_instances),2), test_instances, test_label_pos, test_label_neg, float(test_tokens/test_instances), score ])

    def write_label_to_csv(self, label):
        with open(self.csv_file, 'a', encoding='utf-8') as output_csv:
            csv_writer = csv.writer(output_csv, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([label])

    def write_score_to_csv(self, scores, m):
        scores = np.array(scores)
        with open(self.csv_file, 'a', encoding='utf-8') as output_csv:
            csv_writer = csv.writer(output_csv, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["", "", "", "", "", "", "", "", "", m + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)])
            

if __name__ == '__main__':
    c = Baseline_SVM('./daten/germeval/germeval2018.training.txt', "extended_lexicon_results_testfinal.csv")
    root_folder = Path("/Users/ulisteinbach/Desktop/SS18/software_projekt/softwareprojekt/daten/results/extracted_words_with_baselist/")
    subfolders = [x for x in root_folder.iterdir() if not x.is_dir()]
    for p in subfolders:
        if ("extracted_words_with_baselist/extended_mfw_200718_075most_frequent_words_seeds_80_100" in (str(p)) or "extracted_words_with_baselist/extended_mfw_200718_080_020most_frequent_words_seeds_100_80" in str(p)):
            c.baseline_mi(p)
    # c.baseline_mi(p)
    # c.baseline_mi()
    # c.preprocess()
    # c.createBaselineClassifier(False)
    # c.createBaselineClassifier(True)
    # c.feature_selector()
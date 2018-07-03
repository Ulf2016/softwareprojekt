import csv
import sys
import string
import json
import math
import ast
import copy 
import statistics
import numpy as np
from operator import itemgetter
from random import randrange
from random import seed
from scipy import spatial


# TXT-FILE einlesen (Baselexikon)

class GraphPreprocessor:

	def __init__(self, path_to_baselexicon, path_to_embeddingfile):
		self.baselexicon = {}
		self.most_freq_wiki = []
		self.basepath = path_to_baselexicon
		self.embeddingraw = path_to_embeddingfile
		self.read_most_frequent_words_from_wikipedia()

	def read_baselexicon(self):
		with open(self.basepath, 'r') as basefile:
			for line in basefile:
				# POS-tag	\t 	abusiveTag
				l = line.split()
				k = l[0].strip()
				t = [l[1].strip(), l[2].strip()]
				self.baselexicon[k] = t
	
	def read_embeddings(self):
		# file not found --> create baselexicon from raw embeddings	
		f = open(self.embeddingraw, 'r')
		for line in f: 
			token = line.split(" ")[0].strip()
			try:
				value = self.baselexicon[token]
				vector = line.split()[1:]
				self.baselexicon[token].append(np.array(vector))
			except KeyError as e:
				continue

	def read_most_frequent_words_from_wikipedia(self):
		words = []
		try:
			f = open("./daten/most_freq_wikipedia_embeddings.txt", 'r')
			print("reading embeddings for most frequent wikipedia words")
			for line in f:
				l = line.split("\t")
				token = l[0].strip()
				v = ast.literal_eval(l[1])
				vector = np.array(v, dtype=np.float)
				self.most_freq_wiki.append((token, vector))
				
		except FileNotFoundError as e:
			print(e)
			print("lookup embeddings for most frequent wikipedia words")
			with open("./daten/most_freq_wikipedia_words.txt", "r") as wiki:
				for line in wiki:
					word = line.strip().lower()
					words.append(word)
			f = open(self.embeddingraw, 'r')
			for l in f: 
				token = l.split(" ")[0].strip()
				if token in words:
					vector = l.split()[1:]
					self.most_freq_wiki.append((token, np.array(vector)))
			print("writing embeddings to file")
			with open("./daten/most_freq_wikipedia_embeddings.txt", "w") as embeddings:
				for t in self.most_freq_wiki:
					embeddings.write(t[0] + "\t" + np.array2string(t[1], separator=',').replace("\n", "") + "\n")

	def add_wikiwords_to_baselexicon(self, N):
		for t in range(N):
			word = self.most_freq_wiki[t][0]
			vector = self.most_freq_wiki[t][1]
			self.baselexicon[word] = ["Dummy_Tag", 2, vector] 

	def write_baselexicon_to_file(self):
		out = open('annotation/baselexicon_embeddings.txt', 'w')
		for key, val in self.baselexicon.items():
			if len(val) == 3:
				#out.write(key + "\t" + val[0] + "\t" + str(val[-1]) + "\t" + str(val[1]) + "\n")
				out.write(key + "\t" + val[0] + "\t" + np.array2string(val[-1], separator=',').replace("\n", "") + "\t" + str(val[1]) + "\n")

	def read_baselexicon_from_file(self):
		self.baselexicon = {}
		try:
			with open('annotation/baselexicon_embeddings.txt', 'r') as basefile:
		 		for line in basefile:
		 			l = line.split("\t")
		 			token = l[0].strip()
		 			tag = l[1].strip()
		 			v = ast.literal_eval(l[2])
		 			vector = np.array(v, dtype=np.float)
		 			abusive = l[3].strip()
		 			self.baselexicon[token] = [tag, abusive, vector] 
		 		return 1
		except FileNotFoundError as e:
			return 0

	def calculate_cosine_similarity(self):
		self.read_baselexicon_from_file()
		print("creating graph input")
		d = {}
		dic = {}
		l = list(self.baselexicon)
	
		for i in range(len(l)):
			key = l[i]
			for j in l[i:]:
				if(key != j):
					sim = 1.0 - (spatial.distance.cosine(self.baselexicon[key][-1], self.baselexicon[j][-1]))
					if(sim >= 0.1):
						d[key + "+" + j] = sim
		# normalize
		key = d.keys()
		values = d.values()
		scores = np.array(list(values))
		median = np.median(scores)
		mean = np.mean(scores)
		
		# for k,v in zip(key, scores):
		# 	if float(v) > mean:
		# 		dic[k] = v

		for k,v in zip(key, scores):
			if abs(float(v) - mean) >= 0.2:
				dic[k] = v

		############### MAD ##########
		# median = np.median(scores)
		# MAD = np.median(np.absolute(scores - median))
		# scores2 = (scores - median) / MAD
		# scores3 = np.clip(scores2, -2.0, 2.0)
		# scores4 = ( (scores3/4.0) + 0.5)
		# # scores4[scores4 == 0] = 0.5
		# dic = {}
		# for k,v in zip(key, scores4):
		# 	if float(v) > 0.2:
		# 		dic[k] = v
		
		li = []
		for d in dic.keys():
			l = d.split("+")
			c = l[0]
			li.append(c)

		self.graph_input = set(li)
		
		# write to file
		self.create_graph_output(dic)


	def create_graph_output(self, d):
		with open('daten/graph_input.txt', 'w') as graph_input_file:
			for key, val in d.items():
				w1, w2 = key.split("+")
				graph_input_file.write(w1 + "\t" + w2 + "\t" + "%0.2f" % (val) + "\n")

	def create_graph_seed_and_goldlabel(self, nSeedsPos, nSeedsNeg, most_freq):
		"""
		Wort \t Label \t Weight
		Extract n words from baselexicon and m < n seed words from abusive words in baselexicon
		"""
		seed(2)
		seeds_pos = []
		seeds_neg = []
		words = []
		abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 1]
		if(most_freq):
			non_abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 2]
			allWords = [word for word, val in self.baselexicon.items()]
		else:
			non_abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 0]
			allWords = [word for word, val in self.baselexicon.items() if int(val[1]) != 2]
		
		assert(nSeedsPos <= len(abusiveWords))
		assert(nSeedsNeg <= len(non_abusiveWords))
		

		while(len(seeds_pos) < nSeedsPos):
			index = randrange(len(abusiveWords))
			word = abusiveWords.pop(index)
			seeds_pos.append(word)
			i = allWords.index(word)
			allWords.pop(i)

		while(len(seeds_neg) < nSeedsNeg):
			index = randrange(len(non_abusiveWords))
			word = non_abusiveWords.pop(index)
			seeds_neg.append(word)
			i = allWords.index(word)
			allWords.pop(i)

		assert((len(seeds_pos) + len(seeds_neg) + len(allWords)) == len(self.baselexicon))
		print(len(seeds_neg), len(seeds_pos), len(allWords), len(self.baselexicon))

		excluded_words = []
		# write to outputfile
		#  <source_node>TAB<target_node>TAB<edge_weight>
		with open('daten/gold_labels.txt', 'w') as goldlabel_file:
			for word in allWords:
				label = int(self.baselexicon[word][1])
				if(label):
					label_string = 'off'
				else:
					label_string = 'neg'
				if(word in self.graph_input):
					goldlabel_file.write(word + "\t" + label_string + "\t" + str(1.0) + "\n")
				else:
					excluded_words.append(word)
		print("excluded " + str(len(excluded_words)) + " words")
		print(excluded_words)

		with open('daten/seeds.txt', 'w') as seeds_file:
			for word in seeds_pos:
				label_string = 'off'
				seeds_file.write(word + "\t" + label_string + "\t" + str(1.0) + "\n")
			for word in seeds_neg: 
				label_string = 'neg'
				seeds_file.write(word + "\t" + label_string + "\t" + str(1.0) + "\n")

# set score normalization
# konnektivität 
# anzahl seeds und nur die häufigsten abusive words (Twitter Korpus Institut)
# Ruppenhofer anschreiben
# Evaluation: Baselexicon - seeds = GL

if __name__ == '__main__':
	g = GraphPreprocessor('/Users/ulisteinbach/Desktop/SS18/software_projekt/softwareprojekt/daten/Baselist/final_list.txt', '/Users/ulisteinbach/Downloads/embed_tweets_de_300M_52D')
	if not (g.read_baselexicon_from_file()):
		print("create baselexicon with embeddings")
		g.read_baselexicon()
		g.read_embeddings()
		g.add_wikiwords_to_baselexicon(60)
		g.write_baselexicon_to_file()
	else:
		print("reading baselexicon from file")
	g.calculate_cosine_similarity()
	g.create_graph_seed_and_goldlabel(40, 60, True)
	
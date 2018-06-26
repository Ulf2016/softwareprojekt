import csv
import sys
import string
import json
import math
import ast
import copy 
import numpy as np
from operator import itemgetter
from random import randrange
from random import seed
from scipy import spatial


# TXT-FILE einlesen (Baselexikon)

class GraphPreprocessor:

	def __init__(self, path_to_baselexicon, path_to_embeddingfile):
		self.baselexicon = {}
		self.basepath = path_to_baselexicon
		self.embeddingraw = path_to_embeddingfile

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

	def write_baselexicon_to_file(self):
		out = open('annotation/baselexicon_embeddings.txt', 'w')
		for key, val in self.baselexicon.items():
			if len(val) == 3:
				out.write(key + "\t" + val[0] + "\t" + np.array2string(val[2], separator=',').replace("\n", "") + "\t" + val[1] + "\n")

	def read_baselexicon_from_file(self):
		try:
			with open('annotation/baselexicon_embeddings.txt', 'r') as basefile:
		 		for line in basefile:
		 			l = line.split("\t")
		 			token = l[0].strip()
		 			tag = l[1].strip()
		 			v = ast.literal_eval(l[2])
		 			vector = np.array(v, dtype=np.float)
		 			abusive = l[3].strip()
		 			self.baselexicon[token] = [tag, vector, abusive] 
		 		return 1
		except FileNotFoundError as e:
			return 0

	def calculate_cosine_similarity(self):
		d = {}
		l = list(self.baselexicon)
	
		for i in range(len(l)):
			key = l[i]
			for j in l[i:]:
				if(key != j):
					sim = spatial.distance.cosine(self.baselexicon[key][1], self.baselexicon[j][1])
					d[key + "+" + j] = sim
		assert(len(d) == ((len(self.baselexicon) * (len(self.baselexicon)-1))/2))
		# write to file
		self.create_graph_output(d)


	def create_graph_output(self, d):
		with open('daten/graph_input.txt', 'w') as graph_input_file:
			for key, val in d.items():
				w1, w2 = key.split("+")
				graph_input_file.write(w1 + "\t" + w2 + "\t" + str(val) + "\n")

	def create_graph_seed_and_goldlabel(self, nGold, nSeedsPos, nSeedsNeg):
		"""
		Wort \t Label \t Weight
		Extract n words from baselexicon and m < n seed words from abusive words in baselexicon
		"""
		seed(2)
		seeds_pos = []
		seeds_neg = []
		words = []
		abusiveWords = [word for word, val in self.baselexicon.items() if int(val[-1]) == 1]
		non_abusiveWords = [word for word, val in self.baselexicon.items() if int(val[-1]) != 1]
		allWords = [word for word, val in self.baselexicon.items()]
		
		assert(nSeedsPos < len(abusiveWords))
		assert(nSeedsNeg < len(non_abusiveWords))
		assert((len(abusiveWords) + len(non_abusiveWords)) == len(self.baselexicon))

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

		while(len(words) < nGold):
			index = randrange(len(allWords))
			word = allWords.pop(index)
			words.append(word)

		assert(len(words) == nGold)

		# write to outputfile
		with open('daten/gold_labels.txt', 'w') as goldlabel_file:
			for word in words:
				goldlabel_file.write(word + "\t" + self.baselexicon[word][-1] + "\t" + str(1.0) + "\n")

		with open('daten/seeds.txt', 'w') as seeds_file:
			for word in seeds_pos: 
				seeds_file.write(word + "\t" + self.baselexicon[word][-1] + "\t" + str(1.0) + "\n")
			for word in seeds_neg: 
					seeds_file.write(word + "\t" + self.baselexicon[word][-1] + "\t" + str(1.0) + "\n")


if __name__ == '__main__':
	g = GraphPreprocessor('/Users/ulisteinbach/Desktop/SS18/software_projekt/softwareprojekt/daten/Baselist/final_list.txt', '/Users/ulisteinbach/Downloads/embed_tweets_de_300M_52D')
	if not (g.read_baselexicon_from_file()):
		print("create baselexicon with embeddings")
		g.read_baselexicon()
		g.read_embeddings()
		g.write_baselexicon_to_file()
	else:
		print("reading baselexicon from file")
	# g.calculate_cosine_similarity()
	g.create_graph_seed_and_goldlabel(100,2, 2)
	
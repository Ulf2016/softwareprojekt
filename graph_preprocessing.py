import csv
import sys, os
import string
import json
import math
import ast
import copy 
import statistics
import argparse
import numpy as np
import subprocess
from operator import itemgetter
from random import randrange
from random import seed
from scipy import spatial


# TXT-FILE einlesen (Baselexikon)

class GraphPreprocessor:

	def __init__(self, path_to_baselexicon, pathoutput, path_to_embeddingfile, cutoff):
		self.baselexicon = {}
		self.most_freq_wiki = []
		self.basepath = path_to_baselexicon
		self.outputpath = pathoutput
		self.cutoff = cutoff
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
				self.baselexicon[token].append(np.array(vector, dtype=np.float))
			except KeyError as e:
				continue

	def read_most_frequent_words_from_wikipedia(self):
		words = []
		try:
			path = self.create_path_if_not_exists(self.outputpath)
			f = open(os.path.join(path, "most_freq_wikipedia_embeddings.txt"), 'r')
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
			path = self.create_path_if_not_exists(self.outputpath)
			with open(os.path.join(path, "most_freq_wikipedia_embeddings.txt"), "w") as embeddings:
				for t in self.most_freq_wiki:
					embeddings.write(t[0] + "\t" + np.array2string(t[1], separator=',').replace("\n", "") + "\n")

	def add_wikiwords_to_baselexicon(self, N):
		for t in range(N):
			try:
				word = self.most_freq_wiki[t][0]
				vector = self.most_freq_wiki[t][1]
				self.baselexicon[word] = ["Dummy_Tag", 2, vector]
			except Exception as e:
				print(e)
			 
	def write_baselexicon_to_file(self, path=""):
		if(path):
			out = open(os.path.join(path, "baselexicon_extended_wikipedia.txt"), 'w')
		else:
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

	def calculate_cosine_similarity(self, path):
		self.read_baselexicon_from_file()
		print("creating graph input")
		d = {}
		dic = {}
		l = list(self.baselexicon)
		try:
			w = []
			f = open(os.path.join(path, "graph_input.txt"), "r")
			for line in f:
				word = line.split("\t")[0].strip()
				w.append(word)
			self.graph_input = set(w)

		except FileNotFoundError as e:
				
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
				if abs(float(v) - mean) >= self.cutoff:
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
		path = self.create_path_if_not_exists(self.outputpath)
		with open(os.path.join(path, 'graph_input.txt'), 'w') as graph_input_file:
			for key, val in d.items():
				w1, w2 = key.split("+")
				graph_input_file.write(w1 + "\t" + w2 + "\t" + "%0.2f" % (val) + "\n")

	def read_most_freq_words(self):
		abusive_words = []
		non_abusive_words = []
		with open('daten/Baselist_counted', 'r') as input_file:
			for line in input_file:
				l = line.split("\t")
				word = l[0]
				label = int(l[2].strip())
				if(label):
					abusive_words.append(word)
				else:
					non_abusive_words.append(word)
		return abusive_words, non_abusive_words

	def create_graph_seed_and_goldlabel_from_most_freq(self, nSeedsPos, nSeedsNeg, path):
		abusiveWords, non_abusiveWords = self.read_most_freq_words()
	
		assert(nSeedsPos <= len(abusiveWords))
		assert(nSeedsNeg <= len(non_abusiveWords))

		seeds_pos = abusiveWords[:nSeedsPos]
		seeds_neg = non_abusiveWords[:nSeedsNeg]

		allWords = [word for word, val in self.baselexicon.items() if not ( (word in seeds_pos) or (word in seeds_neg) ) ]
		# assert(( len(seeds_pos) + len(seeds_neg) + len(allWords) ) == len(self.baselexicon.keys()))
		self.write_seeds_and_goldlabels(allWords, seeds_pos, seeds_neg, path)
		
	def create_graph_seed_and_goldlabel_from_wikipedia(self, nSeedsPos, nSeedsNeg, path):
		seeds_pos = []
		if not self.most_freq_wiki:
			self.read_most_frequent_words_from_wikipedia()
		self.read_baselexicon_from_file()
		self.add_wikiwords_to_baselexicon(nSeedsNeg)
		allWords = [word for word, val in self.baselexicon.items() if int(val[1]) != 2]
		non_abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 2]
		abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 1]
		assert(nSeedsPos <= len(abusiveWords))
		while(len(seeds_pos) < nSeedsPos):
			index = randrange(len(abusiveWords))
			word = abusiveWords.pop(index)
			seeds_pos.append(word)
			i = allWords.index(word)
			allWords.pop(i)
		self.write_seeds_and_goldlabels(allWords, seeds_pos, non_abusiveWords, path)

	def write_seeds_and_goldlabels(self, allWords, seeds_pos, seeds_neg, path):
		# write to outputfile
		#  <source_node>TAB<target_node>TAB<edge_weight>
		
		excluded_words = []
		path = self.create_path_if_not_exists(path)
		with open(os.path.join(path, 'gold_labels.txt'), 'w') as goldlabel_file:
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

		with open(os.path.join(path, 'seeds.txt'), 'w') as seeds_file:
			for word in seeds_pos:
				label_string = 'off'
				seeds_file.write(word + "\t" + label_string + "\t" + str(1.0) + "\n")
			for word in seeds_neg: 
				label_string = 'neg'
				seeds_file.write(word + "\t" + label_string + "\t" + str(1.0) + "\n")


	def create_graph_seed_and_goldlabel(self, nSeedsPos, nSeedsNeg, path):
		"""
		Wort \t Label \t Weight
		Extract n words from baselexicon and m < n seed words from abusive words in baselexicon
		"""
		seed(2)
		seeds_pos = []
		seeds_neg = []
		words = []

		abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 1]
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

		# assert((len(seeds_pos) + len(seeds_neg) + len(allWords)) == len(self.baselexicon))
		print(len(seeds_neg), len(seeds_pos), len(allWords), len(self.baselexicon))
		self.write_seeds_and_goldlabels(allWords, seeds_pos, seeds_neg, path)
	
	def create_path_if_not_exists(self, path):
		if (os.path.exists(path)):
			pass
		else:
			print("creating directory " + path)
			os.makedirs(path)
		return path

	def createConfig(self, graph_file, seed_file, gold_labels_file, iters, verbose, prune_threshold, algo, mu1, mu2, mu3, beta, output_file):
	    #template for the config file. 
	    to_write='''#inputs
	    graph_file = {}
	    seed_file = {}
	    gold_labels_file = {}
	    #Parameters
	    iters = {}
	    verbose = {}
	    prune_threshold = {}
	    algo = {}
	    #Hyperparameters
	    mu1 = {}
	    mu2 = {}
	    mu3 = {}
	    beta = {}
	    output_file = {}
	    '''.format(graph_file, seed_file, gold_labels_file, iters, verbose, 
	    prune_threshold, algo, mu1, mu2, mu3, beta, output_file)
	 
	    with open(os.path.join(self.outputpath,'new_config'), 'w') as write_file:
	        write_file.write(to_write)

# set score normalization
# konnektivität 
# anzahl seeds und nur die häufigsten abusive words (Twitter Korpus Institut)
# Ruppenhofer anschreiben
# Evaluation: Baselexicon - seeds = GL

# Ausbalancieren des Lexikons
# 

def main(argv):

	parser = argparse.ArgumentParser(description="Preprocessing für Graph-Algorithmus")
	parser.add_argument('--outputFolder', '-O', help="Name für outputfolder (wird in daten/outputfolder angelegt)")
	parser.add_argument('--embeddingFile', '-E', help="absoluter Pfad für trained Embeddings-File")
	
	parser.add_argument('--upper', '-U',type=int, help='upper bound for number of seeds range')
	parser.add_argument('--lower', '-L', type=int, help='lower bound for number of seeds range')
	parser.add_argument('--methode','-M', type=int, help='1=seeds nur baselexikon, 2=most-frequent-words seeds aus baselexikon, 3=most-frequent-words seeds aus Wikipedia')
	parser.add_argument('--cutoff', '-C', type=float, default=0.1, help='Parameter: Abweichung der Cosinus-Ähnlichkeit vom MW < Cutoff (default=0.1)')

	try:
		args = parser.parse_args()
		outputpath, embeddingFile, upperBound, lowerBound, methode, cutoff = args.outputFolder, args.embeddingFile, args.upper, args.lower, args.methode, args.cutoff
		if outputpath and embeddingFile and upperBound and lowerBound and methode:
			outputpath = "./daten/" + outputpath.strip("/")
			if not(os.path.exists(embeddingFile)):
				print("Embedding-File nicht gefunden")
			g = GraphPreprocessor('./daten/Baselist/final_list.txt', outputpath, embeddingFile, cutoff)
			if not (g.read_baselexicon_from_file()):
				print("create baselexicon with embeddings")
				g.read_baselexicon()
				g.read_embeddings()
				g.write_baselexicon_to_file()
			else:
				print("reading baselexicon from file")
			
			g.calculate_cosine_similarity(outputpath)
			experiments = []
			for i in range(upperBound):
				neg = upperBound - i
				for j in range(upperBound):
					off = upperBound - j
					if(off>lowerBound and neg > lowerBound):
						experiments.append((off, neg))
			
			out_csv = open(os.path.join(outputpath, "results.csv"), "w")
			writer = csv.writer(out_csv, delimiter=' ', quotechar='"', quoting=csv.QUOTE_ALL)
			for i in range(len(experiments)):
				parameters = experiments[i]

				if(methode == 1):
					path = os.path.join(outputpath, "baselexicon_random_seeds" + str(parameters[0]) + "_" + str(parameters[1]))
					print("Taking random seeds from baselexicon: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel(parameters[0], parameters[1], path)
				elif(methode == 2):
					path = os.path.join(outputpath, "most_frequent_words_seeds_" + str(parameters[0]) + "_" + str(parameters[1]))
					print("Taking most frequent words from baselexicon as seeds: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel_from_most_freq(parameters[0], parameters[1], path)
				elif (methode == 3):
					path = os.path.join(outputpath, "most_frequent_wikipedia_seeds_" + str(parameters[0]) + "_" + str(parameters[1]))
					print("Taking most frequent words from Wikipedia as seeds: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel_from_wikipedia(parameters[0], parameters[1], path)
				else:
					print("Unknown Method : Taking default random seeds from baselexikon")
					path = os.path.join(outputpath, "baselexicon_random_seeds" + str(parameters[0]) + "_" + str(parameters[1]))
					print("Taking random seeds from baselexicon: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel(parameters[0], parameters[1], path)
		
		else:
			sys.exit("please specifiy parameters: --outputfolder / --embeddingFile / --upper / --lower / --methode / --cutoff \nRefer to -h for help" )

	except UnboundLocalError as e:
		print(e)

if __name__ == '__main__':
	main(sys.argv[1:])
	# path_to_graph_algorithm = "/Users/ulisteinbach/Downloads/junto-master/examples/simple"

	# print("processing graph algorithm with standard parameters")
	# absp1 = os.path.abspath(outputpath)
	# absp2 = os.path.abspath(path)
	# g.createConfig(os.path.join(absp1, "graph_input.txt"), os.path.join(absp2, "seeds.txt"), os.path.join(absp2, "gold_labels.txt"), 20, True, 0, "adsorption", 1, 0.02, 0.02, 2, os.path.join(absp2, "graph_output"))
	# p = subprocess.run(['junto', 'config' , os.path.join(absp1,'new_config')], cwd=path_to_graph_algorithm, encoding="437")
		
	
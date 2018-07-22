import csv
import sys, os
import string
import json
import math
import re
import ast
import copy 
import statistics
import argparse
import datetime
import numpy as np
import subprocess
from operator import itemgetter
from random import randrange
from random import seed
from scipy import spatial

class GraphPreprocessor:

	def __init__(self, path_to_baselexicon, mfw_list, pathoutput, path_to_embeddingfile, cutoff, confidence_cutoff):
		self.baselexicon = {}
		self.most_freq_wiki = []
		self.basepath = path_to_baselexicon
		self.mfw_list = mfw_list
		self.outputpath = pathoutput
		self.cutoff = cutoff
		self.confidence_cutoff = confidence_cutoff
		self.embeddingraw = path_to_embeddingfile

	def read_baselexicon(self):
		with open(self.basepath, 'r') as basefile:
			for line in basefile:
				# POS-tag	\t 	abusiveTag
				l = line.split()
				k = l[0].strip().lower()
				t = [l[1].strip(), l[2].strip()]
				self.baselexicon[k] = t
	
	def read_embeddings(self):
		umlaut_pattern = re.compile(".*(ae).*|.*(oe).*|.*(ue).*", re.I)
		f = open(self.embeddingraw, 'r')
		for line in f: 
			token = line.split(" ")[0].strip()
			try:
				if(umlaut_pattern.match(token)):
					token = token.lower().replace("ae", "ä").replace("ue", "ü").replace("oe", "ö")	
				value = self.baselexicon[token]
				vector = line.split()[1:]
				self.baselexicon[token].append(np.array(vector, dtype=np.float))
			except KeyError as e:
				continue

	def read_embeddings_for_wiktionary_words(self, wiktionary_words):
		print("lookup embeddings for " + str(len(wiktionary_words)) + " new negative words from wiktionary")
		before = len(self.baselexicon)
		found = [] 
		umlaut_pattern = re.compile(".*(ae).*|.*(oe).*|.*(ue).*", re.I)
		f = open(self.embeddingraw, 'r')
		for line in f: 
			token = line.split(" ")[0].strip()
			try:
				if(umlaut_pattern.match(token)):
					token = token.lower().replace("ae", "ä").replace("ue", "ü").replace("oe", "ö")	
				vector = np.array(line.split()[1:], dtype=np.float)
				value = wiktionary_words[token]
				if(len(vector) == 52):
					self.baselexicon[token] = ["X", "0", vector]
					found.append(token)
			except KeyError as e:
				continue
		after = len(self.baselexicon)
		
		print("Found embeddings for " + str(len(set(found))) + " Words")
		print("Extended embeddings baselexicon from " + str(before) + " words to " + str(after) + " words")
		

	# def read_most_frequent_words_from_wikipedia(self):
	# 	words = []
	# 	try:
	# 		path = self.create_path_if_not_exists(self.outputpath)
	# 		f = open(os.path.join(path, "most_freq_wikipedia_embeddings.txt"), 'r')
	# 		print("reading embeddings for most frequent wikipedia words")
	# 		for line in f:
	# 			l = line.split("\t")
	# 			token = l[0].strip()
	# 			v = ast.literal_eval(l[1])
	# 			vector = np.array(v, dtype=np.float)
	# 			self.most_freq_wiki.append((token, vector))
				
	# 	except FileNotFoundError as e:
	# 		print(e)
	# 		print("lookup embeddings for most frequent wikipedia words")
	# 		with open("./daten/most_freq_wikipedia_words.txt", "r") as wiki:
	# 			for line in wiki:
	# 				word = line.strip().lower()
	# 				words.append(word)
	# 		f = open(self.embeddingraw, 'r')
	# 		for l in f: 
	# 			token = l.split(" ")[0].strip()
	# 			if token in words:
	# 				vector = l.split()[1:]
	# 				self.most_freq_wiki.append((token, np.array(vector)))
	# 		print("writing embeddings to file")
	# 		path = self.create_path_if_not_exists(self.outputpath)
	# 		with open(os.path.join(path, "most_freq_wikipedia_embeddings.txt"), "w") as embeddings:
	# 			for t in self.most_freq_wiki:
	# 				embeddings.write(t[0] + "\t" + np.array2string(t[1], separator=',').replace("\n", "") + "\n")

	# def add_wikiwords_to_baselexicon(self, N):
	# 	for t in range(N):
	# 		try:
	# 			word = self.most_freq_wiki[t][0]
	# 			vector = self.most_freq_wiki[t][1]
	# 			self.baselexicon[word] = ["Dummy_Tag", 2, vector]
	# 		except Exception as e:
	# 			print(e)
			 
	def write_baselexicon_to_file(self, path=""):
		print("writing baselexicon to file")
		if not (os.path.exists(path)):
			path = self.create_path_if_not_exists(path)
		self.noEmbeddings = []
		out = open(path, 'w')
		for key, val in self.baselexicon.items():
			if len(val) == 3:
				#out.write(key + "\t" + val[0] + "\t" + str(val[-1]) + "\t" + str(val[1]) + "\n")
				out.write(key + "\t" + val[0] + "\t" + np.array2string(val[-1], separator=',').replace("\n", "") + "\t" + str(val[1]) + "\n")
			else:
				self.noEmbeddings.append(key)
		
	def read_baselexicon_from_file(self, path):
		try:
			with open(path, 'r') as basefile:
		 		for line in basefile:
		 			l = line.split("\t")
		 			token = l[0].strip()
		 			tag = l[1].strip()
		 			v = ast.literal_eval(l[2])
		 			vector = np.array(v, dtype=np.float)
		 			abusive = l[3].strip()
		 			self.baselexicon[token] = [tag, abusive, vector] 
		 		print("read baselexicon with " + str(len(self.baselexicon)) + " words")
		 		return 1
		except FileNotFoundError as e:
			return 0

	def calculate_cosine_similarity(self, path, tofile=False):

		self.create_path_if_not_exists(path)
		print("Creating graph input")

		try:
			print("Check for existing graph input file")
			w = []
			f = open(path, "r")
			for line in f:
				word = line.split("\t")[0].strip()
				w.append(word)
			self.graph_input = set(w)

		except FileNotFoundError as e:
			print("No existing graph input file with same parameters found")
			print("Create new graph input file")
			d = {}
			dic = {}
			l = list(self.baselexicon)
			counter = 0
			if tofile:
				out = open(path, "w")
			print("Calculating cosine similarity between words")
			for i in range(len(l)):
				counter += 1
				key = l[i]
				print("Word " + str(counter))
				for j in l[i:]:
					if(key != j):
						try:
							sim = 1.0 - (spatial.distance.cosine(self.baselexicon[key][-1], self.baselexicon[j][-1]))
							if(sim >= 0.1):
								if tofile:
									out.write(key + "\t" + j + "\t" + str(sim) + "\n")
								else: 
									d[key + "+" + j] = sim

						except Exception as e:
							continue

			if tofile:
				total = 0
				counter = 0
				f = open(path, "r")
				for line in f:
					l = line.split("\t")
					counter += 1
					total += float(l[2].strip())
				mw = total/counter
				f = open(path, "r")
				for line in f:
					l = line.split("\t")
					if abs(float(l[2].strip()) - mw) >= self.cutoff:
						dic[l[0] + "+" + l[1]] = float(l[2].strip())
			else:

				# normalize
				print("calculate median value")
				key = d.keys()
				values = d.values()
				scores = np.array(list(values))
				median = np.median(scores)
				mean = np.mean(scores)
			
				print("reducing graph density")
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
			print("writing graph input file")
			# write to file
			self.create_graph_output(dic, os.path.dirname(path))


	def create_graph_output(self, d, path):
		with open(os.path.join(path, 'graph_input.txt'), 'w') as graph_input_file:
			for key, val in d.items():
				w1, w2 = key.split("+")
				graph_input_file.write(w1 + "\t" + w2 + "\t" + "%0.2f" % (val) + "\n")

	def read_most_freq_words(self):
		abusive_words = []
		non_abusive_words = []
		try:
			with open(self.mfw_list, 'r') as input_file:
				for line in input_file:
					l = line.split("\t")
					word = l[0]
					try:
						self.baselexicon[word]
					except KeyError as e:
						continue
					label = int(l[2].strip())
					if(label):
						abusive_words.append(word)
					else:
						non_abusive_words.append(word)
			return abusive_words, non_abusive_words
		except FileNotFoundError as e:
			print(e)
			print("Method needs a file with most frequent words")
			exit()
		

	def create_pool_of_words(self, w1, w2, pool_size):
		assert(len(w1) >= pool_size)
		pool_pos = w1[:pool_size]
		assert(len(w2) >= pool_size)
		pool_neg = w2[:pool_size]
		return pool_pos, pool_neg

	def create_graph_seed_and_goldlabel_from_most_freq(self, nSeedsPos, nSeedsNeg, path):
		
		abusiveWords, non_abusiveWords = self.read_most_freq_words()
		pool_pos, pool_neg = self.create_pool_of_words(abusiveWords, non_abusiveWords, 100)
		# separate pool of words for each class
		
		assert(nSeedsPos <= len(pool_pos))
		assert(nSeedsNeg <= len(pool_neg))
		allWords = [word for word, vector in self.baselexicon.items() if vector[0] != "X"]
		xWords = [word for word, vector in self.baselexicon.items() if vector[0] == "X"]
		seeds_pos = pool_pos[:nSeedsPos]
		seeds_neg = pool_neg[:nSeedsNeg]

		allWords = (set(allWords) - set(pool_pos)) - set(pool_neg)
		
		assert(( len(pool_pos) + len(pool_neg) + len(allWords) + len(xWords) ) == len(self.baselexicon.keys()))
	
		self.write_seeds_and_goldlabels(allWords, seeds_pos, seeds_neg, path)
		
	# def create_graph_seed_and_goldlabel_from_wikipedia(self, nSeedsPos, nSeedsNeg, path):
	# 	seeds_pos = []
	# 	if not self.most_freq_wiki:
	# 		self.read_most_frequent_words_from_wikipedia()
	# 	self.read_baselexicon_from_file()
	# 	self.add_wikiwords_to_baselexicon(100)
		
	# 	allWords = [word for word, val in self.baselexicon.items() if (int(val[1]) != 2) ]

	# 	non_abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 2]
	# 	abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 1]
		
	# 	pool_pos, pool_neg = self.create_pool_of_words(abusiveWords, non_abusiveWords, 100)

	# 	assert(nSeedsPos <= len(pool_pos))
	# 	seeds_pos = pool_pos[:nSeedsPos]
	# 	seeds_neg = pool_neg[:nSeedsNeg]

	# 	allWords = (set(allWords) - set(pool_pos)) - set(pool_neg)
	# 	print(len(allWords), len(self.baselexicon.keys()))
	# 	assert(( len(pool_pos) + len(pool_neg) + len(allWords) ) == len(self.baselexicon.keys()))

	# 	self.write_seeds_and_goldlabels(allWords, seeds_pos, seeds_neg, path)

	def write_seeds_and_goldlabels(self, allWords, seeds_pos, seeds_neg, path):
		# write to outputfile
		#  <source_node>TAB<target_node>TAB<edge_weight>
		
		pathNew = self.create_path_if_not_exists(os.path.join(path, 'gold_labels.txt'))
		with open(pathNew, 'w') as goldlabel_file:
			for word in allWords:
				label = int(self.baselexicon[word][1])
				if(label == 1):
					label_string = 'off'
				elif(label == 0):
					label_string = 'neg'
				else:
					continue
				# if(word in self.graph_input):
				goldlabel_file.write(word + "\t" + label_string + "\t" + str(1.0) + "\n")
		# 		else:
		# 			excluded_words.append(word)
		# print("excluded " + str(len(excluded_words)) + " words")
		# print(excluded_words)

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
		
		seeds_pos = []
		seeds_neg = []
		words = []

		abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 1]
		non_abusiveWords = [word for word, val in self.baselexicon.items() if int(val[1]) == 0]
		allWords = [word for word, val in self.baselexicon.items() if val[0] != "X"]
		xWords = [word for word, val in self.baselexicon.items() if val[0] == "X"]
		print(len(abusiveWords), len(non_abusiveWords))
		pool_pos, pool_neg = self.create_pool_of_words(abusiveWords, non_abusiveWords, 100)

		assert(nSeedsPos <= len(pool_pos))
		assert(nSeedsNeg <= len(pool_neg))
		
		seeds_pos = pool_pos[:nSeedsPos]
		seeds_neg = pool_neg[:nSeedsNeg]

		allWords = (set(allWords) - set(pool_pos)) - set(pool_neg)
		
		assert(( len(pool_pos) + len(pool_neg) + len(allWords) + len(xWords) ) == len(self.baselexicon.keys()))

		self.write_seeds_and_goldlabels(allWords, seeds_pos, seeds_neg, path)
	

	def extend_baselexicon(self, path1, path2):
		try:
			print("Trying to read extended lexicon from file")
			if(self.read_baselexicon_from_file(path1)):
				pass
			else:
				raise FileNotFoundError
			
		except FileNotFoundError as e:
			print("extended baselexicon_embeddings_X.txt not found")
			print("create baselexicon_embeddings_X.txt from wiktionary negative words")
			wiktionary_words = {}
			counter = 0
			with open(path2, "r") as wiktionary:
				for line in wiktionary:
					token = line.split("\t")[0].lower()
					v = line.split("\t")[2].replace("[", "").replace("]", "")
					confidence_score = np.fromstring(v, dtype=np.float, sep=" ")
					if token in self.baselexicon:
						counter += 1
						continue
					elif confidence_score[0] >= self.confidence_cutoff:
						wiktionary_words[token] = confidence_score[0]
			print(str(counter) + " words already found in baselexicon")
			print(str(len(wiktionary_words)) + " new negative words from wiktionary")
			self.read_embeddings_for_wiktionary_words(wiktionary_words)
			self.write_baselexicon_to_file(path1)


	def create_path_if_not_exists(self, path):
		if (os.path.exists(path)):
			pass
		else:
			if not (os.path.exists(os.path.dirname(path))):
				print("creating directories for " + path)
				os.makedirs(os.path.dirname(path))
		return path



def main(argv):

	parser = argparse.ArgumentParser(description="Preprocessing für Graph-Algorithmus")
	parser.add_argument('--outputFolder', '-O', help="Name für outputfolder (wird in daten/outputfolder angelegt)")
	parser.add_argument('--embeddingFile', '-E', help="absoluter Pfad für trained Embeddings-File")
	parser.add_argument('--upper', '-U',type=int, help='upper bound for number of seeds range')
	parser.add_argument('--lower', '-L', type=int, help='lower bound for number of seeds range')
	parser.add_argument('--methode','-M', type=int, help='1=seeds nur baselexikon, 2=most-frequent-words seeds aus baselexikon, 3=most-frequent-words seeds aus Wikipedia')
	parser.add_argument('--cutoff', '-C', type=float, default=0.1, help='Parameter: Abweichung der Cosinus-Ähnlichkeit vom MW < Cutoff (default=0.1)')
	parser.add_argument('--extend', '-X', help="absoluter Pfad für wiktionary Liste mit negativen Wörtern")
	parser.add_argument('--confidence_cutoff', '-cc', type=float, default=0.75, help="confidence score cutoff")
	parser.add_argument('--baselist', '-B', help="Pfad zu annotierter Wortliste (final_list.txt)")
	parser.add_argument('--most_frequent', '-F', help="Pfad zu most frequent words Liste")
	try:
		args = parser.parse_args()
		outputpath, embeddingFile, upperBound, lowerBound, methode, cutoff, extend, confidence_cutoff, baselist, mfw_list = args.outputFolder, args.embeddingFile, args.upper, args.lower, args.methode, args.cutoff, args.extend, args.confidence_cutoff, args.baselist, args.most_frequent
		if outputpath and embeddingFile and upperBound and lowerBound and methode and baselist and mfw_list:
			outputp = os.path.join(outputpath, "daten")
			if not(os.path.exists(embeddingFile)):
				print("#############################")
				print("Embedding-File nicht gefunden")
				exit()

			if not (os.path.exists(baselist)):
				print("#############################")
				print("Annotierte Wortliste nicht gefunden")
				exit()

			if not (os.path.exists(mfw_list)):
				print("#############################")
				print("Most frequent Wortliste nicht gefunden")
				exit()

			g = GraphPreprocessor(baselist, mfw_list, outputp, embeddingFile, cutoff, confidence_cutoff)
			
			if not (g.read_baselexicon_from_file(os.path.join(outputpath, "annotation/baselexicon_embeddings.txt"))):

				print("##################################")
				print("create baselexicon with embeddings")
				g.read_baselexicon()
				g.read_embeddings()
				g.write_baselexicon_to_file(os.path.join(outputpath, "annotation/baselexicon_embeddings.txt"))
				print("could not find embeddings for " + str(len(g.noEmbeddings)) + " words")
				print(g.noEmbeddings)		
		
			#Graph Erweiterung
			if(os.path.exists(extend)):
				filename = "annotation/baselexicon_embeddings_X_" + str(g.confidence_cutoff) + ".txt"
				g.extend_baselexicon(os.path.join(outputpath, filename), extend)
			
			experiments = []
			for i in range(lowerBound, upperBound + 10, 10):
				for j in range(lowerBound, upperBound + 10, 10):
					experiments.append((i, j))
			
			# out_csv = open(os.path.join(outputpath, "results.csv"), "w")
			# writer = csv.writer(out_csv, delimiter=' ', quotechar='"', quoting=csv.QUOTE_ALL)
			for i in range(len(experiments)):
				parameters = experiments[i]

				if(methode == 1):
					# pathRandom : /Users/ulisteinbach/Desktop/newTest/daten/random_2018_07_22_01_099/random_seeds_80_80
					pathRandom = os.path.join(os.path.join(outputp, "random_" + datetime.datetime.now().strftime("%Y_%m_%d") + "_" + str(cutoff).replace(".", "") + "_" + str(confidence_cutoff).replace(".", "")), "random_seeds_" + str(parameters[0]) + "_" + str(parameters[1]))				
					g.calculate_cosine_similarity(os.path.join(os.path.dirname(pathRandom), "graph_input.txt"), True)
					print("Taking random seeds from baselexicon: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel(parameters[0], parameters[1], pathRandom)
				elif(methode == 2):
					pathMFW = os.path.join(os.path.join(outputp, "mfw_" + datetime.datetime.now().strftime("%Y_%m_%d") + "_" + str(cutoff) + "_" + str(confidence_cutoff)), "most_frequent_words_seeds_" + str(parameters[0]) + "_" + str(parameters[1]))
					g.calculate_cosine_similarity(os.path.join(os.path.dirname(pathMFW), "graph_input.txt"), True)
					print("Taking most frequent words from baselexicon as seeds: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel_from_most_freq(parameters[0], parameters[1], pathMFW)
			# 	elif (methode == 3):
			# 		path = os.path.join(outputpath, "most_frequent_wikipedia_seeds_" + str(parameters[0]) + "_" + str(parameters[1]))
			# 		print("Taking most frequent words from Wikipedia as seeds: processing experiment " + str(i) + " with parameters " + str(parameters))
			# 		g.create_graph_seed_and_goldlabel_from_wikipedia(parameters[0], parameters[1], path)
				else:
					print("Unknown Method : Taking default random seeds from baselexikon")
					path = os.path.join(outputp, "baselexicon_random_seeds" + str(parameters[0]) + "_" + str(parameters[1]))
					print("Taking random seeds from baselexicon: processing experiment " + str(i) + " with parameters " + str(parameters))
					g.create_graph_seed_and_goldlabel(parameters[0], parameters[1], path)
		
		else:
			sys.exit("please specifiy parameters: --outputfolder / --embeddingFile / --upper / --lower / --methode / --cutoff \nRefer to -h for help" )

	except Exception as e:
		print(e)

if __name__ == '__main__':
	main(sys.argv[1:])
	
	
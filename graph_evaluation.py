import csv
import sys
import string
import json
import math
import ast
import copy 
import statistics
import re
import numpy as np
from operator import itemgetter
from random import randrange
from random import seed
from scipy import spatial


class GraphEvaluator:

	def __init__(self):
		self.graph_output_regex = re.compile('([a-zßöäü]+)\W[(?:off\W+1\.0\W+off\W+1\.0\W+)|(?:neg\W+1\.0\W+neg\W+1\.0\W+)]*(?:(off\W+[0-9].[0-9]+\W+)|(neg\W+[0-9].[0-9]+\W+))*(__DUMMY__\W+[0-9].[0-9]+)\W+(true|false)')

	def read_graph_outputfile(self):
		d = {}
		off = 0
		neg = 0
		dummy = 0
		with open('/Users/ulisteinbach/Downloads/junto-master/examples/simple/data/label_prop_output', 'r') as infile:
			for line in infile:
				line = line.strip()
				regex = self.graph_output_regex.match(line)
				if not (regex):
					continue
				word = regex.group(1).strip()
				score_off = float(regex.group(2).split(" ")[1])
				score_neg = float(regex.group(3).split(" ")[1])
				score_dummy = float(regex.group(4).split(" ")[1])
				max_score = max(score_off, score_neg, score_dummy)
				if(max_score == score_off):
					off += 1
					d[word] = [score_off, 'off']
				elif(max_score == score_neg):
					neg += 1
					d[word] = [score_neg, 'neg']
				else:
					dummy += 1
					d[word] = [score_dummy, 'dummy']

		# for word, score in d.items():
		# 	print(word + "\t" + str(score))
		print("offensive: " + str(off))
		print("negative: " + str(neg))
		print("dummy: " + str(dummy))
									

	
if __name__ == '__main__':
	g = GraphEvaluator()
	g.read_graph_outputfile()
	
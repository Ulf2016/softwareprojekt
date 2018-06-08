from random import randrange
from random import seed
import sys

def create_annotation_testset(path_to_baselist, n=50):
	"""
	Inputfile: Baselist 
	"""
	nouns = []
	verbs = []
	adj = []
	final_list = []
	out = open('./annotation_testset.txt', 'w')
	with open(path_to_baselist, 'r', encoding="utf-8") as inputfile:
		for line in inputfile:
			token = line.split("|")[0].strip()
			tag = line.split("|")[1].strip()
			if tag == "NOUN":
				nouns.append(line)
			elif tag == "VERB":
				verbs.append(line)
			elif tag == "ADJ":
				adj.append(line)
	for i in range(n):
		index_noun = randrange(len(nouns))
		index_verb = randrange(len(verbs))
		index_adj = randrange(len(adj))
		final_list.append(nouns.pop(index_noun))
		final_list.append(verbs.pop(index_verb))
		final_list.append(adj.pop(index_adj))

	for token in sorted(final_list, key=lambda x: x.split("|")[1].strip()):
		out.write(token)

if __name__ == '__main__':
	# Pfad zum baselist Lexikon als commandline argument
	create_annotation_testset(sys.argv[1], 50)

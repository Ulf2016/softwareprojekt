import os
import operator

#defaults:
graph_file = 'data/graph_input.txt'
seed_file = 'data/seeds.txt'
gold_labels_file = 'data/gold_labels.txt'
iters = 10
verbose = 'true'
prune_threshold = 0
algo = 'adsorption' #Possible: (lp_zgl), adsorption, mad

mu1 = 1
mu2 = 0.01
mu3 = 0.01
beta = 2

output_file = 'data/output/label_prop_output'

evaluation_file = 'data/evaluation.txt'


def createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
    prune_threshold, algo, mu1, mu2, mu3, beta, output_file):
    #inputs we dont need to change:
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

    with open('new_config', 'w+') as write_file:
        write_file.write(to_write)

count = 0
e = []
for i in range(10):
    mu2 = float(i+1)/1000
    for j in range(10):
        mu3 = float(j+1)/1000
        createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
            prune_threshold, algo, mu1, mu2, mu3, beta, output_file+str(count))
        os.system('junto config new_config')
        correct = 0
        false = 0
        with open(output_file+str(count), 'r') as read_file:
            for line in read_file.read().splitlines():
                line = line.split('\t')
                if(len(line[1])>0):  
                    if(line[1].split()[0] == line[3].split()[2]):
                        correct += 1
                    else:
                        false += 1
        e.append([mu2, mu3, correct, false, float(correct)/(float(false)+float(correct))])
        

        count+=1

with open(evaluation_file, 'w') as write_file:
    write_file.write('[mu2, mu3, correct, false, correct%]\n')
    s = sorted(e, key=lambda e: e[2], reverse=True)
    for i in s:
        write_file.write(str(i) + '\n')

    




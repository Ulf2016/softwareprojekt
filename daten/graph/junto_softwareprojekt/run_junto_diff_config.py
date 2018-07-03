import os
import operator

#defaults:
graph_file = 'data/graph_input.txt'
seed_file = 'data/seeds.txt'
gold_labels_file = 'data/gold_labels.txt'
iters = 20
verbose = 'true'
prune_threshold = 0
algo = 'adsorption' #Possible: (lp_zgl), adsorption, mad

mu1 = 1
mu2 = 0.01
mu3 = 0.02
beta = 2

output_file = 'data/output/label_prop_output'

evaluation_file = 'data/evaluation.txt'


def createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
    prune_threshold, algo, mu1, mu2, mu3, beta, output_file):
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

    with open('new_config', 'w+') as write_file:
        write_file.write(to_write)

count = 0
e = []
for i in range(10):
    mu2 = float(i+1)/10000
    for j in range(10):
        mu3 = float(j+2)/10
        createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
            prune_threshold, algo, mu1, mu2, mu3, beta, output_file+'mu2'+str(mu2)+'mu3'+str(mu3))
        if(not os.path.isfile(output_file+'mu2'+str(mu2)+'mu3'+str(mu3))):
            os.system('junto config new_config')
        correctoff = 0
        falseoff = 0

        correctneg = 0
        falseneg = 0

        neg = 0
        off = 0
        with open(output_file+'mu2'+str(mu2)+'mu3'+str(mu3), 'r') as read_file:
            for line in read_file.read().splitlines():
                line = line.split('\t')
                if(len(line[1])>0):  
                    if(line[3].split()[2]=='neg'):
                        neg += 1
                    else:
                        off += 1
                    if(line[1].split()[0] == line[3].split()[0]):
                        if(line[3].split()[0] == 'neg'):
                            correctneg += 1
                        else: 
                            correctoff += 1
                    else:
                        if(line[3].split()[0] == 'neg'):
                            falseneg += 1
                        else:
                            falseoff += 1

        negoff = 'neg: ' + str(neg) + '/off: ' + str(off)
        e.append([mu2, mu3, correctneg, falseneg, correctoff, falseoff, float(correctoff)/(float(falseoff)+float(correctoff)), negoff])
        

        count+=1

with open(evaluation_file, 'w') as write_file:
    write_file.write('[mu2, mu3, correctneg, falseneg, correctoff, falseoff, correctoff%, negative/offensive]\n')
    s = sorted(e, key=lambda e: e[6], reverse=True)
    for i in s:
        write_file.write(str(i) + '\n')

    




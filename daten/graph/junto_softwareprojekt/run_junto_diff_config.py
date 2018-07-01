import os

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
for i in range(9):
    mu2 = float(i+1)/100
    print(mu2)
    for j in range(9):
        mu3 = float(j+1)/100
        print(mu3)
        createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
            prune_threshold, algo, mu1, mu2, mu3, beta, output_file+str(count))
        os.system('junto config new_config')
        count+=1


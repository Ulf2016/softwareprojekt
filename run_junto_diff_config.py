import sys, os
import operator
import argparse
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="Ausfuehrung des Adsorption Algorithmus mit verschiedenen Inputs")
    parser.add_argument('--rootFolder', '-R', help="absoluter Pfad zu der einzulesenden Ordnerstruktur mit den Input-Daten.")
    parser.add_argument('--range_mu2','-mu2', type=float, nargs='+', default=[0.001, 0.006, 0.01, 0.06, 0.1, 0.6, 1.0], help='Range zum testen fuer Hyperparameter mu2')
    parser.add_argument('--range_mu3','-mu3', type=float, nargs='+', default=[0.001, 0.006, 0.01, 0.06, 0.1, 0.6, 1.0], help='Range zum testen fuer Hyperparameter mu3')
    parser.add_argument('--iter','-i', type=int, default=15, help='Anzahl der Iterationen fuer Algorithmus')
    parser.add_argument('--algo','-a', default='mad', help='Welcher Algorithmus? (mad oder adsorption)')

    try:
        #getting Parameters from Command Line
        args = parser.parse_args()
        root_folder, iters, algo = Path(args.rootFolder), args.iter, args.algo
        r_mu2, r_mu3 = args.range_mu2, args.range_mu3

        graph_file = root_folder / 'graph_input.txt'
        results = root_folder / 'results.csv'
        write_results = True
        subfolders = [x for x in root_folder.iterdir() if x.is_dir()]

        #defaults
        verbose = 'true'
        prune_threshold = 0
        mu1 = 1
        beta = 2

        #iterating over subfolders of root_folder
        for p in subfolders:
            seed_file = p / 'seeds.txt'
            gold_labels_file = p / 'gold_labels.txt'
            evaluation = p / 'evaluation.txt'
            output_path = p / 'output/'
            if(not output_path.is_dir()):
                os.makedirs(output_path)
            run_junto(graph_file, seed_file, gold_labels_file, iters, verbose, prune_threshold, algo, mu1, r_mu2, r_mu3, beta, output_path, p, evaluation, write_results, results)



    except UnboundLocalError as e:
        print(e)


def createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
    prune_threshold, algo, mu1, mu2, mu3, beta, output_file, root_folder):
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

    with open(root_folder / 'new_config', 'w+') as write_file:
        write_file.write(to_write)



def run_junto(graph_file, seed_file, gold_labels_file, iters, verbose, prune_threshold, algo, mu1, r_mu2, r_mu3, beta, output_path, p, evaluation, write_results, results):

    count = 0
    e = []
    for i in r_mu2:
        mu2 = i
        for j in r_mu3:
            mu3 = j
            output_file = output_path / 'output-mu2-{}-mu3-{}'.format(mu2, mu3)
            createConfig(graph_file, seed_file, gold_labels_file, iters, verbose, 
                prune_threshold, algo, mu1, mu2, mu3, beta, output_file, p)
            if(not Path.exists(output_file)):
                config_path = p / 'new_config'
                os.system('junto config ' + config_path.__str__())
            correctoff = 0
            falseoff = 0

            correctneg = 0
            falseneg = 0

            dummy = 0
            with open(output_file, 'r') as read_file:
                for line in read_file.read().splitlines():
                    line = line.split('\t')
                    if(len(line[1])>0):  
                        if(line[1].split()[0] == line[3].split()[0]):
                            if(line[1].split()[0] == 'neg'):
                                correctneg += 1
                            elif(line[1].split()[0] == 'off'): 
                                correctoff += 1
                            else:
                                print('wat')
                        else:
                            if(line[1].split()[0] == 'neg'):
                                falseneg += 1
                            elif(line[1].split()[0] == 'off'):
                                falseoff += 1
                            else:
                                dummy+=1 
                                falseneg += 1
                                falseoff += 1

            off = falseoff+correctoff
            neg = falseneg+correctneg
            all_ = correctneg+correctoff+falseneg+falseoff
            allcorrect = correctneg+correctoff
            negoff = 'neg: ' + str(neg) + '/off: ' + str(off) + '/dummy:' + str(dummy)

            peroff = float(correctoff)/float(off)
            perneg = float(correctneg)/float(neg)
            perall = float(allcorrect)/float(all_)
            e.append([mu2, mu3, correctneg, falseneg, perneg, correctoff, falseoff, peroff, perall])
            

            count+=1

    s = sorted(e, key=lambda e: e[8], reverse=True)
    if(write_results and len(s) != 0):
        with open(results, 'a') as result_file:
            result_file.write(p.stem + ', ')
            result_file.write(str(s[0]).strip('[]')+'\n')

    with open(evaluation, 'w') as write_file:
        write_file.write('[mu2, mu3, correctneg, falseneg, correctneg%, correctoff, falseoff, correctoff%, correct%]\n')
        for i in s:
            write_file.write(str(i) + '\n')

if __name__ == '__main__': 
    main(sys.argv[1:])





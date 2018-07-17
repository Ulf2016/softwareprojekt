import sys, os
import operator
import argparse
from pathlib import Path
import numpy as numpy
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
        results = root_folder / 'report'
        write_results = True
        subfolders = [x for x in root_folder.iterdir() if x.is_dir()]

        #defaults
        verbose = 'true'
        prune_threshold = 0
        mu1 = 1
        beta = 2

        #iterating over subfolders of root_folder
        report, for_sorting, confu, f1 = [], [], [], []
        for i, p in enumerate(subfolders):
            print(i+1,'/',len(subfolders))
            seed_file = p / 'seeds.txt'
            gold_labels_file = p / 'gold_labels.txt'
            evaluation = p / 'evaluation.txt'
            output_path = p / 'output/'
            if(not output_path.is_dir()):
                Path.mkdir(output_path)
            tmp_rep, tmp_sort, tmp_confu, tmp_f1 = run_junto(graph_file, seed_file, gold_labels_file, iters, verbose, prune_threshold, algo, mu1, r_mu2, r_mu3, beta, output_path, p, evaluation, write_results, results)
            report.extend(tmp_rep)
            f1.append(tmp_f1)
            confu.extend(tmp_confu)
            for_sorting.extend(tmp_sort)

        sorted_report = sorted(zip(for_sorting, report, confu, f1), key=operator.itemgetter(3), reverse=True)

        with results.open(mode='w') as write_file:
            for i in sorted_report:
                write_file.write(i[1][0]+'\n')
                write_file.write(i[1][1]+'\n')
                write_file.write('off_gold: ' + str(i[1][2].count(1))+' ')
                write_file.write('neg_gold: ' + str(i[1][2].count(0))+'\n')
                write_file.write('off_pred: ' + str(i[1][3].count(1))+' ')
                write_file.write('neg_pred: ' + str(i[1][3].count(0))+'\n')
                write_file.write('Recall: ')
                write_file.write(str(i[2]))
                write_file.write('\n')
                write_file.write('f1: ')
                write_file.write(str(i[3]))
                write_file.write('\n')
                write_file.write('\n')
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

    config_path = root_folder / 'new_config'
    with config_path.open(mode='w') as write_file:
        write_file.write(to_write)



def run_junto(graph_file, seed_file, gold_labels_file, iters, verbose, prune_threshold, algo, mu1, r_mu2, r_mu3, beta, output_path, p, evaluation, write_results, results):
    report = []
    confu = []
    for_sorting = []
    count = 0
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

            y_true = []
            y_pred = []
            
            

            with output_file.open(mode='r') as read_file:
                for line in read_file.read().splitlines():
                    line = line.split('\t')
                    if(len(line[1])>0):  
                        gold_label = line[1].split()[0]
                        has_seed_label = len(line[2]) == 0
                        pred_label = line[3].split()[0]
                        if(not has_seed_label):
                            if(gold_label == 'off'):
                                y_true.append(1)
                            elif(gold_label == 'neg'):
                                y_true.append(0)
                            if(pred_label == 'off'):
                                y_pred.append(1)
                            elif(pred_label == 'neg'):
                                y_pred.append(0)
                            else:
                                y_pred.append(0)
            name = output_path.parts[-2] + output_file.name
            confusion = recall_score(y_true, y_pred, labels=[0, 1], average='macro')
            confu.append(confusion)
            f1 = f1_score(y_true, y_pred, labels=[0, 1], average='macro')
            rep = classification_report(y_true, y_pred, labels=[0,1,2], target_names=['negative', 'offensive', 'dummy'])
            for_sorting.append(float(rep.split()[-2]))
            report.append([name, rep, y_true, y_pred])
            count+=1
            assert(len(y_true) == len(y_pred))
    return report, for_sorting, confu, f1


if __name__ == '__main__': 
    main(sys.argv[1:])





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
            tmp_rep  = run_junto(graph_file, seed_file, gold_labels_file, iters, verbose, prune_threshold, algo, mu1, r_mu2, r_mu3, beta, output_path, p, evaluation, write_results, results)
            report.extend(tmp_rep)


        with results.open(mode='w') as write_file:
            for i in sorted(report, key=operator.itemgetter(-1), reverse=True):
                write_file.write(str(i[0]) + '\n')
                write_file.write('  \t off \t neg\n')
                write_file.write('off \t {} \t {} \n'.format(i[1], i[2]))
                write_file.write('neg \t {} \t {} \n'.format(i[3], i[4]))
                write_file.write('Precision: ' + str(i[5]) + '\n')
                write_file.write('Recall: ' + str(i[6]) + '\n')
                write_file.write('Accuracy: ' + str(i[7]) + '\n')
                write_file.write('F1-Score: ' + str(i[8]) + '\n\n')
                
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
    f1 = []
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
                        if(has_seed_label):
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

            tp = 0
            fp = 0
            fn = 0
            tn = 0 
            for gold, pred in zip(y_true, y_pred):
                
                if(gold == pred):
                    if(gold == 1):
                        tp += 1
                    if(gold == 0):
                        tn += 1
                elif(gold != pred):
                    if(gold == 1):
                        fn += 1
                    if(gold == 0):
                        fp += 1

            if(tp == 0):
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1 = 2*((precision*recall)/(precision+recall))
            accuracy = (tp+tn)/(tp+tn+fp+fn)




            name = output_path.parts[-2] + output_file.name
            '''confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
            confu.append(confusion)
            f1.append(f1_score(y_true, y_pred, labels=[0, 1]))
            rep = classification_report(y_true, y_pred, labels=[0,1], target_names=['negative', 'offensive'])
            for_sorting.append(float(rep.split()[-2]))
            report.append([name, rep, y_true, y_pred])
            count+=1'''

            report.append([name, tp, fp, fn, tn, precision, recall, accuracy, f1])
            assert(len(y_true) == len(y_pred))
    return report


if __name__ == '__main__': 
    main(sys.argv[1:])





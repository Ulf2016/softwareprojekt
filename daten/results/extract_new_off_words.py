from pathlib import Path
import operator

root = Path('extended_mfw_200718_075/')
for_name = root.stem.__str__().strip('/')
subfolders = [x for x in root.iterdir() if(x.is_dir())]

for i in subfolders:
    p = i / 'output'
    for_name2 = i.stem.__str__().strip('/')
    files = [f for f in p.iterdir() if(f.suffix != '.out')]
    for j in files:
        new_words = []
        with j.open() as read_file:
            for line in read_file:
                pred = line.split('\t')[3].split()[0]
                certainty = line.split('\t')[3].split()[1]
                word = line.split('\t')[0]
                if(pred == 'off' and len(line.split('\t')[2])==0):
                    new_words.append((word, certainty))
                    

        o = Path('extracted_words/'+for_name+for_name2+'.out')

        new_words = sorted(new_words, key = operator.itemgetter(1), reverse=True)

        with o.open(mode='w') as write_file:
            for line in new_words:
                write_file.write(line[0]+'\t'+line[1]+'\n')
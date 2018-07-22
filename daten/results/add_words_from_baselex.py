from pathlib import Path

off_words = []
with open('../Baselist/final_list.txt') as read_file:
    for line in read_file:
        if(int(line.split('\t')[2]) == 1):
            off_words.append(line.split('\t')[0])

root = Path('extracted_words')
files = [x for x in root.iterdir() if x.suffix == '.out']


out_path = Path('extracted_words_with_baselist/')
for i in files:
    new_off_words = []
    output = out_path / i.stem
    with i.open() as read_file:
        for line in read_file:
            new_off_words.append(line.split('\t')[0])
    
    with output.open('w') as write_file:
        for j in off_words:
            write_file.write(j + '\n')
        for x in new_off_words:
            write_file.write(x + '\n')
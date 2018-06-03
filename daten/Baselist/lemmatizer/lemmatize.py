from iwnlp.iwnlp_wrapper import IWNLPWrapper

output = 'output/Baselist.txt'
path = "Baselist.txt"
lemmatizer = IWNLPWrapper(lemmatizer_path = 'IWNLP.Lemmatizer_20170501.json')

with open(path, 'r') as read_file:
    data = read_file.read().splitlines()

tokens = []
lemmatized = []
tags = []

for i in data: 
    j = i.split('|')
    tokens.append(j[0])
    tags.append(j[1])


for token, tag in zip(tokens, tags):
    lemma = lemmatizer.lemmatize(token, pos_universal_google=tag)
    lemmatized.append(lemma)

for token, lemma in zip(tokens, lemmatized):
    ''
 #   print(token, lemma)
    
with open(output, 'w') as write_file:
    for i, lemma in enumerate(lemmatized):
        if(lemmatized[i]==None):
            write_file.write(tokens[i]+'|'+tags[i]+"\n")
        else:
            for j in lemmatized[i]:
                write_file.write(j.encode('utf-8')+'|'+tags[i])
            write_file.write('\n')



output = 'output/'
path = 'output/Baselist_without_duplicates_lemmatized.txt'

with open(path, 'r') as read_file:
    data = read_file.read().splitlines()

d = {}

for word in data:
    if(word in d):
        d[word] += 1
    else:
        d[word] = 1

with open(output+'duplicates_v2.txt', 'w') as write_file:
    for key in d:
        if(d[key] != 1):
            write_file.write(key + '\t' + str(d[key]) + '\n')

with open(output+'Baselist_final_v1.txt', 'w') as write_file:
    for key in d:
        write_file.write(key+'\n')
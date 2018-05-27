
import json
import operator

path = "gangster_rap.json"
files = ['gangster_rap.json', 'gangster_rap_2.json', 'gangster_rap_3.json', 'gangster_rap_4.json']
output = 'output/insults_sorted'
path_insults = "../schimpfwortliste"

with open(path_insults, 'r') as insults_file:
    insults_list = insults_file.read().splitlines()

insult_dict = {}

for f in files:
    with open(f, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)

    for artist in data['total']:
        for song in artist['songs']:
            for word in song['lyrics'].split():
                if(word.strip(r'[\w,\.?!()]') in insults_list):
                    insult = word.strip(r'[\w,\.?!()]')
                    if(insult in insult_dict):
                        insult_dict[insult] += 1
                    else:
                        insult_dict[insult] = 1

with open(output, 'w+') as write_file:
    write_file.write(json.dumps(sorted(insult_dict.items(), key=operator.itemgetter(1), reverse=True)))
print(sorted(insult_dict.items(), key=operator.itemgetter(1), reverse=True))


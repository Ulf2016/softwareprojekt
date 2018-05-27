import json
import operator

pattern_path = 'pattern_du_N.txt'
insult_path = 'output/insults_sorted'
output_json = 'output/insults_with_patterns_sorted.json'
output_txt = 'output/insults_with_patterns_sorted.txt'

with open(pattern_path, 'r') as pattern_file:
    pattern_list = pattern_file.read().splitlines()

with open(insult_path, 'r') as insult_file:
    data = json.load(insult_file)

insult_list = []
count = 0
for insult in data:
    if(insult[0] in pattern_list):
        print(insult)
        insult_list.append(insult)
        count += 1

print(len(data))
print(count)

with open(output_json, 'w+') as write_file:
    write_file.write(json.dumps(insult_list, indent=4))

with open(output_txt, 'w+') as write_file:
    for insult in insult_list:
        write_file.write(insult[0] + '\n')

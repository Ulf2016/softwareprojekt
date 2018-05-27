
with open('insults_handsortiert.txt', 'r') as read_file:
    insults_list = read_file.read().splitlines()

with open('insults_handsortiert_tagged.txt', 'r') as read_file:
    insults_list_t = read_file.read().splitlines()

with open('SentiWS_negativ_sorted.txt', 'r') as read_file:
    senti_list = read_file.read().splitlines()

for word in insults_list_t:
    if(word in senti_list):
        print(word)

#with open('insults_handsortiert_tagged.txt', 'w+') as write_file:
#    for insult in insults_list:
#        write_file.write(insult+'|NN\n')
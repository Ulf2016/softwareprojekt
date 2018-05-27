
with open('insults_handsortiert.txt', 'r') as read_file:
    insults_list = read_file.read().splitlines()

with open('insults_handsortiert_tagged.txt', 'w+') as write_file:
    for insult in insults_list:
        write_file.write(insult+'|NN\n')
import operator

tweet_path = '/home/duke/Downloads/tweets-de-2013-2014-2015-2017.cleaned.txt'
#tweet_path = 'daten/tokenized_tweets.txt'
baselist = 'daten/Baselist/final_list.txt'
output = 'daten/Baselist_counted'

off_count = {}
neg_count = {}
with open(baselist) as read_file:
    for line in read_file.readlines():
        line = line.split()
        if(line[2]=='1'):
            off_count[line[0]] = 0
        else:
            neg_count[line[0]] = 0



def count():
    with open(tweet_path) as read_file:
        for line in read_file:
            for word in line.split():
                word = word.lower().strip()
                if(word in off_count):
                    off_count[word] += 1
                elif(word in neg_count):
                    neg_count[word] += 1
    return off_count, neg_count


off_count, neg_count = count()
sorted_off = sorted(off_count.items(), key=operator.itemgetter(1), reverse=True)
sorted_neg = sorted(neg_count.items(), key=operator.itemgetter(1), reverse=True)

with open(output, 'w') as write_file:
    for item in sorted_off:
        
        write_file.write(item[0] + '\t')
        write_file.write(str(item[1])+'\t')
        write_file.write('1\n')
    for item in sorted_neg:
        write_file.write(item[0] + '\t')
        write_file.write(str(item[1])+'\t')
        write_file.write('0\n')
            

def getSeeds(n, list):
    pass
from scipy import spatial

data = {}
with open("embeddings") as read_file:
    for i in read_file:
        temp = i.split()
        data[temp[0]] = list(map(float, temp[1:-1]))

with open("output", 'w') as write_file:
    for i in data:
        for j in data:
            sim = spatial.distance.cosine(data[i], data[j])
            write_file.write(i + '\t' + str(sim) + '\t' + j + '\n')


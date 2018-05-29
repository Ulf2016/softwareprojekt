from sklearn.metrics import cohen_kappa_score
def agreement(path1, path2):
    with open(path1, 'r') as read_file:
        file1 = read_file.read().splitlines()
    with open(path2, 'r') as read_file:
        file2 = read_file.read().splitlines()

    vector1 = []
    vector2 = []

    
    correct = 0
    false = 0
    for i in range(len(file1)):
        vector1.append(file1[i].split()[1])
        vector2.append(file2[i].split()[1])
        if(file1[i].split()[1]==file2[i].split()[1]):
            correct+=1
        else:
            false+=1
            print(file1[i].split(), file2[i].split())
        
    print(correct)
    print(false)

    kappa = cohen_kappa_score(vector1, vector2)
    return kappa
        

path1 = "annotation/anno_test_deutsch_Dueker"
path2 = "annotation/anno_test_deutsch_Steinbach" 

print(agreement(path1, path2))

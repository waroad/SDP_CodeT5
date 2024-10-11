import os
import statistics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_accuracy(gold_file, output_file):
    # Read the gold (true labels) and output (predictions) files
    with open(gold_file, 'r') as gf, open(output_file, 'r') as of:
        gold_labels = [line.strip().split('\t')[1] for line in gf.readlines()]
        output_labels = [line.strip().split('\t')[1] for line in of.readlines()]

    # Check if both files have the same number of lines
    if len(gold_labels) != len(output_labels):
        print("Error: Files have different numbers of lines.")
        return None, None, None, None
    else:
        # Convert 'true'/'false' to binary labels (1/0)
        gold_labels = [1 if label == 'true' else 0 for label in gold_labels]
        output_labels = [1 if label == 'true' else 0 for label in output_labels]

        # Calculate metrics using sklearn
        accuracy = accuracy_score(gold_labels, output_labels)
        precision = precision_score(gold_labels, output_labels)
        recall = recall_score(gold_labels, output_labels)
        f1 = f1_score(gold_labels, output_labels)

        return accuracy*100, precision, recall, f1


folders = ['base1','base2','base3','base4','base5']


dict1={}
dict2={}
dict3={}
dict4={}
for folder in folders:
    files = os.listdir(folder)

    # Filter .gold and .output files
    gold_files = [f for f in files if f.endswith('.gold')]
    output_files = [f for f in files if f.endswith('.output')]
    gold_files.sort()
    output_files.sort()

    for gold_file, output_file in zip(gold_files, output_files):
        if gold_file.replace('.gold', '') == output_file.replace('.output', ''):
            temp=calculate_accuracy(os.path.join(folder, gold_file), os.path.join(folder, output_file))
            if gold_file not in dict1:
                dict1[gold_file]=[]
                dict2[gold_file]=[]
                dict3[gold_file]=[]
                dict4[gold_file]=[]
            dict1[gold_file].append(temp[0])
            dict2[gold_file].append(temp[1])
            dict3[gold_file].append(temp[2])
            dict4[gold_file].append(temp[3])
        else:
            raise ValueError(f"Mismatched filenames: {gold_file} and {output_file}")

for item in dict1:
    if len(dict1[item])>4:
        print(item, f'{sum(dict1[item])/5:.2f}', end=' ')
        print(item, sum(dict2[item])/5,  statistics.stdev(dict2[item]))
        print(item, sum(dict3[item])/5,  statistics.stdev(dict3[item]))
        print(item, sum(dict4[item])/5,  statistics.stdev(dict4[item]))
    else:
        print(item, sum(dict1[item]))
        print(item, sum(dict2[item]))
        print(item, sum(dict3[item]))
        print(item, sum(dict4[item]))
        print("")

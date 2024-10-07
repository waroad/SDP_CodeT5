import os


def calculate_accuracy(gold_file, output_file):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    # Read the gold (true labels) and output (predictions) files
    with open(gold_file, 'r') as gf, open(output_file, 'r') as of:
        gold_labels = gf.readlines()
        output_labels = of.readlines()

    # Check if both files have the same number of lines
    if len(gold_labels) != len(output_labels):
        print("Error: Files have different numbers of lines.")
        return

    # Count correct predictions
    total = len(gold_labels)

    for gold, output in zip(gold_labels, output_labels):
        gold_id, gold_label = gold.strip().split('\t')
        output_id, output_label = output.strip().split('\t')

        # Ensure the IDs match
        if gold_id != output_id:
            print(f"Error: IDs do not match at line {gold_id}.")
            return

        # Compare the true label with the predicted label
        if gold_label == output_label:
            true_positive += 1
        elif gold_label != output_label:  # Incorrect prediction
            if output_label == 'true':  # False positive
                false_positive += 1
            if gold_label == 'true':  # False negative
                false_negative += 1
    # Calculate and print accuracy
    accuracy = true_positive / total
    print(f"{gold_file} Accuracy: {accuracy * 100:.2f}%")
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"{gold_file} F1 Score: {f1:.3f}")


folder='base1'

files = os.listdir(folder)

# Filter .gold and .output files
gold_files = [f for f in files if f.endswith('.gold')]
output_files = [f for f in files if f.endswith('.output')]
gold_files.sort()
output_files.sort()

for gold_file, output_file in zip(gold_files, output_files):
    if gold_file.replace('.gold', '') == output_file.replace('.output', ''):
        calculate_accuracy(os.path.join(folder, gold_file), os.path.join(folder, output_file))
    else:
        raise ValueError(f"Mismatched filenames: {gold_file} and {output_file}")


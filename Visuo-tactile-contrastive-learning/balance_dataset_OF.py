import argparse
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the file containing the list of images and their labels")
    parser.add_argument("--method", type=str, default="median", choices=["median", "max", "min"], help="Method to balance the dataset")
    return parser.parse_args()

def balance_dataset(lines, method="median"):
    """
    Balances the dataset by creating a new dataset with the same number of samples for each class
    The maximum number of samples per class is the median of the number of samples per class in the original dataset
    The new dataset is saved in the same directory as the original dataset + "_balanced.txt"
    """
    class_counts = {}
    for line in lines:
        class_idx = int(line.split(",")[1])
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    if method == "median":
        train_samples = np.median(list(class_counts.values()))
    elif method == "max":
        train_samples = max(list(class_counts.values()))
    elif method == "min":
        train_samples = min(list(class_counts.values()))
    else:
        raise ValueError(f"Invalid method: {method}")
    test_samples = train_samples//4
    print(f"Balancing dataset with {method} method - train: {train_samples} samples per class, test: {test_samples} samples per class")


    class_to_insert = {k: train_samples for k, _ in class_counts.items()}
    class_to_insert_test = {k: test_samples for k, _ in class_counts.items()}

    new_lines = []
    new_lines_test = []

    for line in lines:
        class_idx = int(line.split(",")[1])
        if class_to_insert[class_idx] > 0:
            class_to_insert[class_idx] -= 1
            new_lines.append(line)
        elif class_to_insert_test[class_idx] > 0:
            class_to_insert_test[class_idx] -= 1
            new_lines_test.append(line)

    # write the new train file
    random.shuffle(new_lines)
    with open(args.file.replace(".txt", "_balanced.txt"), "w") as file:
        for line in new_lines:
            file.write(line)

    # write the new test file
    test_file = args.file.replace(".txt", "_balanced.txt")
    test_file = test_file.replace("train", "test")
    random.shuffle(new_lines_test)
    with open(test_file, "w") as file:
        for line in new_lines_test:
            file.write(line)

if __name__ == "__main__":
    args = parse_args()

    path_to_file = args.file
    with open(path_to_file, "r") as file:
        lines = file.readlines()

    balance_dataset(lines, args.method)
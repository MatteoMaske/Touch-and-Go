"""
Creating a test file for Touch-and-Go dataset having just a subfolder of the original dataset
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create dataset file from an existing one')
    parser.add_argument('--dataset', type=str, default="/media/mmlab/Volume/matteomascherin/touch_and_go/dataset/", help='Dataset path')
    parser.add_argument('--size', type=int, default=0, help='Size of the new dataset file')
    parser.add_argument('--file', type=str, default="dataset/test.txt", help='Path to the file to be filtered')
    args = parser.parse_args()

    return args

def create_from_tg(args):
    dataset_file = open(args.file, "r")
    dataset_lines = dataset_file.readlines()
    dataset_file.close()

    new_file_name = args.file.split("/")[-1]
    new_file_name = new_file_name.split(".")[0] + "_new.txt"
    newTestFile = open(f"dataset/{new_file_name}", "w")
    newTestSize = args.size if args.size > 0 else len(dataset_lines)

    missing_frames = 0

    for i, line in enumerate(dataset_lines):
        line = line.strip()
        dataset_file, _ = line.split(",")
        folder, photo = dataset_file.split("/")

        if not os.path.exists(args.dataset + folder):
            print("Missing folder: " + args.dataset + folder)
        elif not os.path.exists(args.dataset + folder + "/video_frame/" + photo):
            missing_frames += 1
        else:
            if i % 5 == 0:
                newTestFile.write(line + "\n")
                newTestSize -= 1
        
        if newTestSize == 0:
            break

    newTestFile.close()
    print(f"Total missing frames: {missing_frames} out of {len(dataset_lines)}")
    print(f"New test file created: {new_file_name} with size {len(dataset_lines) - newTestSize}")

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dataset):
        print("Dataset path does not exist")
        exit()

    create_from_tg(args)
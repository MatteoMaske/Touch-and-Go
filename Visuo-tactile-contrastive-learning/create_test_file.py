"""
Creating a test file for Touch-and-Go dataset having just a subfolder of the original dataset
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create test file')
    parser.add_argument('--dataset', type=str, default="/media/mmlab/Volume/matteomascherin/touch_and_go/dataset/", help='Dataset path')
    parser.add_argument('--size', type=int, default=256, help='Size of the new test file')
    args = parser.parse_args()

    return args

def create_from_tg(args):
    testFile = open("dataset/test.txt", "r")
    testFileLines = testFile.readlines()

    newTestFile = open("dataset/test_new.txt", "w")
    newTestSize = args.size

    folderMissing = set()
    overallFolder = set()

    for line in testFileLines:
        line = line.strip()
        file, label = line.split(",")
        folder, photo = file.split("/")
        overallFolder.add(folder)

        if not os.path.exists(args.dataset + folder):
            folderMissing.add(folder)
        elif not os.path.exists(args.dataset + folder + "/video_frame/" + photo):
            print("Missing video frame: " + args.dataset + folder + "/video_frame/" + photo)
        else:
            newTestFile.write(line + "\n")
            newTestSize -= 1
        
        if newTestSize == 0:
            break


    print("Total missing folders: " + str(len(folderMissing)) + " out of " + str(len(overallFolder)))

def create_from_of(args):
    return True
    

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dataset):
        print("Dataset path does not exist")
        exit()

    if "touch_an_go" in args.dataset:
        create_from_tg(args)
    elif "ObjectFolder" in args.dataset:
        create_from_of(args)
    else:
        print("Dataset not recognized")
        exit()
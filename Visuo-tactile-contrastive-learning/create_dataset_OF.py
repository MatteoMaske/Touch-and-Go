import os
import random

# Ceramic, glass, wood, plastic, iron, polycarbonate, and steel -> OF classes
# label.txt -> TG classes

material_map_of_tg = {
    # Object_folder_material: Touch-and-Go_material_id
    "Ceramic": "6",
    "Glass": "2",
    "Wood": "3",
    "Plastic": "1",
    "Iron": "4",
    "Polycarbonate": "1",
    "Steel": "4",
}

material_map_of_of = {
    # Object_folder_material: iterative_id
    "Ceramic": "0",
    "Glass": "1",
    "Wood": "2",
    "Plastic": "3",
    "Iron": "4",
    "Polycarbonate": "5",
    "Steel": "6",
}

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/media/mmlab/Volume/matteomascherin/touch_and_go/dataset/", help="Dataset to create the file")
    parser.add_argument("--ofmaterialfile", type=str, default="./../../ObjectFolder/objects.csv", help="File containing the material of the objects")
    parser.add_argument("--materialmap", type=str, default="of", choices=["of", "tg"], help="Material map to use")
    parser.add_argument("--split", type=int, default=2, choices=[2, 3], help="Split to create either train, test or train, val, test")
    args = parser.parse_args()

    return args

def get_objects_material(args)->dict:
    """
    Get the material of the objects from the objects.csv file

    Args:
    - args: the arguments of the script
    
    Returns:
    objects_material: dict containing the object_id as key and the material_name as value
    """
    objects_material = {}
    with open(args.ofmaterialfile, "r") as file:
        lines = file.readlines()
        for line in lines:
            info_list = line.strip().split(",")
            object = info_list[0]
            material = info_list[3]
            objects_material[object] = material
    return objects_material

def create_dataset_file(folders, args):
    """
    Create the dataset file for the ObjectFolder dataset following Touch-and-Go structure

    Args:
    - args: the arguments of the script
    """
    filename = "dataset/train_offull.txt"
    if args.materialmap == "of":
        material_map = material_map_of_of
    else:
        material_map = material_map_of_tg
        filename = filename + "_" + args.materialmap + "mat.txt"

    file = open(filename, "w") # write the dataset file    

    for folder in folders:
        material = objects_material[folder]
        material_id = material_map[material]
        for frame in os.listdir(args.dataset_path + folder + "/video_frame"):
            file.write(folder + "/" + frame + f",{material_id}\n")

    file.close()

    shuffle_file(filename)
    if args.split > 1:
        split_file(filename, args.split)

def split_file(file_name, splits):
    """
    Split the file into the given splits
    """
    # train, test
    if splits == 2:
        with open(file_name, "r") as file:
            lines = file.readlines()
            test_size = int(0.2 * len(lines))
            test_lines = random.sample(lines, test_size)
            train_lines = [line for line in lines if line not in test_lines]

        with open(file_name, "w") as file:
            for line in train_lines:
                file.write(line)

        with open(file_name.replace("train", "test"), "w") as file:
            for line in test_lines:
                file.write(line)
    # train, val, test
    elif splits == 3:
        val_ratio, test_ratio = 0.1, 0.1

        with open(file_name, "r") as file:
            lines = file.readlines()
            val_size = int(val_ratio * len(lines))
            test_size = int(test_ratio * len(lines))
            
            val_lines = random.sample(lines, val_size)
            test_lines = random.sample(lines, test_size)
            train_lines = [line for line in lines if line not in val_lines and line not in test_lines]

        with open(file_name, "w") as file:
            for line in train_lines:
                file.write(line)

        with open(file_name.replace("train", "val"), "w") as file:
            for line in val_lines:
                file.write(line)

        with open(file_name.replace("train", "test"), "w") as file:
            for line in test_lines:
                file.write(line)

def shuffle_file(file_name):
    """
    Shuffle the lines of the file
    """
    import random
    with open(file_name, "r") as file:
        lines = file.readlines()
        random.shuffle(lines)
        file = open(file_name, "w")
        for line in lines:
            file.write(line)
        file.close()

if __name__ == "__main__":
    args = get_args()
    folders = os.listdir(args.dataset_path)
    of_folders = [f for f in folders if len(f) <= 3] # ObjectFolder folders have a length of 3

    objects_material = get_objects_material(args)
    print(objects_material, len(objects_material))  

    create_dataset_file(of_folders, args)

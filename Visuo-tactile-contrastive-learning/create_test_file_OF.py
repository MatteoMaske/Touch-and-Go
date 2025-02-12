import os

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
    parser.add_argument("--tgdataset", type=str, default="/media/mmlab/Volume/matteomascherin/touch_and_go/dataset/", help="Dataset to create the file")
    parser.add_argument("--ofmaterialfile", type=str, default="./../../ObjectFolder/objects.csv", help="File containing the material of the objects")
    parser.add_argument("--filename", type=str, default="dataset/pretrain_OF", help="Name of the file to write the dataset")
    parser.add_argument("--materialmap", type=str, default="of", choices=["of", "tg"], help="Material map to use")
    args = parser.parse_args()

    args.filename = args.filename + "_" + args.materialmap + "mat.txt" if args.materialmap == "of" else args.filename + ".txt"
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

    file = open(args.filename, "w") # write the dataset file
    if args.materialmap == "of":
        material_map = material_map_of_of
    else:
        material_map = material_map_of_tg

    for folder in folders:
        material = objects_material[folder]
        material_id = material_map[material]
        for frame in os.listdir(args.tgdataset + folder + "/video_frame"):
            file.write(folder + "/" + frame + f",{material_id}\n")

    file.close()

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
    folders = os.listdir(args.tgdataset)
    of_folders = [f for f in folders if len(f) <= 3] # ObjectFolder folders have a length of 3

    objects_material = get_objects_material(args)
    print(objects_material, len(objects_material))  

    create_dataset_file(of_folders, args)
    shuffle_file(args.filename)

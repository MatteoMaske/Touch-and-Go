"""
Creating a test file for the ObjectFolder dataset, following the same structure of the Touch-and-Go dataset
"""

import os

tgDataset = "/media/mmlab/Volume/matteomascherin/touch_and_go/dataset/"
ofMaterialFile = "./../../ObjectFolder/objects.csv"
file_name = "dataset/pretrain_OF.txt"

# Ceramic, glass, wood, plastic, iron, polycarbonate, and steel -> OF classes
# label.txt -> TG classes

material_map = {
    # Object_folder_material: Touch-and-Go_material_id
    "Ceramic": "6",
    "Glass": "2",
    "Wood": "3",
    "Plastic": "1",
    "Iron": "4",
    "Polycarbonate": "1",
    "Steel": "4",
}

def get_obejcts_material():
    #ObjectName, MaterialName
    objects_material = {}
    with open(ofMaterialFile, "r") as file:
        lines = file.readlines()
        for line in lines:
            info_list = line.strip().split(",")
            object = info_list[0]
            material = info_list[3]
            objects_material[object] = material
    return objects_material

folders = os.listdir(tgDataset)
of_folders = [f for f in folders if len(f) <= 3]

objects_material = get_obejcts_material()
print(objects_material)

file = open(file_name, "w")

for folder in of_folders:
    material = objects_material[folder]
    material_id = material_map[material]
    for frame in os.listdir(tgDataset + folder + "/video_frame"):
        file.write(folder + "/" + frame + f",{material_id}\n")

file.close()

#read the same file and shuffle it
with open(file_name, "r") as file:
    lines = file.readlines()
    import random
    random.shuffle(lines)
    file = open(file_name, "w")
    for line in lines:
        file.write(line)
    file.close()

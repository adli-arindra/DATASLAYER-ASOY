import os
import shutil

path = "train/"

output = "output/"

count0 = 0
count1 = 0

for subjects in os.listdir(path):
    for category in os.listdir(os.path.join(path, subjects)):
        for sub_category in os.listdir(os.path.join(path, subjects, category)):
            idx = 0
            for png in os.listdir(os.path.join(path, subjects, category, sub_category)):
                current_path = os.path.join(path, subjects, category, sub_category, png)

                if (category == "fall"): 
                    label = 1
                    count1 += 1
                    filename = str(label) + "_" + sub_category.split("_")[1] + "_" + str(count1)
                else: 
                    label = 0
                    count0 += 0
                    filename = str(label) + "_" + sub_category.split("_")[1] + "_" + str(count0)

                shutil.copyfile(current_path, os.path.join(output, filename) + ".png")
import os
from remove_backgroud import mask_foreground

path = "renamed/"
output = "processed/"

for png in os.listdir(os.path.join(path)):
    current_path = os.path.join(path, png)
    mask_foreground(current_path, "median_image.png", os.path.join(output, png))
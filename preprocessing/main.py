from PIL import Image
import numpy as np
import crop
import masking
import cv2
import os

def process_image(input_path: str, output_path: str):
    opencv_image = cv2.imread(input_path)
    cropped_image = crop.crop(opencv_image, 
                              detection_height=300, 
                              detection_width=250, 
                              slide_up=50,
                              slide_right=50)
    pil_image = Image.fromarray(cropped_image)
    masked_image = masking.apply_mask(pil_image)
    masked_scaled = masked_image * 255
    cv2.imwrite(output_path, masked_scaled)

def process_renamed():
    input_path = "renamed/"
    output_path = "processed/"
    for file in os.listdir(input_path):
        process_image(os.path.join(input_path, file), 
                      os.path.join(output_path, file))
        print(f"Successfully masked {file}")

    print("All done")


def process_test():
    input_path = "test/"
    output_path = "processed_test/"
    for file in os.listdir(input_path):
        process_image(os.path.join(input_path, file),
                      os.path.join(output_path, file))
        print(f"Successfully masked {file}")
    print("Done")


if __name__ == "__main__":
    process_test()
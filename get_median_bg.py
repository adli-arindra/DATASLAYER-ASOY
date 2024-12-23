import os
import cv2
import numpy as np

def get_unique_resolutions(folder_path):
    resolutions = set()  # Use a set to store unique resolutions
    
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a PNG image
        if file_name.lower().endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            # Read the image
            image = cv2.imread(file_path)
            pixel = []

    
    return list(resolutions)

def calculate_pixel_medians_optimized(folder_path):
    image_count = 0
    pixel_values = None

    # Iterate through the images in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
            gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

            if gray_image is not None:
                # Initialize the container on the first valid blurred_image
                if pixel_values is None:
                    height, width, channels = gray_image.shape
                    pixel_values = np.zeros((height, width, channels), dtype=np.uint64)

                # Sum pixel values across blurred_images
                pixel_values += image
                image_count += 1
            else:
                print(f"Warning: Could not read {file_name}")

    if image_count == 0:
        print("No valid images found in the folder.")
        return None

    # Calculate the average (sum divided by the number of images)
    median_values = (pixel_values / image_count).astype(np.uint8)

    return median_values

if __name__ == "__main__":
    folder_path = "renamed/"  # Replace with your folder path
    median_image = calculate_pixel_medians_optimized(folder_path)

    if median_image is not None:
        print("Median RGB values calculated for each pixel.")
        print(f"Median image shape: {median_image.shape}")
        # Optionally save the result as an image for visualization
        cv2.imwrite("median_image.png", median_image)

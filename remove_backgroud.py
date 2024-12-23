import cv2
import numpy as np
import os

def remove_background(image_path, background_path, output_path="output.png", threshold=20):
    """
    Removes the background from the foreground image by comparing it with a background image.
    The pixels similar to the background are removed (made transparent).

    Args:
        image_path (str): Path to the foreground image.
        background_path (str): Path to the background image.
        output_path (str): Path to save the output image (with transparent background).
        threshold (int): Difference threshold to consider pixels similar to the background.
    """
    # Load the images
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    # Ensure the images have the same dimensions
    if image.shape != background.shape:
        raise ValueError("Foreground and background images must have the same dimensions.")

    # Convert the background to grayscale and blur it
    blurred_background = cv2.GaussianBlur(background, (15, 15), 0)
    gray_background = cv2.cvtColor(blurred_background, cv2.COLOR_BGR2GRAY)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the grayscale image and the background
    diff = cv2.absdiff(gray_image, gray_background)

    # Create a mask where the difference is greater than the threshold
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Convert the image to have an alpha channel (transparency)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # Adding an alpha channel (4th channel)
    
    # Set the alpha channel to 0 (transparent) where the mask is 0 (background)
    result[mask == 0] = [0, 0, 0, 0]  # Set transparent background for matching pixels

    # Save the result as a PNG image with transparency
    cv2.imwrite(output_path, result)
    print(f"Image with background removed saved to {output_path}")
    
    return result

# Example usage
# if __name__ == "__main__":
#     image_path = "renamed/0_jumping_0.png"       # Path to the foreground image
#     background_path = "median_image.png" # Path to the background image
#     output_path = "output.png"   # Path to save the result

#     result = remove_background(image_path, background_path)


if __name__ == "__main__":
    folder_path = "renamed/"
    for file in os.listdir(folder_path):
        if file.lower().endswith('.png'):
            image_path = os.path.join(folder_path, file)
            output_path = os.path.join("processed", file)
            result = remove_background(image_path, "median_image.png", output_path)
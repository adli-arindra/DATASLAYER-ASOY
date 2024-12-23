import cv2
import numpy as np

def mask_foreground(image_path, background_path, output_path = "", threshold=30):
    """
    Masks the foreground image by comparing it with a background image.
    Pixels similar to the background are set to white.

    Args:
        image_path (str): Path to the foreground image.
        background_path (str): Path to the background image.
        output_path (str): Path to save the masked image.
        threshold (int): Difference threshold to consider pixels similar.
    """
    # Load the images
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)
    blurred_background = cv2.GaussianBlur(background, (15, 15), 0)

    # Ensure the images have the same dimensions
    if image.shape != background.shape:
        raise ValueError("Foreground and background images must have the same dimensions.")

    # Calculate the absolute difference between the images
    diff = cv2.absdiff(image, background)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Create a mask where the difference is less than the threshold
    mask = diff_gray < threshold

    # Create the output image, setting masked pixels to white
    result = np.copy(image)
    result[mask] = [207, 15, 251]  # Set pixels to white

    # Save the result
    cv2.imwrite(output_path, result)

    # cv2.imshow("yo", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"Masked image saved to {output_path}")
    return result

# Example usage
if __name__ == "__main__":
    image_path = "renamed/0_jumping_0.png"       # Path to the foreground image
    background_path = "median_image.png" # Path to the background image
    output_path = "output.png"   # Path to save the result

    result = mask_foreground(image_path, background_path, output_path)

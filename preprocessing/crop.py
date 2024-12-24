import cv2
import numpy as np

# Load the median background
median_background = cv2.imread("median_image.png", cv2.IMREAD_GRAYSCALE)

if median_background is None:
    raise FileNotFoundError("Median background image not found or failed to load.")

def crop_max_difference_bottom(
    image_path, 
    median_background, 
    detection_height=300, 
    detection_width=50, 
    crop_size=(500, 500)
):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Input image '{image_path}' not found or failed to load.")

    # Ensure the images have the same resolution
    if image.shape != median_background.shape:
        image = cv2.resize(image, (median_background.shape[1], median_background.shape[0]))

    # Create a mask for white pixels (255) in the image
    white_pixel_mask = (image == 255)

    # Compute the absolute difference
    diff = cv2.absdiff(image, median_background)

    # Ignore white pixels by setting their differences to 0
    diff[white_pixel_mask] = 0

    # Parameters
    crop_height, crop_width = crop_size
    image_height, image_width = diff.shape

    if image_height < crop_height or image_width < crop_width:
        raise ValueError("Image dimensions are smaller than the crop size.")

    # Restrict the detection area vertically to the bottom `detection_height` pixels
    detection_start_y = max(0, image_height - detection_height)
    diff[:detection_start_y, :] = 0  # Ignore regions above the detection height

    # Restrict the detection area horizontally to avoid `detection_width` padding
    diff[:, :detection_width] = 0  # Ignore the leftmost padding
    diff[:, image_width - detection_width:] = 0  # Ignore the rightmost padding

    # Initialize variables
    max_diff_sum = -1
    best_top_left = (detection_width, image_height - crop_height)

    # Sliding window (restricted area)
    for y in range(detection_start_y, image_height - crop_height + 1):
        for x in range(detection_width, image_width - detection_width - crop_width + 1):
            # Extract the region of interest (ROI)
            roi = diff[y:y + crop_height, x:x + crop_width]
            diff_sum = np.sum(roi)

            # Update max_diff_sum and best_top_left
            if diff_sum > max_diff_sum:
                max_diff_sum = diff_sum
                best_top_left = (x, y)

    # Crop the region from the original image
    x, y = best_top_left
    cropped_image = image[y:y + crop_height, x:x + crop_width]

    return cropped_image, best_top_left

if __name__ == "__main__":
    image_path = "coba/1_standing_1107.png"
    cropped_image, top_left = crop_max_difference_bottom(
        image_path, 
        median_background, 
        detection_height=200, 
        detection_width=450
    )

    # Display the cropped image
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("cropped.png", cropped_image)

    # Save the cropped image if needed
    cv2.imwrite("cropped_image.jpg", cropped_image)

    print(f"Top-left corner of the cropped region: {top_left}")

import cv2
import numpy as np

# Load the median background
median_background = cv2.imread("median_image.png", cv2.IMREAD_GRAYSCALE)

if median_background is None:
    raise FileNotFoundError("Median background image not found or failed to load.")

def crop_image(
    image_path, 
    median_background, 
    detection_height=300, 
    detection_width=50, 
    crop_size=(500, 500)
):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Input image '{image_path}' not found or failed to load.")
    if image.shape != median_background.shape:
        image = cv2.resize(image, (median_background.shape[1], median_background.shape[0]))
    white_pixel_mask = (image == 255)
    diff = cv2.absdiff(image, median_background)
    diff[white_pixel_mask] = 0
    crop_height, crop_width = crop_size
    image_height, image_width = diff.shape
    if image_height < crop_height or image_width < crop_width:
        raise ValueError("Image dimensions are smaller than the crop size.")

    detection_start_y = max(0, image_height - detection_height)
    diff[:detection_start_y, :] = 0  
    diff[:, :detection_width] = 0 
    diff[:, image_width - detection_width:] = 0

    max_diff_sum = -1
    best_top_left = (detection_width, image_height - crop_height)

    for y in range(detection_start_y, image_height - crop_height + 1):
        for x in range(detection_width, image_width - detection_width - crop_width + 1):
            roi = diff[y:y + crop_height, x:x + crop_width]
            diff_sum = np.sum(roi)

            if diff_sum > max_diff_sum:
                max_diff_sum = diff_sum
                best_top_left = (x, y)

    x, y = best_top_left
    cropped_image = image[y:y + crop_height, x:x + crop_width]
    return cropped_image, best_top_left

if __name__ == "__main__":
    image_path = "coba/1_standing_1107.png"
    cropped_image, top_left = crop_image(
        image_path, 
        median_background, 
        detection_height=200, 
        detection_width=450
    )

    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("cropped.png", cropped_image)

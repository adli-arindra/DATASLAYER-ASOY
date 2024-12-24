import cv2
import numpy as np

# Load the median background
median_background = cv2.imread("preprocessing/median.png", cv2.IMREAD_GRAYSCALE)

if median_background is None:
    raise FileNotFoundError("Median background image not found or failed to load.")

def crop(
    image,  
    detection_height=200,
    detection_width=450,
    slide_up=0,
    slide_right=0,
    crop_size=(600, 600)
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray_image is None:
        raise FileNotFoundError(f"Input image is not found or failed to load.")
    if gray_image.shape != median_background.shape:
        gray_image = cv2.resize(gray_image, (median_background.shape[1], median_background.shape[0]))
    white_pixel_mask = (gray_image == 255)
    diff = cv2.absdiff(gray_image, median_background)
    diff[white_pixel_mask] = 0
    crop_height, crop_width = crop_size
    gray_image_height, gray_image_width = diff.shape
    if gray_image_height < crop_height or gray_image_width < crop_width:
        raise ValueError("gray_image dimensions are smaller than the crop size.")

    detection_start_y = max(0, (gray_image_height - detection_height)**3)
    diff[:detection_start_y, :] = 0  
    diff[:, :detection_width] = 0 
    diff[:, gray_image_width - detection_width:] = 0

    max_diff_sum = -1
    best_top_left = (detection_width, gray_image_height - crop_height)

    for y in range(detection_start_y, gray_image_height - crop_height + 1):
        for x in range(detection_width, gray_image_width - detection_width - crop_width + 1):
            roi = diff[y:y + crop_height, x:x + crop_width]
            diff_sum = np.sum(roi)

            if diff_sum > max_diff_sum:
                max_diff_sum = diff_sum
                best_top_left = (x, y)

    x, y = best_top_left
    cropped_image = image[y - slide_up:y - slide_up + crop_height, 
                          x + slide_right:x + slide_right + crop_width]
    return cropped_image

if __name__ == "__main__":
    image_path = "coba/1_standing_1107.png"
    cropped_image = crop(image_path)

    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("cropped.png", cropped_image)

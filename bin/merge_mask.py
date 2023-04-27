import cv2
import numpy as np

# Load the original image and the predicted mask
img = cv2.imread("old-0100-B.png")
print(img.shape)
mask = cv2.imread("old-0100-B_seg.png", 0)

def merge_mask(img, mask, show_img=False, result_file="result.png"):
# Create a new image with red color
    red_color = np.zeros_like(img)
    red_color[:] = (0, 0, 255)

    # Replace white pixels in the mask with red pixels
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    red_mask = cv2.inRange(mask_rgb, (255, 255, 255), (255, 255, 255))
    result = cv2.bitwise_and(red_color, red_color, mask=red_mask)

    # Replace black pixels in the mask with black pixels in the original image
    black_mask = cv2.inRange(mask_rgb, (0, 0, 0), (0, 0, 0))
    result += cv2.bitwise_and(img, img, mask=black_mask)
    cv2.imwrite(result_file, result)
    if show_img:
        # Display the result
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

merge_mask(img, mask)
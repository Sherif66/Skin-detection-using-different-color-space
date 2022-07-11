import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Name and path of the images to load:
image_names = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
path = ''


# Load all test images
def load_all_test_images():

    skin_images = []
    for index_ima, name_image in enumerate(image_names):
        image_path = os.path.join(path, name_image)
        skin_images.append(cv2.imread(image_path))
    return skin_images

# Show all the images of the array
def show_images(array_img, title, pos):

    for index_image, image in enumerate(array_img):
        show_with_matplotlib(image, title + "_" + str(index_image + 1), pos + index_image)


# Shows the image according to it's position
def show_with_matplotlib(color_img, title, pos):

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(5, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Lower and upper boundaries for the HSV skin segmentation method:
lower_hsv = np.array([0, 48, 80], dtype="uint8")
upper_hsv = np.array([20, 255, 255], dtype="uint8")


# Skin detector based on the HSV color spaces
def skin_detector_hsv(bgr_image):


    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    skin_region = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return skin_region

# Lower and upper boundaries for the HSV skin segmentation method:
lower_hsv_2 = np.array([0, 50, 0], dtype="uint8")
upper_hsv_2 = np.array([120, 150, 255], dtype="uint8")


# Skin detector based on the HSV color space
def skin_detector_hsv_2(bgr_image):


    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)


    skin_region = cv2.inRange(hsv_image, lower_hsv_2, upper_hsv_2)
    return skin_region


lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
upper_ycrcb = np.array([255, 173, 127], dtype="uint8")


# Skin detector based on the YCrCb color space
def skin_detector_ycrcb(bgr_image):

    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)

    skin_region = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
    return skin_region


def bgr_skin(b, g, r):

    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
    abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2

# Skin detector based on the BGR color space
def skin_detector_bgr(bgr_image):

    h = bgr_image.shape[0]
    w = bgr_image.shape[1]


    res = np.zeros((h, w, 1), dtype="uint8")

    for y in range(0, h):
        for x in range(0, w):
            (b, g, r) = bgr_image[y, x]
            if bgr_skin(b, g, r):
                res[y, x] = 255

    return res

# Implemented skin detectors to be used:
skin_detectors = {
'ycrcb': skin_detector_ycrcb,
'hsv': skin_detector_hsv,
'hsv_2': skin_detector_hsv_2,
'bgr': skin_detector_bgr}



def apply_skin_detector(array_img, skin_detector):


    skin_detector_result = []
    for index_image, image in enumerate(array_img):
        detected_skin = skin_detectors[skin_detector](image)
        bgr = cv2.cvtColor(detected_skin, cv2.COLOR_GRAY2BGR)
        skin_detector_result.append(bgr)
    return skin_detector_result

plt.figure(figsize=(15, 8))
plt.suptitle("Skin segmentation using different color spaces", fontsize=14, fontweight='bold')

for i, (k, v) in enumerate(skin_detectors.items()):
    print("index: '{}', key: '{}', value: '{}'".format(i, k, v))


test_images = load_all_test_images()

show_images(test_images, "test img", 1)

show_images(apply_skin_detector(test_images, 'ycrcb'), "ycrcb", 7)
show_images(apply_skin_detector(test_images, 'hsv'), "hsv", 13)
show_images(apply_skin_detector(test_images, 'hsv_2'), "hsv_2", 19)
show_images(apply_skin_detector(test_images, 'bgr'), "bgr", 25)

# Show the created image:
plt.show()

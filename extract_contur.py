import sys
import os
from copy import deepcopy
from functools import wraps
from io import BytesIO

import cv2
import numpy as np

sys.path.append(os.path.abspath("../plots_storage"))
from plots import display_plot

PLOTS_TO_DISPLAY = {
    "import_image": False,
    "blur_image": False,
    "get_mask": False,
    "replace_extra": False,
    "invert_mask": True,
}


def display_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        file_name = kwargs.get("file_name")
        try:
            result = func(*args, **kwargs)
        except TypeError:
            kwargs.pop("file_name")
            result = func(*args, **kwargs)

        if not PLOTS_TO_DISPLAY.get(func.__name__, False):
            return result

        display_result = deepcopy(result)
        if len(display_result.shape) == 2:
            display_result = cv2.cvtColor(display_result, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(f"./plots/{file_name}_{func.__name__}.jpg", display_result)

        return result

    return wrapper


@display_decorator
def import_image(file_name):
    img = cv2.imread(f"./input_data/{file_name}.jpg")
    return img


@display_decorator
def blur_image(img, kernel):
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return img


@display_decorator
def get_mask(image, white=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if not white:
        # TODO  Rewrite for any color
        lower_color = (
            np.min(
                np.vstack((hsv[0:20, 0:20], hsv[-20:, -20:]), dtype=np.int32),
                axis=(0, 1),
            )[0]
            - 15
        )
        upper_color = (
            np.max(
                np.vstack((hsv[0:20, 0:20], hsv[-20:, -20:]), dtype=np.int32),
                axis=(0, 1),
            )[0]
            + 15
        )
        lower_bound = np.array([np.clip(lower_color, 0, 179).astype(np.uint8), 50, 50])
        upper_bound = np.array(
            [np.clip(upper_color, 0, 179).astype(np.uint8), 255, 255]
        )

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
    else:
        lower_bound = np.array([0, 0, 235])
        # lower_bound = np.array([0, 0, 255])
        upper_bound = np.array([179, 50, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return mask


def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_sorted = sorted(
        contours, key=lambda c: cv2.arcLength(c, True), reverse=True
    )

    return contours_sorted


@display_decorator
def replace_extra(contours, mask):
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, 0, 255, thickness=cv2.FILLED)
    output = np.where(contour_mask == 0, 255, mask)

    return output


@display_decorator
def invert_mask(mask):
    return 255 - mask


def display_contours(img, contours, file_name):
    for ind, c in enumerate(contours):
        if c.shape[0] > 1000:
            contur_reverse = np.array([[x, img.shape[0] - y] for [x, y] in c[:, 0, :]]).T

            data = [contur_reverse, "lines", "Сontours", "#2325b8", {"line": {"width": 1}}, True]

            display_plot(
                [data],
                filename = f"{file_name}_{ind + 1}_contours",
                save_path = "/Users/dmyrto_koltsov/PycharmProjects/process_conturs/plots/",
                img = True,
                title = "contours",
                equal = True,
                background_image = f"/Users/dmyrto_koltsov/PycharmProjects/process_conturs/input_data/{file_name}.jpg",
            )


for number in range(1, 6):
    file_name = f"all_good_{number}"
    # file_name = f"inside_out_{number}"

    input_img = import_image(file_name=file_name)
    blur_img = blur_image(img=input_img, kernel=3, file_name=file_name)

    mask = get_mask(blur_img, white=True, file_name=file_name)

    contours = extract_contours(mask)

    cutted_img = replace_extra(contours, mask, file_name=file_name)

    inverted_mask = invert_mask(cutted_img, file_name=file_name)

    contours = extract_contours(inverted_mask)
    display_contours(cutted_img, contours, file_name)


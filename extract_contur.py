import argparse
import os.path
from copy import deepcopy
from functools import wraps

import cv2
import numpy as np
from plots_lib import display_plot

from config import *


def display_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if PLOTS_TO_DISPLAY.get(func.__name__, False):
            save_path = os.path.join(MATERIALS_PATH, f"processed_contour/{file_name.rsplit("_", 1)[0]}/plots/{side}")
            os.makedirs(save_path, exist_ok=True)

            display_result = deepcopy(result)
            if len(display_result.shape) == 2:
                display_result = cv2.cvtColor(display_result, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(os.path.join(save_path, f"{func.__name__}.jpg"), display_result)

        return result

    return wrapper


@display_decorator
def import_image(file_name):
    img = cv2.imread(
        os.path.join(
            MATERIALS_PATH,
            f"input_data/{file_name.rsplit("_", 1)[0]}",
            f"{file_name}.jpg"
        )
    )
    return img


@display_decorator
def blur_image(img, kernel):
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return img


@display_decorator
def get_mask(image, white=False, threshold=255):
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
        lower_bound = np.array([0, 0, threshold])
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


def display_contours(img, contours, file_name, side):
    save_path = os.path.join(MATERIALS_PATH, f"processed_contour/{file_name.rsplit("_", 1)[0]}/plots/{side}")
    bg_path = os.path.join(MATERIALS_PATH, f"input_data/{file_name.rsplit("_", 1)[0]}", f"{file_name}.jpg")

    for ind, c in enumerate(contours):
        if c.shape[0] > 1000:
            transformed_contur = process_contur(c, img).T

            data = [
                transformed_contur,
                "lines",
                "Сontours",
                "#2325b8",
                {"line": {"width": 1}},
                True,
            ]

            display_plot(
                [data],
                filename=f"contour_{ind + 1}",
                save_path=save_path,
                s_json=True,
                title="contours",
                equal=True,
                background_image=bg_path,
            )


def save_contours(contours, number, img):
    save_path = os.path.join(MATERIALS_PATH, f"processed_contour/{file_name.rsplit("_", 1)[0]}/res_data")
    os.makedirs(save_path, exist_ok=True)

    for ind, c in enumerate(contours):
        if c.shape[0] > 1000:
            transformed_contur = process_contur(c, img)
            np.savetxt(
                os.path.join(save_path, f"contour_{number}.txt"),
                transformed_contur,
                fmt="%d",
            )
            number += 1

    return number


def delete_same_points(contur):
    _, idx = np.unique(contur, axis=0, return_index=True)
    unique_points = contur[np.sort(idx)]

    return unique_points


def process_contur(contour, img):
    contur_reverse = np.array([[x, img.shape[0] - y] for [x, y] in contour[:, 0, :]])
    unique_contur = delete_same_points(contur_reverse)

    return unique_contur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--template", help="File name template", required=True, type=str
    )
    parser.add_argument(
        "-w",
        "--is_white",
        help="Background colour white",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-wt", "--white_threshold", help="Background colour white threshold", type=int
    )
    parser.add_argument(
        "-i",
        "--import_image",
        help="Display import image",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--blur_image",
        help="Display blur image",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m", "--mask", help="Display mask", action="store_true", default=False
    )
    parser.add_argument(
        "-e",
        "--replace_extra",
        help="Display replace extra",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-im",
        "--invert_mask",
        help="Display invert mask",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-dc",
        "--display_contour",
        help="Display res contour",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sc",
        "--save_contour",
        help="Save res contour",
        action="store_true",
        default=False,
    )
    parser.add_argument("-n", "--numbers", help="Display colour", type=str, nargs="+")

    argv, unknown_argv = parser.parse_known_args()

    if argv.is_white and argv.white_threshold is None:
        parser.error("--white_threshold/-wt is required when --is_white/-w is set")

    PLOTS_TO_DISPLAY = {
        "import_image": argv.import_image,
        "blur_image": argv.blur_image,
        "get_mask": argv.mask,
        "replace_extra": argv.replace_extra,
        "invert_mask": argv.invert_mask,
    }

    if "all" in argv.numbers:
        numbers = range(1, int(argv.numbers[1]) + 1)
    else:
        numbers = argv.numbers

    general_number = 1

    for number in numbers:
        file_name = argv.template.format(number)
        side = f"side_{number}"

        input_img = import_image(file_name=file_name)
        blur_img = blur_image(img=input_img, kernel=3)

        mask = get_mask(
            blur_img,
            white=argv.is_white,
            threshold=argv.white_threshold,
        )

        contours = extract_contours(mask)

        cutted_img = replace_extra(contours, mask)

        inverted_mask = invert_mask(cutted_img)

        contours = extract_contours(inverted_mask)

        if argv.save_contour:
            general_number = save_contours(contours, general_number, input_img)

        if argv.display_contour:
            display_contours(cutted_img, contours, file_name, side)

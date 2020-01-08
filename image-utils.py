import os

from PIL import Image
from shutil import copyfile
import numpy as np


def move_image(
    image_name, raw_dir, wide_dir="wide", narrow_dir="narrow",
):
    for dir_ in [wide_dir, narrow_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    src_path = f"{raw_dir}/{image_name}"
    im = Image.open(src_path)

    width, height = im.size

    if width / height >= 1.33:
        copyfile(src_path, f"{wide_dir}/{image_name}")
    else:
        copyfile(src_path, f"{narrow_dir}/{image_name}")


def split_horizontal(src_path, dst_path, filename):
    # https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python/47581978#47581978
    im = np.array(Image.open(f"{src_path}/{filename}"))
    M = im.shape[0]
    N = im.shape[1] // 2

    halves = [
        im[x : x + M, y : y + N]
        for x in range(0, im.shape[0], M)
        for y in range(0, im.shape[1], N)
    ]

    for i, half in enumerate(halves[:2]):
        name = filename.split(".")[0]
        Image.fromarray(half, "RGB").save(f"{dst_path}/{name}_{i}.png")

    del im


def make_square(src_path, dst_path, filename):
    im = Image.open(f"{src_path}/{filename}")
    width, height = im.size  # Get dimensions
    new_len = min(width, height)

    left = (width - new_len) / 2
    top = (height - new_len) / 2
    right = (width + new_len) / 2
    bottom = (height + new_len) / 2

    # Crop the center of the image
    im.crop((left, top, right, bottom)).save(f"{dst_path}/{filename}")

    del im


def resize(src_path, dst_path, filename):
    im = Image.open(f"{src_path}/{filename}")
    im.resize((128, 128)).save(f"{dst_path}/{filename}")

    del im


def to_jpg(src_path, dst_path, filename):
    # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil

    png = Image.open(f"{src_path}/{filename}.png").convert("RGBA")
    background = Image.new("RGBA", png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png).convert("RGB")
    alpha_composite.save(f"{dst_path}/{filename}.jpg", "JPEG", quality=90)

    del alpha_composite


if __name__ == "__main__":
    raw_dir = "data"
    dst_dir = "dataset"

    for image_name in os.listdir(raw_dir):
        to_jpg(raw_dir, dst_dir, image_name.split(".")[0])

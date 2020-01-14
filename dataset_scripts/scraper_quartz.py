import json
import os
import shutil
import time

import requests


def sync_image(session, lego_id, sync_directory):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0"
    }

    img_response = session.get(
        url=f"https://data.qz.com/2018/lego-faces/assets/mf-img/{lego_id}.png",
        headers=headers,
        stream=True,
    )

    with open(f"{sync_directory}/{lego_id}.png", "wb+") as out_file:
        shutil.copyfileobj(img_response.raw, out_file)

    del img_response


def sync_all_images(session, sync_directory):
    if not os.path.exists(sync_directory):
        os.mkdir(sync_directory)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0"
    }

    response = session.get(
        url="https://data.qz.com/2019/lego-emotion-ranker/data/lego_coords.json",
        headers=headers,
    )

    for lego_id in {c["legoID"] for c in json.loads(response.text)}:
        sync_image(session, lego_id, sync_directory)

        # crude rate limit
        time.sleep(0.05)


if __name__ == "__main__":
    # get parts that have not been synced yet
    sync_directory = "images-quartz"
    session = requests.Session()

    sync_all_images(session, sync_directory)

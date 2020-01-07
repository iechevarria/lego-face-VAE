import os
import re
import shutil
import time

import pandas as pd
import requests

from secrets import username, password


def sync_part_image(session, part_id, sync_directory, save_image=True):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0"
    }

    response = session.get(
        url=f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={part_id}",
        headers=headers,
    )

    img_url = re.search(
        rf"\bimg\.bricklink\.com\/ItemImage\/[A-Z]+\/[0-9]+\/{part_id}\.png\b", response.text
    ).group(0)

    img_response = session.get(url=f"https://{img_url}", headers=headers, stream=True)
    with open(f"images/{part_id}.png", "wb") as out_file:
        shutil.copyfileobj(img_response.raw, out_file)

    del img_response


def get_session():
    session = requests.Session()

    data = {
        "userid": username,
        "password": password,
        "override": "false",
        "keepme_loggedin": "false",
        "mid": "16b44759b6f00000-293bd8adde9ee65b",
        "pageid": "MAIN",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0"
    }

    res = session.post(
        "https://www.bricklink.com/ajax/renovate/loginandout.ajax",
        data=data,
        headers=headers,
    )

    print(f"logged in status code {res.status_code}")

    # load homepage once to avoid initial redirect
    session.get("https://www.bricklink.com/")

    return session


if __name__ == "__main__":
    # get relevant parts to sync
    df = pd.read_csv("parts-to-sync.csv")

    # # get sets that have not been synced yet
    sync_directory = "images"
    synced_images = {i.split(".")[0] for i in os.listdir(sync_directory)}
    images_to_sync = {i for i in df["Number"] if i not in synced_images}

    # init session
    session = get_session()

    # save tsvs of sets
    num_synced = 0
    for part_id in images_to_sync:

        try:
            sync_part_image(session, part_id, sync_directory)

            num_synced += 1
            if num_synced % 50 == 0:
                print(f"synced {num_synced} of {len(images_to_sync)}")
        
        except AttributeError:
            print(f'Sync failed for part {part_id}')

        # crude rate limit
        time.sleep(0.05)

import os
import re
import shutil
import time

import requests

from secrets import USERNAME, PASSWORD


def sync_part_image(session, part_id, sync_directory, save_image=True):
    if not os.path.exists(sync_directory):
        os.mkdir(sync_directory)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0"
    }

    response = session.get(
        url=f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={part_id}",
        headers=headers,
    )

    img_url = re.search(
        rf"\bimg\.bricklink\.com\/ItemImage\/[A-Z]+\/[0-9]+\/{part_id}\.png\b",
        response.text,
    ).group(0)

    print(img_url)

    img_response = session.get(url=f"https://{img_url}", headers=headers, stream=True)
    with open(f"{sync_directory}/{part_id}.png", "wb+") as out_file:
        shutil.copyfileobj(img_response.raw, out_file)

    del img_response


def get_session(username, password):
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

    print(f"logged in with status code {res.status_code}")

    # load homepage once to avoid initial redirect
    session.get("https://www.bricklink.com/")

    return session


if __name__ == "__main__":
    parts_to_sync = [
        "4865pb009",
        "2336p90",
        "4592c03",
        "6026c01",
    ]
    sync_directory = "images"

    # init session
    session = get_session(USERNAME, PASSWORD)

    # save images
    num_synced = 0
    for part_id in parts_to_sync:

        try:
            sync_part_image(session, part_id, sync_directory)

            num_synced += 1
            if num_synced % 50 == 0:
                print(f"synced {num_synced} of {len(parts_to_sync)}")

        except AttributeError:
            print(f"Sync failed for part {part_id}")

        # crude rate limit
        time.sleep(0.05)

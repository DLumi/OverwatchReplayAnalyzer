import time
import random
import requests
from pathlib import Path

import cv2
import pandas as pd
import numpy as np


def save_local_database(to: Path, api: str | None = None):
    out = to / 'overwatch_images.csv'

    if not api:
        api = "https://overwatch.fandom.com/api.php"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ORA image collector"
    })

    # Load checkpoint if it exists
    if out.exists():
        df = pd.read_csv(out)
        seen = set(df["name"])
    else:
        df = pd.DataFrame(columns=["name", "url"])
        seen = set()

    params = {
        "action": "query",
        "list": "allimages",
        "ailimit": "500",
        "aiprop": "url",
        "format": "json",
        "maxlag": "5",
    }

    while True:
        r = session.get(api, params=params, timeout=30)

        if r.status_code == 429:
            print('Too many requests, sleeping...')
            time.sleep(int(r.headers.get("Retry-After", "60")))
            continue

        r.raise_for_status()
        data = r.json()

        if data.get("error", {}).get("code") == "maxlag":
            print('Error, sleeping')
            time.sleep(int(r.headers.get("Retry-After", "10")))
            continue

        batch = []
        for img in data["query"]["allimages"]:
            name = img["name"]
            if name not in seen:
                batch.append({
                    "name": name,
                    "url": img["url"],
                })
                seen.add(name)

        if batch:
            batch_df = pd.DataFrame(batch)
            df = pd.concat([df, batch_df], ignore_index=True)
            df.to_csv(out, index=False)
            print(f"Saved {len(batch)} new images; total: {len(df)}")
        else:
            print(f"No new images; total: {len(df)}")

        cont = data.get("continue")
        if not cont:
            break

        params.update(cont)
        time.sleep(1.5 + random.random())

    print('Done!')


def retrieve_hero_portraits_2d(local_database_path: Path, save_to: Path):
    df = pd.read_csv(local_database_path)

    # look for stuff that has _Hero or 2D_portrait / 2D_Icon in the name + additional filtering
    pattern = (
        r"^(?!Spray_)"
        r"^(?!PI_)"
        r"^(?!Athena_)"
        r"(?!.*(?:_VP_|_Emote_|Season_|Intro_|Skin_|_Bug_|CTF_|Hanamura_|16-Bit_|16-bit_))"
        r".+_Hero"
        r"\.[A-Za-z0-9]+$"
        r"|2D_portrait|2D_Icon"
    )

    for row in df[df.name.str.contains(pattern)].itertuples():
        r = requests.get(row.url)

        buf = np.frombuffer(r.content, dtype=np.uint8)

        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

        name = row.name.rsplit('.', 1)[0]

        # unify naming
        if '2D' in name:
            name = name.split('2D')[0] + 'Hero'

        success, encoded = cv2.imencode('.png', img)
        if not success:
            raise ValueError(f"Failed to encode image as .png")

        (save_to / f'{name}.png').write_bytes(encoded.tobytes())

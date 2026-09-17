"""Download Rhiannon's English voice lines and write their transcript metadata."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


PROFILE_URL = "https://uttu.merui.net/profiles/rhiannon/"
AUDIO_HOST = "https://voice.merui.net/en/"
OUTPUT_DIR = Path("smaple") / "rhiannon_en_voice"


def clean_text(node) -> str:
    text = node.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def main() -> None:
    response = requests.get(PROFILE_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    records: list[dict[str, str]] = []
    for button in soup.select("button[data-audio-path]"):
        audio_path = button["data-audio-path"]
        row = button.find_parent(class_="voice-line-row")
        label_node = row.select_one("span") if row else None
        transcript_node = row.select_one(".voice-line-text") if row else None
        label = clean_text(label_node) if label_node else audio_path.rsplit("/", 1)[-1]
        transcript = transcript_node.get("data-en", "") if transcript_node else ""

        filename = f"{audio_path.rsplit('/', 1)[-1]}.ogg"
        audio_url = AUDIO_HOST + quote(audio_path, safe="/") + ".ogg"
        target = audio_dir / filename
        if not target.exists() or target.stat().st_size == 0:
            audio = requests.get(audio_url, timeout=120)
            audio.raise_for_status()
            target.write_bytes(audio.content)

        records.append(
            {
                "label": label,
                "transcript_en": transcript,
                "audio_file": str(Path("audio") / filename).replace("\\", "/"),
                "audio_path": audio_path,
                "source_url": audio_url,
            }
        )
        print(f"Downloaded {filename}: {label}")

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with (OUTPUT_DIR / "metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} voice lines to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Download every character's English voice lines from the UTTU archive.

The generated layout intentionally matches the existing ``sample`` datasets::

    sample/<profile-slug>_en_voice/
        audio/*.ogg
        metadata.json
        metadata.csv

Downloads are resumable: valid existing audio files are left untouched and all
new files are first written to a temporary sibling before being renamed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROFILE_INDEX = "https://uttu.merui.net/profiles/"
SITEMAP_URL = "https://uttu.merui.net/sitemap-0.xml"
AUDIO_HOST = "https://voice.merui.net/en/"
UTILITY_PAGES = {"analytics", "height", "items"}
# These datasets already exist under slightly different historical folder names.
EXISTING_SAMPLE_SLUGS = {"rhiannon", "ms._stranger"}
FOLDER_ALIASES = {"ms._stranger": "stranger"}
FIELDS = ("label", "transcript_en", "audio_file", "audio_path", "source_url")
_local = threading.local()


def session() -> requests.Session:
    current = getattr(_local, "session", None)
    if current is not None:
        return current
    current = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    current.mount("https://", HTTPAdapter(max_retries=retries))
    current.headers["User-Agent"] = "euphonia-voice-dataset/1.0"
    _local.session = current
    return current


def profile_urls() -> list[tuple[str, str]]:
    response = session().get(SITEMAP_URL, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    result: list[tuple[str, str]] = []
    for node in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        url = (node.text or "").strip()
        path = urlparse(url).path.rstrip("/")
        if not path.startswith("/profiles/"):
            continue
        slug = unquote(path.removeprefix("/profiles/"))
        if not slug or slug in UTILITY_PAGES:
            continue
        result.append((slug, url))
    return result


def folder_slug(slug: str) -> str:
    if slug in FOLDER_ALIASES:
        return FOLDER_ALIASES[slug]
    value = unicodedata.normalize("NFKC", slug).lower().replace(".", "")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"_+", "_", value).strip("_ .")
    if not value:
        raise ValueError(f"Profile slug cannot form a directory name: {slug!r}")
    return value


def clean_text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True))


def parse_profile(url: str) -> tuple[str, list[dict[str, str]]]:
    response = session().get(url, timeout=90)
    response.raise_for_status()
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    character_name = title.split(" | ", 1)[0]
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for button in soup.select("button[data-audio-path]"):
        audio_path = button.get("data-audio-path", "").strip()
        if not audio_path or audio_path in seen:
            continue
        seen.add(audio_path)
        row = button.find_parent(class_="voice-line-row")
        label_node = row.select_one("span") if row else None
        transcript_node = row.select_one(".voice-line-text") if row else None
        label = clean_text(label_node) if label_node else audio_path.rsplit("/", 1)[-1]
        transcript = transcript_node.get("data-en", "") if transcript_node else ""
        filename = f"{audio_path.rsplit('/', 1)[-1]}.ogg"
        audio_url = AUDIO_HOST + quote(audio_path, safe="/") + ".ogg"
        records.append(
            {
                "label": label,
                "transcript_en": transcript,
                "audio_file": f"audio/{filename}",
                "audio_path": audio_path,
                "source_url": audio_url,
            }
        )
    return character_name, records


def download_audio(url: str, target: Path) -> bool:
    if target.is_file() and target.stat().st_size > 0:
        return False
    temp = target.with_name(target.name + f".{os.getpid()}.{threading.get_ident()}.part")
    try:
        with session().get(url, timeout=(30, 120), stream=True) as response:
            response.raise_for_status()
            with temp.open("wb") as output:
                for chunk in response.iter_content(128 * 1024):
                    if chunk:
                        output.write(chunk)
        if temp.stat().st_size == 0:
            raise RuntimeError(f"Empty response for {url}")
        temp.replace(target)
        return True
    finally:
        if temp.exists():
            temp.unlink()


def write_metadata(output_dir: Path, records: list[dict[str, str]]) -> None:
    json_target = output_dir / "metadata.json"
    json_temp = output_dir / "metadata.json.part"
    with json_temp.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")
    json_temp.replace(json_target)

    csv_target = output_dir / "metadata.csv"
    csv_temp = output_dir / "metadata.csv.part"
    with csv_temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    csv_temp.replace(csv_target)


def download_character(slug: str, url: str, root: Path) -> dict[str, object]:
    character_name, records = parse_profile(url)
    if not records:
        raise RuntimeError(f"No English voice entries found at {url}")
    output_dir = root / f"{folder_slug(slug)}_en_voice"
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    available: list[dict[str, str]] = []
    unavailable: list[dict[str, str]] = []
    for record in records:
        target = output_dir / record["audio_file"]
        try:
            if download_audio(record["source_url"], target):
                downloaded += 1
            available.append(record)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            unavailable.append(
                {
                    "label": record["label"],
                    "audio_path": record["audio_path"],
                    "source_url": record["source_url"],
                    "reason": "HTTP 404",
                }
            )
    if not available:
        raise RuntimeError(f"No downloadable English voice entries found at {url}")
    write_metadata(output_dir, available)
    return {
        "slug": slug,
        "character_name": character_name,
        "folder": output_dir.name,
        "voice_lines": len(available),
        "downloaded": downloaded,
        "unavailable": unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sample"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--only", action="append", default=[], metavar="SLUG")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profiles = profile_urls()
    if args.only:
        wanted = set(args.only)
        profiles = [(slug, url) for slug, url in profiles if slug in wanted]
    elif not args.include_existing:
        profiles = [(slug, url) for slug, url in profiles if slug not in EXISTING_SAMPLE_SLUGS]

    print(f"Found {len(profiles)} character profiles", flush=True)
    started = time.monotonic()
    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {
            executor.submit(download_character, slug, url, args.output): (slug, url)
            for slug, url in profiles
        }
        for index, future in enumerate(as_completed(pending), 1):
            slug, url = pending[future]
            try:
                result = future.result()
                successes.append(result)
                print(
                    f"[{index}/{len(profiles)}] {result['character_name']}: "
                    f"{result['voice_lines']} lines ({result['downloaded']} new, "
                    f"{len(result['unavailable'])} unavailable)",
                    flush=True,
                )
            except Exception as error:  # Continue so one unavailable profile is not fatal.
                failures.append({"slug": slug, "url": url, "error": str(error)})
                print(f"[{index}/{len(profiles)}] FAILED {slug}: {error}", flush=True)

    report = {
        "source": PROFILE_INDEX,
        "sitemap": SITEMAP_URL,
        "output": str(args.output),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "characters_completed": len(successes),
        "characters_failed": len(failures),
        "voice_lines": sum(int(item["voice_lines"]) for item in successes),
        "characters": sorted(successes, key=lambda item: str(item["character_name"]).casefold()),
        "failures": failures,
    }
    report_target = args.output / "download_report.json"
    report_target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Completed {len(successes)}/{len(profiles)} characters and "
        f"{report['voice_lines']} voice lines; report: {report_target}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

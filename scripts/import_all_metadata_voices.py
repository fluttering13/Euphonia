"""Import every sample character into Euphonia and build emotion references."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_rhiannon_emotions import build
from euphonia.core import DATA, ROOT, VoiceStore
from euphonia.emotion import EmotionClassifier


LEGACY_VOICE_IDS = {
    "rhiannon_en_voice": "d474ca0976b5497687bd18e338ff0b18",
    "stranger_en_voice": "3dbe331b858f443eaf16dbe20fd77361",
}
LEGACY_NAMES = {
    "rhiannon_en_voice": "Rhiannon",
    "stranger_en_voice": "Ms. Stranger",
}


def character_names(source_root: Path) -> dict[str, str]:
    report_path = source_root / "download_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    names = {item["folder"]: item["character_name"] for item in report["characters"]}
    names.update(LEGACY_NAMES)
    return names


def select_base(source: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    valid = []
    for row in rows:
        transcript = row.get("transcript_en", "").strip()
        if not transcript:
            continue
        duration = sf.info(source / row["audio_file"]).duration
        if 3 <= duration <= 30:
            valid.append(row)
    if not valid:
        raise RuntimeError(f"No 3-30 second transcript-aligned sample in {source}")
    return next((row for row in valid if row["label"] == "First Encounter"), valid[0])


def write_voice_metadata(store: VoiceStore, voice: dict, dataset: str) -> dict:
    target = store.audio_path(voice["id"]).parent / "voice.json"
    metadata = json.loads(target.read_text(encoding="utf-8"))
    metadata["source_dataset"] = dataset
    temporary = target.with_name(f"voice.json.{os.getpid()}.part")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return metadata


def find_voice(store: VoiceStore, dataset: str, name: str) -> dict | None:
    voices = store.list()
    managed = [voice for voice in voices if voice.get("source_dataset") == dataset]
    if managed:
        return managed[0]
    legacy_id = LEGACY_VOICE_IDS.get(dataset)
    if legacy_id:
        return next((voice for voice in voices if voice["id"] == legacy_id), None)
    exact = [voice for voice in voices if voice["name"] == name]
    return exact[0] if len(exact) == 1 else None


def import_character(
    store: VoiceStore,
    source: Path,
    name: str,
    classifier: EmotionClassifier,
) -> dict[str, object]:
    rows = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    base = select_base(source, rows)
    voice = find_voice(store, source.name, name)
    action = "existing"
    if voice is None:
        voice = store.save(name, base["transcript_en"], source / base["audio_file"])
        action = "created"
    elif voice["name"] != name:
        # Keep the existing reference and Voice ID so saved audiobook mappings survive.
        voice = store.update(voice["id"], name, voice["transcript"])
        action = "renamed"
    voice = write_voice_metadata(store, voice, source.name)
    manifest = build(source, name, classifier=classifier, emit=False)
    return {
        "dataset": source.name,
        "character_name": name,
        "voice_id": voice["id"],
        "action": action,
        "base_label": base["label"],
        "emotion_samples": len(manifest["samples"]),
        "emotions": sorted({item["emotion"] for item in manifest["samples"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "sample")
    parser.add_argument("--only", action="append", default=[], metavar="DATASET_OR_NAME")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    names = character_names(source_root)
    sources = sorted(source_root.glob("*_en_voice"), key=lambda path: names.get(path.name, path.name).casefold())
    if args.only:
        wanted = {item.casefold() for item in args.only}
        sources = [path for path in sources if path.name.casefold() in wanted or names.get(path.name, "").casefold() in wanted]
    missing_names = [path.name for path in sources if path.name not in names]
    if missing_names:
        raise RuntimeError(f"Character names missing from download report: {missing_names}")
    if not args.only and len(sources) != 135:
        raise RuntimeError(f"Expected 135 character datasets, found {len(sources)}")

    print(f"Loading emotion classifier for {len(sources)} characters ...", flush=True)
    classifier = EmotionClassifier()
    store = VoiceStore()
    started = time.monotonic()
    completed = []
    failures = []
    for index, source in enumerate(sources, 1):
        name = names[source.name]
        try:
            result = import_character(store, source, name, classifier)
            completed.append(result)
            print(
                f"[{index}/{len(sources)}] {name}: {result['action']}, "
                f"{result['emotion_samples']} emotion samples",
                flush=True,
            )
        except Exception as error:
            failures.append({"dataset": source.name, "character_name": name, "error": str(error)})
            print(f"[{index}/{len(sources)}] FAILED {name}: {error}", flush=True)

    report = {
        "source_root": str(source_root),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "characters_completed": len(completed),
        "characters_failed": len(failures),
        "emotion_samples": sum(int(item["emotion_samples"]) for item in completed),
        "characters": completed,
        "failures": failures,
    }
    report_path = DATA / "voices" / "import_all_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Completed {len(completed)}/{len(sources)} characters with "
        f"{report['emotion_samples']} emotion samples; report: {report_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

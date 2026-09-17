"""Build transcript-aligned reference clips for emotion routing."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euphonia.core import DATA, ROOT, VoiceStore
from euphonia.emotion import EmotionClassifier


MAX_SAMPLES_PER_EMOTION = 2
EMOTIONS = ('anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise')


def build(source, voice_name, classifier=None, emit=True):
    source = Path(source).resolve()
    rows = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    voice = next(v for v in VoiceStore().list() if v['name'] == voice_name)
    voice_dir = VoiceStore().audio_path(voice['id']).parent
    emotions_dir = voice_dir / 'emotions'
    if emotions_dir.parent.resolve() != voice_dir.resolve():
        raise RuntimeError('情緒模板目標路徑驗證失敗。')
    if emotions_dir.exists():
        shutil.rmtree(emotions_dir)
    emotions_dir.mkdir(parents=True, exist_ok=True)
    classifier = classifier or EmotionClassifier()
    candidates = {}
    classified = []
    for row in rows:
        audio_path = source / row['audio_file']
        info = sf.info(audio_path)
        if not 3 <= info.duration <= 30:
            continue
        result = classifier.classify(row['transcript_en'])
        item = dict(row=row, duration=info.duration, result=result)
        classified.append(item)
        candidates.setdefault(result['label'], []).append(item)
    selected = []
    for emotion, items in candidates.items():
        # High-confidence complete utterances give cleaner reference conditioning.
        selected.extend(dict(item, route_emotion=emotion) for item in
                        sorted(items, key=lambda item: item['result']['score'], reverse=True)
                        [:MAX_SAMPLES_PER_EMOTION])
    # A voice pack may have no utterance whose top prediction is one of the seven
    # classes. Keep routing complete by assigning the best unused candidate for it.
    used_audio = {item['row']['audio_file'] for item in selected}
    present = {item['route_emotion'] for item in selected}
    for emotion in EMOTIONS:
        if emotion in present:
            continue
        available = [item for item in classified if item['row']['audio_file'] not in used_audio]
        if not available:
            continue
        best = max(available, key=lambda item: item['result']['scores'].get(emotion, 0))
        selected.append(dict(best, route_emotion=emotion))
        used_audio.add(best['row']['audio_file'])
    selected.sort(key=lambda item: (item['route_emotion'],
                                    -item['result']['scores'].get(item['route_emotion'], 0)))

    manifest = dict(version=2, model='j-hartmann/emotion-english-distilroberta-base',
                    selection='same class, then nearest emotion score vector', samples=[])
    for index, item in enumerate(selected, 1):
        row, result = item['row'], item['result']
        emotion = item['route_emotion']
        audio_path = source / row['audio_file']
        audio, sample_rate = sf.read(audio_path, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        sample_id = f"{emotion}-{index:02d}"
        target_dir = emotions_dir / 'samples' / sample_id
        target_dir.mkdir(parents=True, exist_ok=True)
        sf.write(target_dir / 'reference.wav', audio, sample_rate, subtype='PCM_16')
        transcript = row['transcript_en'].strip()
        (target_dir / 'transcript.txt').write_text(transcript, encoding='utf-8')
        manifest['samples'].append(dict(
            id=sample_id, emotion=emotion, label=row['label'],
            reference=str((target_dir / 'reference.wav').relative_to(voice_dir)).replace('\\', '/'),
            transcript=transcript, seconds=round(len(audio) / sample_rate, 3),
            source=row['audio_file'], score=result['scores'].get(emotion, 0), scores=result['scores']))
    (voice_dir / 'emotion_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    report = DATA / 'sample-tests' / source.name / 'emotion-bank.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if emit:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'smaple' / 'rhiannon_en_voice')
    parser.add_argument('--voice-name', default='Rhiannon（英文採樣）')
    args = parser.parse_args()
    build(args.source, args.voice_name)


if __name__ == '__main__':
    main()

"""Create or refresh a voice and its emotion mapping from metadata.json."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euphonia.core import ROOT, VoiceStore
from build_rhiannon_emotions import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('voice_name')
    parser.add_argument('--base-label', default='First Encounter')
    args = parser.parse_args()
    source = args.source.resolve()
    rows = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    base = next((row for row in rows if row['label'] == args.base_label), None)
    if base is None:
        raise RuntimeError(f'metadata.json 找不到主要採樣標籤：{args.base_label}')
    store = VoiceStore()
    matches = [voice for voice in store.list() if voice['name'] == args.voice_name]
    if len(matches) > 1:
        raise RuntimeError(f'找到多個同名角色：{args.voice_name}')
    audio = source / base['audio_file']
    if matches:
        voice = store.update(matches[0]['id'], args.voice_name, base['transcript_en'], str(audio), None)
        action = 'updated'
    else:
        voice = store.save(args.voice_name, base['transcript_en'], str(audio))
        action = 'created'
    manifest = build(source, args.voice_name)
    print(json.dumps(dict(action=action, voice=voice, emotion_samples=len(manifest['samples']),
                          source=str(source.relative_to(ROOT))), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

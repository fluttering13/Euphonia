"""Rebuild the stored Rhiannon voice from several transcript-aligned samples."""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euphonia.core import DATA, ROOT, VoiceStore


VOICE_NAME = 'Rhiannon（英文採樣）'
SOURCE = ROOT / 'smaple' / 'rhiannon_en_voice'
# One clean conversational reference plus complete short lines with varied delivery.
# Together with 80 ms gaps this stays below the application's 30-second limit.
SELECTED_FILES = [
    'play_hero3146_mainvoc_1.ogg',
    'play_hero3146_fightingvoc_19.ogg',
    'play_hero3146_fightingvoc_20.ogg',
    'play_hero3146_fightingvoc_21.ogg',
    'play_hero3146_fightingvoc_22.ogg',
    'play_hero3146_fightingvoc_23.ogg',
    'play_hero3146_fightingvoc_24.ogg',
    'play_hero3146_fightingvoc_25.ogg',
    'play_hero3146_fightingvoc_26.ogg',
    'play_hero3146_fightingvoc_27.ogg',
    'play_hero3146_fightingvoc_30.ogg',
]
GAP_SECONDS = 0.08


def main():
    metadata_path = SOURCE / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    by_file = {Path(row['audio_file']).name: row for row in metadata}
    missing = [name for name in SELECTED_FILES if name not in by_file]
    if missing:
        raise RuntimeError('metadata.json 缺少音檔對應：' + ', '.join(missing))

    store = VoiceStore()
    matches = [voice for voice in store.list() if voice['name'] == VOICE_NAME]
    if len(matches) != 1:
        raise RuntimeError(f'預期找到一個「{VOICE_NAME}」，實際找到 {len(matches)} 個。')
    voice = matches[0]
    target = store.audio_path(voice['id']).resolve()
    voice_dir = target.parent
    voices_root = store.root.resolve()
    if voices_root not in target.parents or voice_dir.name != voice['id']:
        raise RuntimeError('Voice 目標路徑驗證失敗。')

    sample_rate = None
    chunks = []
    sources = []
    transcripts = []
    for filename in SELECTED_FILES:
        row = by_file[filename]
        path = (SOURCE / row['audio_file']).resolve()
        if SOURCE.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f'找不到安全的來源音檔：{filename}')
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate is None:
            sample_rate = rate
        if rate != sample_rate:
            raise RuntimeError(f'來源取樣率不一致：{filename} 是 {rate} Hz')
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
            raise RuntimeError(f'來源音檔無有效聲音：{filename}')
        chunks.append(audio)
        transcripts.append(row['transcript_en'].strip())
        sources.append(dict(file=row['audio_file'], label=row['label'],
                            seconds=round(len(audio) / rate, 3)))

    gap = np.zeros(round(sample_rate * GAP_SECONDS), dtype=np.float32)
    combined = np.concatenate([part for i, audio in enumerate(chunks)
                               for part in ((gap if i else np.empty(0, dtype=np.float32)), audio)])
    duration = len(combined) / sample_rate
    if not 3 <= duration <= 30:
        raise RuntimeError(f'重建後參考音訊長度 {duration:.3f} 秒，不在 3～30 秒內。')

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_dir = voice_dir / 'backups' / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_dir / 'reference.wav')
    shutil.copy2(voice_dir / 'voice.json', backup_dir / 'voice.json')

    transcript = '\n'.join(transcripts)
    rebuilt = dict(voice, transcript=transcript, duration=round(duration, 1),
                   reference_build='metadata-aligned composite', reference_sources=sources)
    temporary_audio = voice_dir / 'reference.rebuild.wav'
    temporary_meta = voice_dir / 'voice.rebuild.json'
    try:
        sf.write(temporary_audio, combined, sample_rate, subtype='PCM_16')
        temporary_meta.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary_audio, target)
        os.replace(temporary_meta, voice_dir / 'voice.json')
    finally:
        temporary_audio.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)

    report_dir = DATA / 'sample-tests' / 'rhiannon'
    report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(voice_id=voice['id'], voice_name=VOICE_NAME, duration_seconds=round(duration, 3),
                  sample_rate=sample_rate, selected_count=len(sources), available_count=len(metadata),
                  sources=sources, backup=str(backup_dir.relative_to(ROOT)),
                  reference=str(target.relative_to(ROOT)))
    (report_dir / 'rebuild-report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

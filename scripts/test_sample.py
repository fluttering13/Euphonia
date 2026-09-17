"""Import the supplied Rhiannon sample and exercise real OCR and voice cloning."""
import json
from difflib import SequenceMatcher
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QSettings

from euphonia.core import DATA, ROOT, Engine, VoiceStore


def main():
    source = next((ROOT / name / 'rhiannon_en_voice' for name in ('sample', 'smaple')
                   if (ROOT / name / 'rhiannon_en_voice' / 'metadata.json').is_file()), None)
    if source is None:
        raise FileNotFoundError('Cannot find sample/rhiannon_en_voice/metadata.json')
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    reference = next(row for row in metadata if row['label'] == 'First Encounter')
    store = VoiceStore()
    name = 'Rhiannon（英文採樣）'
    voice = next((v for v in store.list() if v['name'] == name and v['transcript'] == reference['transcript_en']), None)
    if voice is None:
        voice = store.save(name, reference['transcript_en'], source / reference['audio_file'])
    assert voice in VoiceStore().list(), 'Saved voice did not survive store reload'
    settings = QSettings('Euphonia', 'Euphonia')
    settings.setValue('voice', voice['id'])
    settings.sync()
    print('Saved voice:', voice['name'], voice['id'], flush=True)

    output_dir = DATA / 'sample-tests' / 'rhiannon'
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'voice': voice,
        'reference_source': str((source / reference['audio_file']).relative_to(ROOT)),
        'reference_label': reference['label'],
        'reference_saved': str(store.audio_path(voice['id']).relative_to(ROOT)),
        'tests': [],
        'limitations': 'Audio validity is checked automatically; voice similarity and pronunciation require listening.',
    }
    engine = Engine(store)
    english = "The morning sky looks beautiful. Shall we explore the forest together?"
    chinese = '你好，今天我們一起出發，看看森林裡有什麼新發現。'
    canvas = Image.new('RGB', (1500, 160), 'white')
    ImageDraw.Draw(canvas).text((25, 42), chinese, fill='black', font=ImageFont.truetype('C:/Windows/Fonts/msjh.ttc', 40))
    canvas.save(output_dir / 'ocr-input.png')
    recognized = engine.recognize(np.asarray(canvas)[:, :, ::-1].copy())
    print('OCR:', recognized, flush=True)
    normalize = lambda text: ''.join(c for c in text if '\u4e00' <= c <= '\u9fff')
    similarity = SequenceMatcher(None, normalize(recognized), normalize(chinese)).ratio()
    assert similarity >= 0.9, 'OCR content differs substantially from test text'
    # The supplied expected text serves as an explicit correction, like editing in the UI.
    report['ocr'] = {'expected': chinese, 'recognized': recognized,
                     'character_similarity': similarity, 'corrected_for_reading': chinese,
                     'correction_needed': recognized != chinese}
    if recognized != chinese:
        print('OCR correction applied before reading:', chinese, flush=True)
    for language, text, filename in [('English', english, 'english.wav'), ('Chinese', chinese, 'chinese-ocr.wav')]:
        start = time.perf_counter()
        generated = engine.synthesize(voice, text, language, lambda message: print(message, flush=True))
        elapsed = time.perf_counter() - start
        samples, sr = sf.read(generated)
        assert len(samples) > sr and np.isfinite(samples).all()
        assert np.max(np.abs(samples)) > 0.001, 'Generated audio is silent'
        target = output_dir / filename
        shutil.copy2(generated, target)
        stats = dict(language=language, text=text, file=str(target.relative_to(ROOT)),
                     sample_rate=sr, duration_seconds=round(len(samples) / sr, 2),
                     elapsed_seconds=round(elapsed, 2), peak=round(float(np.max(np.abs(samples))), 4),
                     rms=round(float(np.sqrt(np.mean(samples ** 2))), 4))
        report['tests'].append(stats)
        (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('PASS:', language, stats['duration_seconds'], 'seconds;', target, flush=True)
    assert voice['id'] in engine.prompts, 'Voice prompt was not cached'
    print('Sample voice, English TTS, and Chinese OCR-to-TTS passed.', flush=True)


if __name__ == '__main__':
    main()

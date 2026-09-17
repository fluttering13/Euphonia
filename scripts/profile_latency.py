"""Measure the installed application path; does not modify runtime settings."""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import soundfile as sf
from euphonia.core import DATA, Engine, VoiceStore


def main():
    engine = Engine(VoiceStore())
    voice = next(v for v in engine.store.list() if v['name'] == 'Rhiannon（英文採樣）')
    image = np.asarray(Image.open(DATA / 'sample-tests/rhiannon/ocr-input.png').convert('RGB'))[:, :, ::-1].copy()
    report = {'ocr_seconds': [], 'runs': [], 'notes': [
        'Wall-clock measurements in a separate process; OS disk caches may already be warm.',
        'CUDA synchronized at stage boundaries. No manual selection time included.',
        'Timings exclude actual audible playback; player startup is measured separately.',
        'No control over concurrent game or desktop GPU load.',
        'Outputs have not been checked with ASR for completeness or pronunciation.',
    ]}
    for _ in range(4):
        start = time.perf_counter()
        recognized = engine.recognize(image)
        report['ocr_seconds'].append(time.perf_counter() - start)
    report['recognized'] = recognized
    print('OCR seconds:', report['ocr_seconds'], flush=True)
    start = time.perf_counter()
    import torch
    torch.cuda.synchronize()
    report['torch_import_cuda_init_seconds'] = time.perf_counter() - start
    report['environment'] = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                             'dtype': 'bfloat16', 'attention': 'sdpa',
                             'reference_seconds': sf.info(engine.store.audio_path(voice['id'])).duration}
    start = time.perf_counter()
    model = engine.load(lambda message: print(message, flush=True))
    torch.cuda.synchronize()
    report['qwen_import_and_model_load_seconds'] = time.perf_counter() - start
    current = {}

    def wrap(obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            current[label] = current.get(label, 0) + time.perf_counter() - start
            if label == 'autoregressive_seconds':
                current.setdefault('generated_codec_frames', []).extend(len(codes) for codes in result[0])
            return result
        setattr(obj, name, measured)

    wrap(model, 'create_voice_clone_prompt', 'voice_prompt_seconds')
    wrap(model.model, 'generate', 'autoregressive_seconds')
    wrap(model.model.speech_tokenizer, 'decode', 'waveform_decode_seconds')
    sentences = [('short', '你好。'), ('medium', '你好，今天我們一起出發，看看森林裡有什麼新發現。'),
                 ('long', '今天我們準備前往森林深處，尋找傳說中的古老遺跡。請先檢查背包裡的食物和裝備，然後到村莊入口集合。')]
    language = 'English' if '--english' in sys.argv else 'Chinese'
    if language == 'English':
        sentences = [('english_sentence', 'The morning sky looks beautiful. Shall we explore the forest together?')]
    report['language'] = language
    destination = DATA / ('latency-profile-english.json' if language == 'English' else 'latency-profile.json')
    runs = [('first_request', sentences[min(1, len(sentences) - 1)][1])] + [entry for entry in sentences for _ in range(3)]
    for index, (label, text) in enumerate(runs):
        current.clear()
        torch.manual_seed(100 + index)
        start = time.perf_counter()
        output = engine.synthesize(voice, text, language, lambda _: None)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        info = sf.info(output)
        run = dict(label=label, chars=len(text), total_seconds=elapsed, audio_seconds=info.duration,
                   rtf=elapsed / info.duration, **current)
        accounted = sum(current.get(key, 0) for key in ('voice_prompt_seconds', 'autoregressive_seconds', 'waveform_decode_seconds'))
        run['other_seconds'] = elapsed - accounted
        run['file'] = str(Path(output).relative_to(DATA.parent))
        report['runs'].append(run)
        print(json.dumps(run, ensure_ascii=False), flush=True)
        report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['medians'] = {label: statistics.median(r['total_seconds'] for r in report['runs'] if r['label'] == label)
                         for label, _ in sentences}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE:', report['medians'], flush=True)


if __name__ == '__main__':
    main()

"""Local ASR sanity check; not a perceptual voice-similarity evaluation."""
import json
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['faster', 'qwen_streaming', 'f5', 'zipvoice'])
    parser.add_argument('--output', default='comparison-asr.json')
    args = parser.parse_args()
    directory = ROOT / 'data/models/whisper-tiny-en'
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(directory, local_files_only=True).to('cuda').eval()
    report = []
    for backend in args.models:
        path = ROOT / 'data' / f'comparison-{backend}.json'
        if not path.is_file():
            continue
        runs = json.loads(path.read_text(encoding='utf-8')).get('runs', [])
        for run in runs:
            if run['label'] == 'first':
                continue
            audio, sr = sf.read(run['file'], dtype='float32')
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            features = processor(audio, sampling_rate=16000, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                tokens = model.generate(**features, max_new_tokens=256)
            recognized = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            item = dict(backend=backend, file=run['file'], expected=run['text'], recognized=recognized)
            report.append(item)
            print(json.dumps(item), flush=True)
    (ROOT / 'data' / args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

"""Benchmark the same persistent worker path used by the desktop application."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from euphonia.backends import BackendProcess

TEXTS = {
    'short': 'Watch out! There is an enemy behind you.',
    'medium': 'The morning sky looks beautiful. Shall we explore the forest together?',
    'long': 'We need to reach the ancient tower before sunset. Take the narrow path through the forest, and keep your weapons ready. There may be enemies waiting near the bridge.',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backend', choices=['faster', 'qwen_streaming', 'f5', 'zipvoice'])
    args = parser.parse_args()
    voice_path = ROOT / 'data/voices/d474ca0976b5497687bd18e338ff0b18'
    voice = json.loads((voice_path / 'voice.json').read_text(encoding='utf-8'))
    destination = ROOT / 'data' / f'comparison-{args.backend}.json'
    report = dict(backend=args.backend, reference=str(voice_path / 'reference.wav'), runs=[],
                  notes=['Full waveform generation and WAV write, not time to first sound.',
                         'Request time includes IPC; first request also includes process/model startup.',
                         'No controlled game load. Audio quality requires listening; no ASR verification yet.'])
    client = BackendProcess(args.backend)
    try:
        for i, label in enumerate(['first', 'medium', 'medium', 'medium', 'short', 'long']):
            text = TEXTS['medium' if label == 'first' else label]
            start = time.perf_counter()
            output = client.request(dict(reference=str(voice_path / 'reference.wav'),
                                         transcript=voice['transcript'], text=text,
                                         language='English', seed=100 + i),
                                    lambda msg: print(msg, flush=True))
            row = dict(client.last_metrics, label=label, text=text,
                       request_seconds=time.perf_counter() - start)
            report['runs'].append(row)
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report['warm_medium_median'] = statistics.median(
            r['request_seconds'] for r in report['runs'] if r['label'] == 'medium')
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        client.close()
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

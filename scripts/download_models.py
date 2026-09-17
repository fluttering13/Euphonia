"""Download the model weights retained by Euphonia.

The accelerated and streaming Qwen backends share the same Qwen3-TTS 0.6B
weights, so they only need one download.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_model import download
from euphonia.core import MODEL


def download_selected(model):
    models_root = ROOT / 'data/models'
    if model == 'qwen':
        download(MODEL, ROOT / 'data/model')
    elif model == 'f5':
        download('charactr/vocos-mel-24khz', models_root / 'vocos-mel-24khz',
                 {'config.yaml', 'pytorch_model.bin'},
                 revision='0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21')
        download('SWivid/F5-TTS', models_root / 'f5',
                 {'F5TTS_v1_Base/model_1250000.safetensors',
                  'F5TTS_v1_Base/vocab.txt'},
                 revision='84e5a410d9cead4de2f847e7c9369a6440bdfaca')
    elif model == 'zipvoice':
        download('charactr/vocos-mel-24khz', models_root / 'vocos-mel-24khz',
                 {'config.yaml', 'pytorch_model.bin'},
                 revision='0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21')
        download('k2-fsa/ZipVoice', models_root / 'zipvoice',
                 {'zipvoice_distill/model.safetensors',
                  'zipvoice_distill/model.json',
                  'zipvoice_distill/tokens.txt'},
                 revision='4ed45fb6e7e9527b780bef9e097a04bf13fe4e6b')
    else:
        raise ValueError(f'Unknown model: {model}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'model', nargs='?', default='all',
        choices=('all', 'qwen', 'faster', 'qwen_streaming', 'f5', 'zipvoice'),
        help='Model weights to download (default: all).')
    selected = parser.parse_args().model
    if selected in {'faster', 'qwen_streaming'}:
        selected = 'qwen'
    targets = ('qwen', 'f5', 'zipvoice') if selected == 'all' else (selected,)
    for target in targets:
        print(f'\n== Downloading {target} ==', flush=True)
        download_selected(target)


if __name__ == '__main__':
    main()

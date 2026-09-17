"""Download the small INT8 English emotion classifier used by Euphonia."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download
from euphonia.emotion import MODEL_DIR, MULTILINGUAL_MODEL_DIR


FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.json', 'merges.txt', 'onnx/model_int8.onnx']
MODELS = [
    ('onnx-community/emotion-english-distilroberta-base-ONNX', MODEL_DIR, FILES),
    ('TrumpMcDonaldz/xlm-emo-t-ONNX', MULTILINGUAL_MODEL_DIR,
     ['config.json', 'tokenizer.json', 'tokenizer_config.json',
      'special_tokens_map.json', 'onnx/model_int8.onnx']),
]


def main():
    for repo, model_dir, files in MODELS:
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            source = Path(hf_hub_download(repo, filename=filename))
            target = model_dir / ('model_int8.onnx' if filename.startswith('onnx/') else Path(filename).name)
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            print(target.relative_to(model_dir.parent.parent))


if __name__ == '__main__':
    main()

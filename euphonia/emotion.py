"""Fast local text-emotion routing for transcript-aligned voice references."""
import json
import time
from pathlib import Path

import numpy as np

from .core import DATA


MODEL_DIR = DATA / 'models' / 'emotion-english-distilroberta-base-onnx'
MULTILINGUAL_MODEL_DIR = DATA / 'models' / 'xlm-emo-t-onnx'


class EmotionClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        if not (model_dir / 'model_int8.onnx').is_file():
            raise FileNotFoundError('情緒模型尚未安裝，請執行 scripts/setup_emotion.py。')
        config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
        self.labels = {int(key): value for key, value in config['id2label'].items()}
        self.tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / 'model_int8.onnx'),
                                            sess_options=options, providers=['CPUExecutionProvider'])
        self.last_seconds = None

    def classify(self, text):
        started = time.perf_counter()
        encoding = self.tokenizer.encode((text or '').strip())
        ids = encoding.ids[:128]
        if not ids:
            return dict(label='neutral', score=1.0, seconds=0.0, scores={'neutral': 1.0})
        inputs = {
            'input_ids': np.asarray([ids], dtype=np.int64),
            'attention_mask': np.ones((1, len(ids)), dtype=np.int64),
        }
        logits = self.session.run(None, inputs)[0][0].astype(np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        scores = {self.labels[i]: float(score) for i, score in enumerate(probabilities)}
        label = max(scores, key=scores.get)
        # Weak predictions sound better with the stable neutral reference.
        if scores[label] < 0.40:
            label = 'neutral'
        self.last_seconds = time.perf_counter() - started
        return dict(label=label, score=scores[label], seconds=self.last_seconds, scores=scores)


def model_for_text(text):
    """Use the multilingual model when CJK characters are present."""
    return MULTILINGUAL_MODEL_DIR if any('\u3400' <= char <= '\u9fff' for char in (text or '')) else MODEL_DIR

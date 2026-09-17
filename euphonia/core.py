"""Local persistence and lazy inference; no UI dependencies."""
import json
import re
import shutil
import uuid
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODEL = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'


class VoiceStore:
    def __init__(self, root=DATA / 'voices'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self):
        voices = []
        for path in self.root.glob('*/voice.json'):
            try:
                item = json.loads(path.read_text(encoding='utf-8'))
                if item['id'] == path.parent.name and (path.parent / 'reference.wav').is_file():
                    voices.append(item)
            except (ValueError, KeyError, OSError):
                continue
        return sorted(voices, key=lambda v: v['name'].casefold())

    def audio_path(self, voice_id):
        if not re.fullmatch(r'[0-9a-f]{32}', voice_id):
            raise ValueError('角色識別碼無效')
        return self.root / voice_id / 'reference.wav'

    def reference_path(self, voice):
        target = self.audio_path(voice['id'])
        relative = voice.get('reference')
        if relative:
            candidate = (target.parent / relative).resolve()
            if target.parent.resolve() not in candidate.parents or not candidate.is_file():
                raise ValueError('情緒參考音檔路徑無效。')
            return candidate
        return target

    @staticmethod
    def read_reference(audio_path, minimum_seconds=3):
        info = sf.info(audio_path)
        if not minimum_seconds <= info.duration <= 30:
            raise ValueError(f'參考語音需為 {minimum_seconds}～30 秒，建議使用清楚、無背景音樂的語音。')
        audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
            raise ValueError('音檔沒有有效的聲音，請重新選擇。')
        return audio, sr

    def save(self, name, transcript, audio_path, emotion_templates=None):
        if not name.strip() or not transcript.strip():
            raise ValueError('請輸入角色名稱和音檔逐字稿。')
        audio, sr = self.read_reference(audio_path)
        voice_id = uuid.uuid4().hex
        target = self.audio_path(voice_id)
        target.parent.mkdir()
        try:
            sf.write(target, audio, sr, subtype='PCM_16')
            voice = dict(id=voice_id, name=name.strip(), transcript=transcript.strip(), duration=round(len(audio) / sr, 1))
            (target.parent / 'voice.json').write_text(json.dumps(voice, ensure_ascii=False, indent=2), encoding='utf-8')
            if emotion_templates:
                self.save_emotion_templates(voice, emotion_templates)
        except Exception:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise
        return voice

    def save_emotion_templates(self, voice, templates):
        allowed = {'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'}
        root = self.audio_path(voice['id']).parent
        prepared = []
        for item in templates:
            emotion = item.get('emotion')
            transcript = item.get('transcript', '').strip()
            if emotion not in allowed or not transcript:
                raise ValueError('情緒模板的分類或逐字稿無效。')
            audio, rate = self.read_reference(item['audio_path'], minimum_seconds=.4)
            prepared.append((item, emotion, transcript, audio, rate))
        emotions_root = root / 'emotions'
        if emotions_root.exists():
            shutil.rmtree(emotions_root)
        manifest = dict(version=2, model='automatic text emotion classification', samples=[])
        for index, (item, emotion, transcript, audio, rate) in enumerate(prepared, 1):
            sample_id = f'sample-{index:03d}'
            sample_dir = root / 'emotions' / 'samples' / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            sf.write(sample_dir / 'reference.wav', audio, rate, subtype='PCM_16')
            (sample_dir / 'transcript.txt').write_text(transcript, encoding='utf-8')
            manifest['samples'].append(dict(
                id=sample_id, emotion=emotion,
                reference=f'emotions/samples/{sample_id}/reference.wav', transcript=transcript,
                seconds=round(len(audio) / rate, 3), score=item.get('score'),
                scores=item.get('scores', {emotion: item.get('score', 0)})))
        (root / 'emotion_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return manifest

    def emotion_templates(self, voice):
        root = self.audio_path(voice['id']).parent
        manifest_path = root / 'emotion_manifest.json'
        if not manifest_path.is_file():
            return []
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        samples = manifest.get('samples')
        if samples is None:
            samples = [dict(item, id=emotion, emotion=emotion)
                       for emotion, item in manifest.get('emotions', {}).items()]
        return [dict(audio_path=str((root / item['reference']).resolve()),
                     transcript=item['transcript'], emotion=item['emotion'],
                     score=item.get('score') or 0, scores=item.get('scores') or {},
                     label=item.get('label')) for item in samples]

    def update(self, voice_id, name, transcript, audio_path=None, emotion_templates=None):
        target = self.audio_path(voice_id)
        meta_path = target.parent / 'voice.json'
        if not target.is_file() or not meta_path.is_file() or not name.strip() or not transcript.strip():
            raise ValueError('角色或主要參考資料無效。')
        current = json.loads(meta_path.read_text(encoding='utf-8'))
        if audio_path:
            audio, rate = self.read_reference(audio_path)
        else:
            audio, rate = sf.read(target, dtype='float32')
        # Read every template before replacing files; edit-mode paths may point into this voice folder.
        templates = None if emotion_templates is None else [dict(item) for item in emotion_templates]
        prepared_templates = None
        if templates is not None:
            prepared_templates = []
            for item in templates:
                sample, sample_rate = self.read_reference(item['audio_path'], minimum_seconds=.4)
                copy = dict(item, _audio=sample, _rate=sample_rate)
                prepared_templates.append(copy)

        backup = target.parent / 'backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        backup.mkdir(parents=True)
        shutil.copy2(target, backup / 'reference.wav')
        shutil.copy2(meta_path, backup / 'voice.json')
        if (target.parent / 'emotion_manifest.json').is_file():
            shutil.copy2(target.parent / 'emotion_manifest.json', backup / 'emotion_manifest.json')
        if (target.parent / 'emotions').is_dir():
            shutil.copytree(target.parent / 'emotions', backup / 'emotions')

        sf.write(target, audio, rate, subtype='PCM_16')
        updated = dict(current, id=voice_id, name=name.strip(), transcript=transcript.strip(),
                       duration=round(len(audio) / rate, 1))
        meta_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding='utf-8')
        if prepared_templates is not None:
            if prepared_templates:
                # Materialize staged arrays temporarily outside the directory save_emotion_templates replaces.
                staged = backup / 'staged'
                staged.mkdir()
                normalized = []
                for index, item in enumerate(prepared_templates, 1):
                    path = staged / f'{index}.wav'
                    sf.write(path, item.pop('_audio'), item.pop('_rate'), subtype='PCM_16')
                    normalized.append(dict(item, audio_path=str(path)))
                self.save_emotion_templates(updated, normalized)
                shutil.rmtree(staged)
            else:
                shutil.rmtree(target.parent / 'emotions', ignore_errors=True)
                (target.parent / 'emotion_manifest.json').unlink(missing_ok=True)
        return updated

    def delete(self, voice_id):
        target = self.audio_path(voice_id)
        if target.parent.parent.resolve() != self.root.resolve():
            raise ValueError('角色路徑無效。')
        shutil.rmtree(target.parent)


def split_text(text, limit=160):
    sentences = re.findall(r'[^。！？!?\n]+[。！？!?\n]*|[。！？!?\n]+', text.strip())
    parts = []
    for sentence in sentences:
        while len(sentence) > limit:
            cut = max(sentence.rfind(p, 0, limit) for p in '，,；;、 ')
            cut = cut + 1 if cut > limit // 2 else limit
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:]
        if sentence.strip():
            parts.append(sentence.strip())
    return parts


class Engine:
    def __init__(self, store):
        self.store = store
        self.ocr = None
        self.ocr_lock = threading.RLock()
        self.ocr_fast = True
        self.backend = 'faster'
        self.remote = None
        self.last_metrics = None
        self.prepared = set()
        self.last_ocr_seconds = None
        self.emotion_classifiers = {}
        self.last_emotion = None
        self.streaming_silence_threshold_db = -38.0
        self.streaming_interval_ms = 75

    def emotion_variants(self, voice):
        manifest_path = self.store.audio_path(voice['id']).parent / 'emotion_manifest.json'
        if not manifest_path.is_file():
            return {}
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        variants = {}
        samples = manifest.get('samples')
        if samples is None:  # Version 1 compatibility.
            samples = [dict(item, id=emotion, emotion=emotion)
                       for emotion, item in manifest.get('emotions', {}).items()]
        for item in samples:
            sample_id = item['id']
            variants[sample_id] = dict(voice, reference=item['reference'], transcript=item['transcript'],
                                       cache_id=f"{voice['id']}:{sample_id}", emotion=item['emotion'],
                                       emotion_sample=item)
        return variants

    def emotional_voice(self, voice, text):
        variants = self.emotion_variants(voice)
        if not variants:
            self.last_emotion = None
            return voice
        try:
            from .emotion import EmotionClassifier, model_for_text
            model_dir = model_for_text(text)
            if model_dir not in self.emotion_classifiers:
                self.emotion_classifiers[model_dir] = EmotionClassifier(model_dir)
            result = self.emotion_classifiers[model_dir].classify(text)
            candidates = [variant for variant in variants.values()
                          if variant['emotion'] == result['label']]
            if not candidates:
                candidates = [variant for variant in variants.values() if variant['emotion'] == 'neutral']
            if not candidates:
                candidates = list(variants.values())
            def distance(variant):
                sample = variant['emotion_sample']
                scores = sample.get('scores') or {}
                common = set(result['scores']) & set(scores)
                if common:
                    return sum((result['scores'][key] - scores[key]) ** 2 for key in common) / len(common)
                return abs(result['score'] - (sample.get('score') or 0))
            selected = min(candidates, key=distance)
            sample = selected['emotion_sample']
            self.last_emotion = dict(result, routed=selected['emotion'], sample_id=sample['id'],
                                     sample_label=sample.get('label'), distance=distance(selected))
            return selected
        except (OSError, ValueError, KeyError):
            self.last_emotion = dict(label='neutral', routed='neutral', score=0, seconds=0, fallback=True)
            return next((variant for variant in variants.values() if variant['emotion'] == 'neutral'), voice)

    def prepare(self, voice, language, progress, cancelled=lambda: False):
        key = (self.backend, voice['id'], language)
        resident = self.remote is not None and self.remote.process is not None and self.remote.process.poll() is None
        if key in self.prepared and resident:
            return True
        progress('正在預載模型與角色並暖機，完成後可連續朗讀…')
        if self.ocr is None:
            self.recognize(np.full((64, 256, 3), 255, dtype=np.uint8))
        text = 'Ready to begin.' if language in ('English', 'Auto') else '你好。'
        output = self.synthesize(voice, text, language, progress, cancelled)
        if output:
            Path(output).unlink(missing_ok=True)
            variants = self.emotion_variants(voice)
            for index, (emotion, variant) in enumerate(variants.items(), 1):
                if cancelled():
                    return None
                progress(f'正在預載情緒聲音 {index} / {len(variants)}：{emotion}')
                self.prepare_conditioning(variant, progress, cancelled)
            self.prepared.add(key)
            return True
        return False

    def prepare_conditioning(self, voice, progress, cancelled):
        reference = str(self.store.reference_path(voice))
        if self.remote is None:
            from .backends import BackendProcess
            self.remote = BackendProcess(self.backend)
        self.remote.request(dict(action='prepare', reference=reference,
                                 transcript=voice['transcript']), progress, cancelled)

    def set_backend(self, backend):
        from .backends import BACKENDS
        if backend not in BACKENDS:
            raise ValueError('未知的語音模型。')
        if backend != self.backend:
            self.close()
            self.backend = backend

    def close(self):
        self.prepared.clear()
        if self.remote is not None:
            self.remote.close()
            self.remote = None

    def recognize(self, image):
        # The monitor and model warmup share this one OCR instance.
        with self.ocr_lock:
            return self._recognize(image)

    def _recognize(self, image):
        started = time.perf_counter()
        if self.ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            options = dict(intra_op_num_threads=4, inter_op_num_threads=1,
                           det_limit_type='max', det_limit_side_len=960) if self.ocr_fast else {}
            self.ocr = RapidOCR(**options)
        result, _ = self.ocr(image)
        self.last_ocr_seconds = time.perf_counter() - started
        return '\n'.join(row[1] for row in (result or []))

    def synthesize(self, voice, text, language, progress, cancelled=lambda: False):
        voice = self.emotional_voice(voice, text)
        reference = str(self.store.reference_path(voice))
        if self.last_emotion is not None:
            progress(f"情緒判斷：{self.last_emotion['routed']}（{self.last_emotion['score']:.0%}）")
        from .backends import BackendProcess
        if self.remote is None:
            self.remote = BackendProcess(self.backend)
        emotion = self.last_emotion or {}
        output = self.remote.request(dict(
            text=text, language=language, reference=reference,
            emotion_reference=reference, transcript=voice['transcript'],
            emotion=emotion.get('routed', 'neutral'),
            emotion_score=emotion.get('score', 0),
            streaming_silence_threshold_db=self.streaming_silence_threshold_db,
            streaming_interval_ms=self.streaming_interval_ms), progress, cancelled)
        self.last_metrics = self.remote.last_metrics
        if self.last_metrics is not None and self.last_emotion is not None:
            self.last_metrics['emotion'] = self.last_emotion
        return output

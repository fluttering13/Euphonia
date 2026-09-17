"""JSON-lines worker; library diagnostics go to stderr, never to the protocol."""
import json
import sys
import time
import traceback
import uuid
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
protocol = sys.stdout
sys.stdout = sys.stderr


def trim_streaming_edges(wav, sample_rate, threshold_db=-38.0,
                         leading_seconds=.025, trailing_seconds=.075):
    """Remove generated edge silence while retaining a short natural join."""
    import numpy as np
    audio = np.asarray(wav).reshape(-1)
    if not audio.size:
        return audio
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak < 1e-4:
        return audio
    threshold = peak * (10 ** (float(threshold_db) / 20))
    active = np.flatnonzero(np.abs(audio) >= threshold)
    if not active.size:
        return audio
    start = max(0, int(active[0]) - int(sample_rate * leading_seconds))
    active_end = int(active[-1]) + 1
    wanted_end = active_end + int(sample_rate * trailing_seconds)
    end = min(len(audio), wanted_end)
    trimmed = audio[start:end]
    if wanted_end > len(audio):
        trimmed = np.pad(trimmed, (0, wanted_end - len(audio)))
    return trimmed


def normalize_streaming_audio(wav, sample_rate, threshold_db=-38.0,
                              target_db=-18.0, peak_db=-1.5, fade_ms=8):
    """Match independently generated chunks and suppress boundary clicks."""
    import numpy as np
    audio = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
    if not audio.size:
        return audio, dict(gain_db=0.0, voiced_rms_db=-120.0, peak_db=-120.0)
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak < 1e-5:
        return audio, dict(gain_db=0.0, voiced_rms_db=-120.0, peak_db=-120.0)
    active = np.flatnonzero(np.abs(audio) >= peak * (10 ** (float(threshold_db) / 20)))
    if not active.size:
        return audio, dict(gain_db=0.0, voiced_rms_db=-120.0, peak_db=-120.0)

    # Measure only the speaking portion; configurable edge silence must not
    # change the computed loudness.
    voiced = audio[int(active[0]):int(active[-1]) + 1]
    voiced_rms = float(np.sqrt(np.mean(np.square(voiced, dtype=np.float64))))
    desired_gain = (10 ** (target_db / 20)) / max(voiced_rms, 1e-8)
    peak_limit = (10 ** (peak_db / 20)) / peak
    # Avoid extreme boosts when a model returns an unusually quiet/broken clip.
    gain = min(max(desired_gain, 10 ** (-15 / 20)), 10 ** (15 / 20), peak_limit)
    audio *= gain

    # A short fade on the active region removes non-zero boundary steps without
    # changing the user-selected silent interval.
    fade = min(int(sample_rate * fade_ms / 1000), len(voiced) // 2)
    if fade > 1:
        start, stop = int(active[0]), int(active[-1]) + 1
        audio[start:start + fade] *= np.linspace(0, 1, fade, endpoint=True, dtype=np.float32)
        audio[stop - fade:stop] *= np.linspace(1, 0, fade, endpoint=True, dtype=np.float32)
    final_peak = float(np.max(np.abs(audio)))
    return audio, dict(
        gain_db=float(20 * np.log10(max(gain, 1e-8))),
        voiced_rms_db=float(20 * np.log10(max(voiced_rms, 1e-8))),
        peak_db=float(20 * np.log10(max(final_peak, 1e-8))),
    )


def emit(**message):
    protocol.write(json.dumps(message, ensure_ascii=False) + '\n')
    protocol.flush()


class Runner:
    def __init__(self, backend):
        import torch
        self.torch = torch
        self.backend = backend
        self.voices = {}
        self.model = None
        start = time.perf_counter()
        emit(progress='正在載入模型，首次使用可能需要下載…')
        if backend in ('f5', 'zipvoice'):
            from euphonia.extra_backends import ADAPTERS
            if not torch.cuda.is_available():
                raise RuntimeError('此模型需要可用的 NVIDIA CUDA 環境。')
            self.model = ADAPTERS[backend]()
        elif backend in ('faster', 'qwen_streaming'):
            if not torch.cuda.is_available():
                raise RuntimeError('此模型需要可用的 NVIDIA CUDA 環境。')
            # Editable installs remember the absolute checkout path.  Resolve the
            # bundled source explicitly so moving or renaming Euphonia does not
            # leave this isolated environment pointing at the old directory.
            sys.path.insert(0, str(ROOT / 'experiments/faster-qwen3-tts'))
            from faster_qwen3_tts import FasterQwen3TTS
            self.model = FasterQwen3TTS.from_pretrained(str(ROOT / 'data/model'))
        else:
            raise ValueError(f'Unknown backend: {backend}')
        self.sync()
        self.load_seconds = time.perf_counter() - start

    def sync(self):
        self.torch.cuda.synchronize()

    def audio_chunk(self, text, ref, transcript, language, key):
        import numpy as np
        if self.backend in ('f5', 'zipvoice'):
            return self.model.generate(self.voices[key], text)
        wavs, sr = self.model.generate_voice_clone(
            text=text, language=language, ref_audio=ref, ref_text=transcript,
            xvec_only=False, max_new_tokens=2048)
        return np.asarray(wavs[0]), sr

    def prepare_reference(self, ref, transcript):
        key = (ref, Path(ref).stat().st_mtime_ns, transcript)
        if self.backend in ('f5', 'zipvoice') and key not in self.voices:
            self.voices[key] = self.model.prepare(ref, transcript)
        self.sync()
        return key

    def generate(self, request):
        import numpy as np
        import soundfile as sf
        text = request['text'].strip()
        if not text or len(text) > 5000:
            raise ValueError('請輸入 1–5000 字的文字。')
        if self.backend in ('f5', 'zipvoice') and request['language'] not in ('Chinese', 'English', 'Auto'):
            raise ValueError('此模型目前支援中文與英文，請切換朗讀語言。')
        ref, transcript = request['reference'], request['transcript']
        key = (ref, Path(ref).stat().st_mtime_ns, transcript)
        self.torch.manual_seed(request.get('seed', 100))
        self.sync()
        start = time.perf_counter()
        context = self.torch.no_grad if self.backend == 'f5' else self.torch.inference_mode
        with context():
            if self.backend in ('f5', 'zipvoice'):
                if key not in self.voices:
                    emit(progress='正在建立角色特徵，後續生成會重用…')
                    self.voices[key] = self.model.prepare(ref, transcript)
            self.sync()
            conditioning = time.perf_counter() - start
            emit(progress='正在生成完整語音…')
            # Bound context for long OCR results; keep ordinary dialogue together.
            from euphonia.core import split_text
            chunks = []
            for part in split_text(text, limit=240):
                if chunks and len(chunks[-1]) + len(part) + 1 <= 240:
                    chunks[-1] += ' ' + part
                else:
                    chunks.append(part)
            clips = []
            inference_started = time.perf_counter()
            for part in chunks:
                wav, sr = self.audio_chunk(part, ref, transcript, request['language'], key)
                clips.extend([np.asarray(wav).reshape(-1), np.zeros(int(sr * 0.18), dtype=np.float32)])
            wav = np.concatenate(clips[:-1])
        self.sync()
        inference_seconds = time.perf_counter() - inference_started
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise RuntimeError('模型產生了無效音訊。')
        untrimmed_samples = len(wav)
        loudness = None
        if self.backend == 'qwen_streaming':
            threshold_db = float(request.get('streaming_silence_threshold_db', -38))
            interval_ms = int(request.get('streaming_interval_ms', 75))
            if not -80 <= threshold_db <= -10 or not 0 <= interval_ms <= 1000:
                raise ValueError('Streaming 靜音閾值或片段間隔超出範圍。')
            wav = trim_streaming_edges(
                wav, sr, threshold_db=threshold_db, trailing_seconds=interval_ms / 1000)
            wav, loudness = normalize_streaming_audio(
                wav, sr, threshold_db=threshold_db)
        output = ROOT / 'data/outputs' / f'{self.backend}-{uuid.uuid4().hex}.wav'
        output.parent.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        sf.write(output, wav, sr)
        write_seconds = time.perf_counter() - write_started
        elapsed = time.perf_counter() - start
        metrics = dict(backend=self.backend, total_seconds=elapsed,
                       conditioning_seconds=conditioning, audio_seconds=len(wav) / sr,
                       inference_seconds=inference_seconds, write_seconds=write_seconds,
                       rtf=elapsed / (len(wav) / sr), load_seconds=self.load_seconds,
                       torch=self.torch.__version__, file=str(output))
        if self.backend == 'qwen_streaming':
            metrics['edge_silence_trimmed_seconds'] = (untrimmed_samples - len(wav)) / sr
            metrics['streaming_silence_threshold_db'] = threshold_db
            metrics['streaming_interval_ms'] = interval_ms
            metrics['streaming_loudness'] = loudness
        if self.backend in ('f5', 'zipvoice'):
            metrics['runtime'] = 'pytorch-cuda'
            metrics['steps'] = {'f5': 16, 'zipvoice': 4}[self.backend]
        return str(output), metrics

    def prepare(self, request):
        started = time.perf_counter()
        context = self.torch.no_grad if self.backend == 'f5' else self.torch.inference_mode
        with context():
            self.prepare_reference(request['reference'], request['transcript'])
        return '', dict(backend=self.backend, prepare_seconds=time.perf_counter() - started,
                        load_seconds=self.load_seconds)


def main():
    runner = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            loaded = runner is None
            if runner is None:
                runner = Runner(sys.argv[1])
            if request.get('action') == 'prepare':
                output, metrics = runner.prepare(request)
            else:
                output, metrics = runner.generate(request)
            metrics['load_seconds_this_request'] = runner.load_seconds if loaded else 0.0
            metrics['worker_pid'] = os.getpid()
            emit(result=output, metrics=metrics)
        except Exception as exc:
            traceback.print_exc()
            emit(error=f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()

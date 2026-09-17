"""GPU cloning adapters, imported only inside their isolated worker process."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'data/models'


class F5Backend:
    def __init__(self):
        # Keep the bundled checkout usable after the project directory moves;
        # editable installs otherwise retain their original absolute path.
        sys.path.insert(0, str(ROOT / 'experiments/F5-TTS/src'))
        from f5_tts.api import F5TTS
        self.model = F5TTS(model='F5TTS_v1_Base',
            ckpt_file=str(MODELS / 'f5/F5TTS_v1_Base/model_1250000.safetensors'),
            vocab_file=str(MODELS / 'f5/F5TTS_v1_Base/vocab.txt'),
            vocoder_local_path=str(MODELS / 'vocos-mel-24khz'), device='cuda')

    def prepare(self, reference, transcript):
        import soundfile as sf
        import torch
        # Upstream's file preprocessor clips >12s recordings while retaining the
        # supplied transcript. Preserve their alignment and cache the whole sample.
        audio, sr = sf.read(reference, dtype='float32', always_2d=True)
        text = transcript.strip()
        if not text.endswith(('.', '!', '?', '。', '！', '？')):
            text += '.'
        return (torch.from_numpy(audio.mean(axis=1)).unsqueeze(0), sr), text + ' '

    def generate(self, state, text):
        from f5_tts.infer.utils_infer import infer_batch_process
        wav, sr, _ = next(infer_batch_process(*state, [text], self.model.ema_model, self.model.vocoder,
            device='cuda', nfe_step=16, progress=None))
        return wav, sr


class ZipVoiceBackend:
    def __init__(self):
        sys.path.insert(0, str(ROOT / 'experiments/ZipVoice'))
        import torch
        import safetensors.torch
        from zipvoice.bin.infer_zipvoice import get_vocoder
        from zipvoice.models.zipvoice_distill import ZipVoiceDistill
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        from zipvoice.utils.feature import VocosFbank
        path = MODELS / 'zipvoice/zipvoice_distill'
        self.tokenizer = EmiliaTokenizer(token_file=str(path / 'tokens.txt'))
        config = json.loads((path / 'model.json').read_text(encoding='utf-8'))
        self.model = ZipVoiceDistill(**config['model'], vocab_size=self.tokenizer.vocab_size,
                                    pad_id=self.tokenizer.pad_id)
        safetensors.torch.load_model(self.model, str(path / 'model.safetensors'))
        self.model.eval().to('cuda')
        self.vocoder = get_vocoder(str(MODELS / 'vocos-mel-24khz')).eval().to('cuda')
        self.extractor = VocosFbank()
        self.sr = config['feature']['sampling_rate']

    def prepare(self, reference, transcript):
        import torch
        from zipvoice.utils.infer import load_prompt_wav, rms_norm, remove_silence, add_punctuation
        wav = load_prompt_wav(reference, sampling_rate=self.sr)
        wav = remove_silence(wav, self.sr, only_edge=False, trail_sil=200)
        wav, rms = rms_norm(wav, 0.1)
        features = self.extractor.extract(wav, sampling_rate=self.sr).to('cuda').unsqueeze(0) * 0.1
        tokens = self.tokenizer.texts_to_token_ids([add_punctuation(transcript)])
        lengths = torch.tensor([features.size(1)], device='cuda')
        return features, lengths, tokens, rms

    def generate(self, state, text):
        from zipvoice.utils.infer import add_punctuation
        features, lengths, tokens, rms = state
        pred, pred_lengths, _, _ = self.model.sample(
            tokens=self.tokenizer.texts_to_token_ids([add_punctuation(text)]),
            prompt_tokens=tokens, prompt_features=features, prompt_features_lens=lengths,
            speed=1.0, t_shift=0.5, duration='predict', num_step=4, guidance_scale=3.0)
        mel = pred.permute(0, 2, 1)[:, :, :int(pred_lengths[0])] / 0.1
        wav = self.vocoder.decode(mel).clamp(-1, 1)
        if rms < 0.1:
            wav = wav * rms / 0.1
        return wav.cpu().numpy().reshape(-1), self.sr


ADAPTERS = {'f5': F5Backend, 'zipvoice': ZipVoiceBackend}

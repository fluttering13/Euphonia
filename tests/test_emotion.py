import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from euphonia.core import Engine, VoiceStore
from euphonia.emotion import EmotionClassifier, MODEL_DIR, MULTILINGUAL_MODEL_DIR, model_for_text


class EmotionTests(unittest.TestCase):
    def test_installed_int8_classifier_routes_clear_emotions(self):
        if not (MODEL_DIR / 'model_int8.onnx').is_file():
            self.skipTest('optional emotion model is not installed')
        classifier = EmotionClassifier()
        self.assertEqual(classifier.classify('I am so happy to see you!')['label'], 'joy')
        self.assertEqual(classifier.classify('I feel so alone and sad.')['label'], 'sadness')
        self.assertEqual(classifier.classify('How dare you! Get out of my sight!')['label'], 'anger')

    def test_engine_selects_manifest_reference_and_matching_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceStore(Path(tmp) / 'voices')
            source = Path(tmp) / 'source.wav'
            sf.write(source, np.ones(48000, dtype=np.float32) * .1, 16000)
            voice = store.save('Rhiannon', 'Neutral words.', source)
            emotion_dir = store.audio_path(voice['id']).parent / 'emotions' / 'joy'
            emotion_dir.mkdir(parents=True)
            sf.write(emotion_dir / 'reference.wav', np.ones(48000, dtype=np.float32) * .1, 16000)
            manifest = {'emotions': {'joy': {'reference': 'emotions/joy/reference.wav',
                                             'transcript': 'Happy words.'}}}
            (store.audio_path(voice['id']).parent / 'emotion_manifest.json').write_text(
                json.dumps(manifest), encoding='utf-8')
            engine = Engine(store)
            classifier = Mock()
            classifier.classify.return_value = {
                'label': 'joy', 'score': .9, 'seconds': .01, 'scores': {'joy': .9}}
            engine.emotion_classifiers[MODEL_DIR] = classifier
            routed = engine.emotional_voice(voice, 'Wonderful!')
            self.assertEqual(routed['emotion'], 'joy')
            self.assertEqual(routed['transcript'], 'Happy words.')
            self.assertTrue(store.reference_path(routed).samefile(emotion_dir / 'reference.wav'))

    def test_nearest_score_vector_wins_within_same_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceStore(Path(tmp) / 'voices')
            source = Path(tmp) / 'source.wav'
            sf.write(source, np.ones(48000, dtype=np.float32) * .1, 16000)
            voice = store.save('Rhiannon', 'Base.', source)
            root = store.audio_path(voice['id']).parent
            samples = []
            for sample_id, anger in [('soft', .55), ('strong', .95)]:
                folder = root / 'emotions' / 'samples' / sample_id
                folder.mkdir(parents=True)
                sf.write(folder / 'reference.wav', np.ones(48000) * .1, 16000)
                samples.append(dict(id=sample_id, emotion='anger', label=sample_id,
                    reference=f'emotions/samples/{sample_id}/reference.wav', transcript=sample_id,
                    score=anger, scores={'anger': anger, 'fear': 1 - anger}))
            (root / 'emotion_manifest.json').write_text(
                json.dumps({'version': 2, 'samples': samples}), encoding='utf-8')
            engine = Engine(store)
            classifier = Mock()
            classifier.classify.return_value = {'label': 'anger', 'score': .9, 'seconds': .01,
                                                 'scores': {'anger': .9, 'fear': .1}}
            engine.emotion_classifiers[MODEL_DIR] = classifier
            routed = engine.emotional_voice(voice, 'I am furious!')
            self.assertEqual(routed['emotion_sample']['id'], 'strong')
            self.assertEqual(engine.last_emotion['sample_label'], 'strong')

    def test_cjk_text_uses_multilingual_model(self):
        self.assertEqual(model_for_text('我真的很害怕！'), MULTILINGUAL_MODEL_DIR)
        self.assertEqual(model_for_text('I am afraid!'), MODEL_DIR)

    def test_emotion_backends_receive_identity_and_emotion_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VoiceStore(Path(tmp) / 'voices')
            source = Path(tmp) / 'source.wav'
            emotion = Path(tmp) / 'joy.wav'
            sf.write(source, np.ones(48000, dtype=np.float32) * .1, 16000)
            sf.write(emotion, np.ones(48000, dtype=np.float32) * .2, 16000)
            voice = store.save('Voice', 'Neutral.', source, [dict(
                audio_path=str(emotion), transcript='Wonderful!', emotion='joy', score=.9,
                scores={'joy': .9, 'neutral': .1})])
            engine = Engine(store)
            classifier = Mock()
            classifier.classify.return_value = {
                'label': 'joy', 'score': .9, 'seconds': .01,
                'scores': {'joy': .9, 'neutral': .1}}
            engine.emotion_classifiers[MODEL_DIR] = classifier
            client = Mock()
            client.request.return_value = 'output.wav'
            client.last_metrics = None
            with patch('euphonia.backends.BackendProcess', return_value=client):
                engine.set_backend('f5')
                engine.synthesize(voice, 'Wonderful!', 'English', lambda _: None)
            request = client.request.call_args.args[0]
            self.assertIn('emotions', request['reference'])
            self.assertEqual(request['reference'], request['emotion_reference'])
            self.assertEqual(request['emotion'], 'joy')
            self.assertEqual(request['emotion_score'], .9)


if __name__ == '__main__':
    unittest.main()

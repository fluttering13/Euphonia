import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from euphonia.core import VoiceStore, split_text


class StoreTests(unittest.TestCase):
    def test_voice_survives_source_removal_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.wav'
            sf.write(source, np.sin(np.arange(64000) * 0.03) * 0.2, 16000)
            store = VoiceStore(root / 'voices')
            voice = store.save('角色一', '測試參考文字', source)
            source.unlink()
            reopened = VoiceStore(root / 'voices')
            self.assertEqual(reopened.list(), [voice])
            self.assertTrue(reopened.audio_path(voice['id']).exists())
            reopened.delete(voice['id'])
            self.assertEqual(reopened.list(), [])

    def test_invalid_audio_does_not_create_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'silence.wav'
            sf.write(source, np.zeros(64000), 16000)
            store = VoiceStore(Path(tmp) / 'voices')
            with self.assertRaises(ValueError):
                store.save('silent', 'hello', source)
            self.assertEqual(store.list(), [])
            with self.assertRaises(ValueError):
                store.audio_path('../../outside')

    def test_long_text_preserves_content(self):
        text = '這是一段很長的文字，' * 50 + '結束！Hello world.'
        parts = split_text(text)
        self.assertTrue(all(len(p) <= 160 for p in parts))
        self.assertEqual(''.join(parts).replace(' ', ''), text.replace(' ', ''))

    def test_voice_emotion_templates_are_grouped_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base.wav'
            angry_a = root / 'angry-a.wav'
            angry_b = root / 'angry-b.wav'
            sf.write(base, np.sin(np.arange(64000) * .03) * .2, 16000)
            sf.write(angry_a, np.sin(np.arange(16000) * .04) * .2, 16000)
            sf.write(angry_b, np.sin(np.arange(24000) * .05) * .2, 16000)
            store = VoiceStore(root / 'voices')
            voice = store.save('角色', 'Base transcript.', base, [
                dict(audio_path=str(angry_a), transcript='Get out!', emotion='anger', score=.9),
                dict(audio_path=str(angry_b), transcript='How dare you!', emotion='anger', score=.8),
            ])
            manifest_path = store.audio_path(voice['id']).parent / 'emotion_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            angry = manifest['samples']
            self.assertEqual([item['transcript'] for item in angry], ['Get out!', 'How dare you!'])
            self.assertAlmostEqual(angry[0]['seconds'], 1.0, places=2)
            self.assertAlmostEqual(angry[1]['seconds'], 1.5, places=2)
            self.assertTrue(store.reference_path(dict(voice, reference=angry[0]['reference'])).is_file())
            store.delete(voice['id'])
            self.assertFalse(manifest_path.parent.exists())

    def test_edit_voice_preserves_id_and_existing_emotion_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base.wav'
            joy = root / 'joy.wav'
            sf.write(base, np.sin(np.arange(64000) * .03) * .2, 16000)
            sf.write(joy, np.sin(np.arange(24000) * .04) * .2, 16000)
            store = VoiceStore(root / 'voices')
            voice = store.save('Old name', 'Base.', base, [dict(
                audio_path=str(joy), transcript='Wonderful!', emotion='joy', score=.9,
                scores={'joy': .9, 'neutral': .1})])
            templates = store.emotion_templates(voice)
            updated = store.update(voice['id'], 'New name', 'Updated base.',
                                   str(store.audio_path(voice['id'])), templates)
            self.assertEqual(updated['id'], voice['id'])
            self.assertEqual(updated['name'], 'New name')
            self.assertEqual(len(store.emotion_templates(updated)), 1)
            self.assertTrue(any((store.audio_path(voice['id']).parent / 'backups').iterdir()))

    def test_edit_voice_can_keep_existing_reference_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.wav'
            sf.write(source, np.sin(np.arange(64000) * .03) * .2, 16000)
            store = VoiceStore(root / 'voices')
            voice = store.save('Old name', 'Original transcript.', source)

            updated = store.update(voice['id'], 'New name', 'Updated transcript.')

            self.assertEqual(updated['id'], voice['id'])
            self.assertEqual(sf.info(store.audio_path(voice['id'])).frames, 64000)
            self.assertEqual(VoiceStore(root / 'voices').list()[0]['name'], 'New name')


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from euphonia.extra_backends import F5Backend


class ReferenceTests(unittest.TestCase):
    def test_f5_preserves_long_reference_and_matching_transcript(self):
        # No model weights needed: this checks the >12s clipping regression.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'reference.wav'
            samples = np.ones(16000 * 13, dtype=np.float32) * 0.05
            sf.write(path, samples, 16000)
            state = F5Backend.__new__(F5Backend).prepare(
                str(path), 'First sentence. Last sentence.')
            (audio, sr), text = state
            self.assertEqual(audio.shape[-1], len(samples))
            self.assertEqual(sr, 16000)
            self.assertEqual(text.strip(), 'First sentence. Last sentence.')


if __name__ == '__main__':
    unittest.main()

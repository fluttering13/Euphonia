import sys
import subprocess
import threading
import unittest
from unittest.mock import patch

from euphonia.backends import BackendProcess
from euphonia.core import Engine
from scripts.backend_worker import normalize_streaming_audio, trim_streaming_edges


class BackendTests(unittest.TestCase):
    def test_streaming_edge_trim_keeps_short_natural_pause(self):
        import numpy as np
        sr = 1000
        wav = np.concatenate([np.zeros(200), np.full(300, .2), np.zeros(400)])
        trimmed = trim_streaming_edges(wav, sr)
        self.assertEqual(len(trimmed), 25 + 300 + 75)
        self.assertTrue(np.allclose(trimmed[25:325], .2))

    def test_streaming_threshold_and_interval_change_edge_detection(self):
        import numpy as np
        sr = 1000
        wav = np.concatenate([np.zeros(100), np.full(200, .2),
                              np.full(150, .005), np.zeros(200)])
        aggressive = trim_streaming_edges(wav, sr, threshold_db=-20, trailing_seconds=.04)
        sensitive = trim_streaming_edges(wav, sr, threshold_db=-60, trailing_seconds=.04)
        self.assertEqual(len(aggressive), 25 + 200 + 40)
        self.assertGreater(len(sensitive), len(aggressive))
        no_gap = trim_streaming_edges(
            np.concatenate([np.zeros(50), np.full(100, .2)]), sr,
            threshold_db=-38, trailing_seconds=0)
        self.assertEqual(len(no_gap), 25 + 100)

    def test_streaming_chunks_are_loudness_matched_and_peak_limited(self):
        import numpy as np
        sr = 24000
        phase = np.arange(sr) * (2 * np.pi * 220 / sr)
        quiet = np.concatenate([np.zeros(600), np.sin(phase).astype(np.float32) * .04,
                                np.zeros(1200)])
        loud = np.concatenate([np.zeros(600), np.sin(phase).astype(np.float32) * .32,
                               np.zeros(1200)])
        quiet_out, quiet_metrics = normalize_streaming_audio(quiet, sr)
        loud_out, loud_metrics = normalize_streaming_audio(loud, sr)
        def voiced_rms(x):
            active = x[np.abs(x) > .005]
            return 20 * np.log10(np.sqrt(np.mean(active * active)))
        self.assertLess(abs(voiced_rms(quiet_out) - voiced_rms(loud_out)), .5)
        self.assertLessEqual(np.max(np.abs(quiet_out)), 10 ** (-1.5 / 20) + 1e-6)
        self.assertLessEqual(np.max(np.abs(loud_out)), 10 ** (-1.5 / 20) + 1e-6)
        self.assertGreater(quiet_metrics['gain_db'], 0)
        self.assertLess(loud_metrics['gain_db'], 0)

    def test_streaming_active_edges_are_faded(self):
        import numpy as np
        audio = np.concatenate([np.zeros(25), np.full(200, .2), np.zeros(75)])
        output, _ = normalize_streaming_audio(audio, 1000, fade_ms=8)
        self.assertEqual(output[25], 0)
        self.assertEqual(output[224], 0)
        self.assertGreater(output[35], 0)

    def test_protocol_and_reuse(self):
        real_popen = subprocess.Popen
        code = ('import sys,json\n'
                'for line in sys.stdin:\n'
                ' r=json.loads(line)\n'
                ' print(json.dumps({"progress":"working"}),flush=True)\n'
                ' print(json.dumps({"result":r["text"],"metrics":{"total_seconds":0.1}}),flush=True)\n')
        def popen(args, **kwargs):
            return real_popen([sys.executable, '-u', '-c', code], **kwargs)
        client = BackendProcess('faster')
        try:
            with patch('euphonia.backends.Path.is_file', return_value=True), \
                    patch('euphonia.backends.subprocess.Popen', side_effect=popen):
                progress = []
                self.assertEqual(client.request({'text': 'hello'}, progress.append), 'hello')
                pid = client.process.pid
                self.assertEqual(client.request({'text': 'second'}, progress.append), 'second')
                self.assertEqual(client.process.pid, pid)
                self.assertEqual(progress, ['working', 'working'])
        finally:
            client.close()
        self.assertIsNone(client.process)

    def test_cancellation_keeps_model_and_drains_stale_reply(self):
        real_popen = subprocess.Popen
        def popen(args, **kwargs):
            code = 'import sys,json,time\nfor line in sys.stdin:\n r=json.loads(line); time.sleep(0.3); print(json.dumps({"result":r["text"]}),flush=True)'
            return real_popen([sys.executable, '-u', '-c', code], **kwargs)
        client = BackendProcess('faster')
        stop = threading.Event()
        timer = threading.Timer(0.2, stop.set)
        try:
            with patch('euphonia.backends.Path.is_file', return_value=True), \
                    patch('euphonia.backends.subprocess.Popen', side_effect=popen):
                timer.start()
                self.assertIsNone(client.request({'text': 'old'}, lambda _: None, stop.is_set))
                pid = client.process.pid
                self.assertEqual(client.request({'text': 'new'}, lambda _: None), 'new')
                self.assertEqual(client.process.pid, pid)
        finally:
            timer.cancel()
            client.close()

    def test_switch_releases_old_worker(self):
        from unittest.mock import Mock
        engine = Engine(None)
        engine.set_backend('faster')
        previous = Mock()
        engine.remote = previous
        engine.set_backend('f5')
        previous.close.assert_called_once()
        self.assertIsNone(engine.remote)
        self.assertEqual(engine.backend, 'f5')
        with self.assertRaises(ValueError):
            engine.set_backend('unknown')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows process-tree behavior')
    def test_close_terminates_the_entire_worker_tree(self):
        from unittest.mock import Mock
        client = BackendProcess('faster')
        process = Mock(pid=43210, stdin=None)
        process.poll.side_effect = [None, 1]
        client.process = process
        with patch('euphonia.backends.subprocess.run') as run:
            client.close()
        command = run.call_args.args[0]
        self.assertEqual(command, ['taskkill', '/PID', '43210', '/T', '/F'])
        self.assertIsNone(client.process)

"""Real desktop crop -> OCR -> persistent TTS -> first decoded playback buffer."""
import json
import argparse
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'windows'
from PySide6.QtCore import QRect, QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtMultimedia import QAudioBufferOutput
from PySide6.QtWidgets import QApplication, QLabel
from euphonia.app import Window
from euphonia.capture_frame import CaptureFrame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['faster', 'qwen_streaming', 'zipvoice', 'f5'])
    parser.add_argument('--output', default='pipeline-profile.json')
    parser.add_argument('--fast-ocr', action='store_true', help='Experiment with bounded OCR image size and CPU threads')
    parser.add_argument('--floating', action='store_true', help='Measure the actual floating button, including hide delay')
    args = parser.parse_args()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    report = {'notes': ['Real desktop screenshot of a controlled English text panel.',
                        'No manual selection time or 300ms pre-selection overlay delay included.',
                        'Playback is first decoded PCM buffer, not measured speaker latency.',
                        'Concurrent game load not controlled. Existing user app may remain loaded.'], 'runs': []}
    panel = QLabel('The morning sky looks beautiful.\nShall we explore the forest together?')
    panel.setWindowFlag(Qt.WindowStaysOnTopHint)
    panel.setStyleSheet('background: white; color: black; font-family: Arial; font-size: 36px; padding: 20px')
    panel.setGeometry(100, 100, 900, 180)
    panel.show()
    with tempfile.TemporaryDirectory() as temporary:
        settings = QSettings(str(Path(temporary) / 'settings.ini'), QSettings.IniFormat)
        with patch('euphonia.app.QSettings', return_value=settings):
            window = Window(auto_preload=False)
        window.screenshot_hotkey.stop()
        if args.floating:
            window.capture_frame = CaptureFrame(panel.screen(), QRect(100, 100, 900, 180), window)
            window.capture_frame.requested.connect(window.repeat_capture)
            window.capture_frame.show()
            report['notes'][1] = 'Actual floating button flow; includes 150ms hide delay. screenshot_seconds includes this delay.'
        if args.fast_ocr:
            from rapidocr_onnxruntime import RapidOCR
            window.engine.ocr = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1,
                                         det_limit_type='max', det_limit_side_len=960)
            report['notes'].append('Experimental OCR: 4 intra-op threads, 1 inter-op thread, max side 960.')
        window.audio.setVolume(0)
        recognize = window.recognize
        def record_input(crop):
            crop.save(str(ROOT / 'data/pipeline-input.png'))
            recognize(crop)
        window.recognize = record_input
        errors, buffers = [], []
        window.failed = errors.append
        window.notify_error = errors.append
        pcm = QAudioBufferOutput(window)
        window.player.setAudioBufferOutput(pcm)
        pcm.audioBufferReceived.connect(lambda b: buffers.append(time.perf_counter()) if b.isValid() else None)
        def wait(condition, timeout=60):
            end = time.monotonic() + timeout
            while not condition() and not errors and time.monotonic() < end:
                app.processEvents()
                time.sleep(0.001)
            if errors or not condition():
                raise RuntimeError(str(errors) or f'Pipeline timed out: {window.status.text()}')
        def save():
            (ROOT / 'data' / args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        try:
            for name in args.models:
                backend = name
                window.model_selector.setCurrentIndex(window.model_selector.findData(backend))
                window.language.setCurrentIndex(window.language.findData('English'))
                prepare_start = time.perf_counter()
                window.preload_model()
                wait(lambda: window.thread is None)
                preload_seconds = time.perf_counter() - prepare_start
                pid = window.engine.remote.process.pid
                for index in range(3):
                    print(f'{backend}: capture {index + 1}/3', flush=True)
                    window.player.stop()
                    buffers.clear()
                    window.output = None
                    panel.raise_()
                    panel.activateWindow()
                    settle_until = time.monotonic() + 0.2
                    wait(lambda: time.monotonic() >= settle_until)
                    if args.floating:
                        window.capture_frame.raise_()
                        settle_until = time.monotonic() + 0.05
                        wait(lambda: time.monotonic() >= settle_until)
                    begin = time.perf_counter()
                    if args.floating:
                        QTest.mouseClick(window.capture_frame.read_button, Qt.LeftButton)
                    else:
                        crop = panel.screen().grabWindow(0, 100, 100, 900, 180)
                        screenshot_seconds = time.perf_counter() - begin
                        if index == 0:
                            crop.save(str(ROOT / 'data/pipeline-input.png'))
                        window.recognize(crop)
                    wait(lambda: window.thread is None and window.output is not None and bool(buffers))
                    if args.floating:
                        screenshot_seconds = window.repeat_capture_seconds
                    row = json.loads((ROOT / 'data/last-pipeline.json').read_text(encoding='utf-8'))
                    row.update(preset='fast' if fast else 'default', index=index,
                               screenshot_seconds=screenshot_seconds,
                               screenshot_to_pcm_seconds=buffers[0] - begin,
                               preload_seconds=preload_seconds, process_pid=pid,
                               recognized=window.editor.toPlainText())
                    row['playback_start_seconds'] = row['screenshot_to_pcm_seconds'] - screenshot_seconds - row['to_play_request_seconds']
                    expected = 'The morning sky looks beautiful. Shall we explore the forest together?'
                    if ' '.join(row['recognized'].split()) != expected:
                        raise RuntimeError(f"Test panel was not captured/OCR'd correctly: {row['recognized']!r}")
                    assert window.engine.remote.process.pid == pid
                    assert row['tts_details']['load_seconds_this_request'] == 0
                    report['runs'].append(row)
                    save()
                    print(json.dumps(row), flush=True)
            # Actual inference cancellation must preserve the model process.
            window.editor.setPlainText('This request is cancelled while the model remains loaded.')
            window.speak()
            QTimer.singleShot(50, window.stop)
            wait(lambda: window.thread is None)
            assert window.engine.remote.process.pid == pid
            window.editor.setPlainText('The next request uses the same loaded model.')
            window.output = None
            window.speak()
            wait(lambda: window.thread is None and window.output is not None)
            assert window.engine.remote.process.pid == pid
            report['cancel_then_infer_same_process'] = True
            report['medians'] = {}
            for key in dict.fromkeys(r['backend'] + '-' + r['preset'] for r in report['runs']):
                rows = [r for r in report['runs'] if r['backend'] + '-' + r['preset'] == key]
                report['medians'][key] = {name: statistics.median(r[name] for r in rows) for name in
                    ('screenshot_seconds', 'ocr_seconds', 'tts_seconds', 'screenshot_to_pcm_seconds', 'playback_start_seconds')}
            save()
        finally:
            errors.clear()
            window.request_exit()
            wait(lambda: window.thread is None)
            window.engine.close()
            panel.close()


if __name__ == '__main__':
    main()

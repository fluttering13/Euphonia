"""Windows acceptance: external foreground window, actual F11, OCR, GPU TTS, PCM.

Uses isolated settings and a temporary test panel; does not change user voices.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'windows'
from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtMultimedia import QAudioBufferOutput
from PySide6.QtWidgets import QApplication, QLabel, QWidget
from euphonia.app import Window
from euphonia.audiobook import defaults
from euphonia.audiobook_monitor import capture_regions
from euphonia.hotkeys import activate_window, foreground_window


def spin(app, condition, timeout=30):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    if not condition():
        raise AssertionError('Native audiobook test timed out')


def send_f11(up=False):
    class Mouse(ctypes.Structure):
        _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('data', wintypes.DWORD),
                    ('flags', wintypes.DWORD), ('time', wintypes.DWORD), ('extra', ctypes.c_size_t)]
    class Keyboard(ctypes.Structure):
        _fields_ = [('vk', wintypes.WORD), ('scan', wintypes.WORD), ('flags', wintypes.DWORD),
                    ('time', wintypes.DWORD), ('extra', ctypes.c_size_t)]
    class Payload(ctypes.Union):
        _fields_ = [('mouse', Mouse), ('keyboard', Keyboard)]
    class Input(ctypes.Structure):
        _fields_ = [('kind', wintypes.DWORD), ('payload', Payload)]
    entry = Input(kind=1, payload=Payload(keyboard=Keyboard(vk=0x7A, flags=2 if up else 0)))
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    assert api.SendInput(1, ctypes.byref(entry), ctypes.sizeof(Input)) == 1


def helper(directory):
    app = QApplication([])
    panel = QWidget()
    panel.setWindowTitle('Euphonia audiobook acceptance panel')
    panel.setWindowFlag(Qt.WindowStaysOnTopHint)
    panel.setStyleSheet('background:white;color:black;font-family:Arial;font-size:32px;')
    panel.setGeometry(100, 100, 980, 230)
    name, text = QLabel('Alice', panel), QLabel('The morning sky looks beautiful.', panel)
    name.setGeometry(20, 10, 300, 65)
    text.setGeometry(20, 90, 940, 100)
    panel.show()
    panel.raise_()
    panel.activateWindow()
    directory = Path(directory)
    def publish():
        def region(widget):
            origin = widget.mapToGlobal(QPoint(0, 0))
            return dict(x=origin.x(), y=origin.y(), width=widget.width(), height=widget.height())
        (directory / 'panel.json').write_text(json.dumps(dict(hwnd=int(panel.winId()),
            character_name=region(name), dialogue_text=region(text))), encoding='utf-8')
    def update():
        path = directory / 'scene.json'
        if path.exists():
            try:
                scene = json.loads(path.read_text(encoding='utf-8'))
                name.setText(scene['name'])
                text.setText(scene['text'])
            except (ValueError, OSError):
                pass
    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    QTimer.singleShot(300, publish)
    QTimer.singleShot(180000, app.quit)
    app.exec()


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    previous = foreground_window()
    cursor = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    window = child = None
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            settings = QSettings(str(Path(tmp) / 'settings.ini'), QSettings.IniFormat)
            settings.setValue('model', 'zipvoice')
            settings.setValue('language', 'English')
            with patch('euphonia.app.QSettings', return_value=settings):
                window = Window(auto_preload=False)
            window.audio.setVolume(0)
            voices = window.store.list()
            assert voices, 'A stored reference voice is required'
            window.engine.prepare(voices[0], 'English', lambda msg: print(msg, flush=True))
            child = subprocess.Popen([sys.executable, __file__, '--helper', tmp],
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            path = Path(tmp) / 'panel.json'
            spin(app, path.exists)
            panel = json.loads(path.read_text(encoding='utf-8'))
            activate_window(panel['hwnd'])
            # Windows grants foreground focus after a click in our own test panel.
            r = panel['character_name']
            ctypes.windll.user32.SetCursorPos(r['x'] + 20, r['y'] + 20)
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            spin(app, lambda: foreground_window() == panel['hwnd'])
            config = defaults()
            config.update(enabled=True, stable_duration_ms=500, ocr_interval_ms=150,
                          ocr_regions={k: panel[k] for k in ('character_name', 'dialogue_text')})
            config['characters'] = [dict(id=name.lower(), name=name, voice_id=voices[0]['id'], enabled=True)
                                    for name in ('Alice', 'Bob')]
            monitor = window.audiobook
            monitor.config = config
            baseline = capture_regions(config['ocr_regions'], QApplication.screens())
            outputs, buffers = [], []
            play = window.play
            def played(path):
                outputs.append(dict(text=window.editor.toPlainText(), path=path))
                play(path)
            window.play = played
            pcm = QAudioBufferOutput(window)
            window.player.setAudioBufferOutput(pcm)
            pcm.audioBufferReceived.connect(lambda b: buffers.append(True) if b.isValid() else None)
            send_f11()
            send_f11(up=True)
            spin(app, lambda: monitor.enabled)
            assert len(monitor.overlays.boxes) == 2
            assert all(box.isVisible() for box in monitor.overlays.boxes)
            boxes = monitor.overlays.boxes
            api = ctypes.WinDLL('user32')
            api.WindowFromPoint.argtypes = [wintypes.POINT]
            api.WindowFromPoint.restype = wintypes.HWND
            for box in boxes:
                point = box.geometry().topLeft() + QPoint(5, 5)
                assert api.WindowFromPoint(wintypes.POINT(point.x(), point.y())) != int(box.winId())
            with monitor.overlays.clean_capture():
                current = capture_regions(config['ocr_regions'], QApplication.screens())
            assert all(np.array_equal(baseline[k], current[k]) for k in baseline), 'Bboxes contaminated OCR pixels'
            worker = monitor.thread
            spin(app, lambda: outputs and buffers)
            assert outputs[0]['text'] == 'The morning sky looks beautiful.', outputs
            assert foreground_window() == panel['hwnd'], 'Monitoring stole foreground focus'
            until = time.monotonic() + 10
            spin(app, lambda: time.monotonic() >= until, timeout=12)
            assert len(outputs) == 1, 'Stationary dialogue repeated'
            (Path(tmp) / 'scene.json').write_text(json.dumps(dict(name='Unknown NPC', text='Do not speak this line.')))
            until = time.monotonic() + 2
            spin(app, lambda: time.monotonic() >= until)
            assert len(outputs) == 1, 'Unknown character spoke'
            (Path(tmp) / 'scene.json').write_text(json.dumps(dict(name='Bob', text='Shall we explore the forest together?')))
            spin(app, lambda: len(outputs) == 2)
            assert outputs[-1]['text'] == 'Shall we explore the forest together?', outputs
            assert monitor.thread is worker
            send_f11()
            send_f11(up=True)
            spin(app, lambda: not monitor.enabled and monitor.thread is None and window.thread is None)
            assert not monitor.timer.isActive()
            assert not monitor.overlays.boxes
            assert not monitor.queue
            # Restart and immediately stop through the real global listener.
            send_f11()
            send_f11(up=True)
            spin(app, lambda: monitor.enabled)
            send_f11()
            send_f11(up=True)
            spin(app, lambda: not monitor.enabled and monitor.thread is None)
            report = dict(passed=True, foreground_preserved=True, stationary_seconds=10,
                          unknown_silent=True, f11_restart=True, single_worker=True,
                          real_ocr=True, real_tts='zipvoice', pcm_received=True, outputs=outputs)
            report.update(two_visible_bboxes=True, click_through=True, clean_ocr_pixels=True,
                          bboxes_hidden_on_stop=True)
            window.show()
            app.processEvents()
            window.grab().save(str(ROOT / 'data/audiobook-main.png'))
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            if window:
                window.stop()
                spin(app, lambda: window.thread is None)
                window.request_exit()
            if child:
                child.terminate()
                child.wait(timeout=5)
            activate_window(previous)
            ctypes.windll.user32.SetCursorPos(cursor.x, cursor.y)
    (ROOT / 'data/audiobook-native-test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--helper':
        helper(sys.argv[2])
    else:
        main()

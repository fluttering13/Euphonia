"""Windows integration: another process owns focus, F12 captures, jobs run hidden.

Stubs OCR and model inference to avoid depending on unrelated foreground pixels;
checks actual capture overlay, thread handoff, and silent Qt playback. The separate
test_sample.py exercises real OCR and model inference.
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
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_QPA_PLATFORM'] = 'windows'
from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel
from euphonia.app import Window
from euphonia.core import DATA
from euphonia.hotkeys import activate_window, foreground_window


def wait_for(app, condition, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
        if condition():
            return
    raise AssertionError('Timed out waiting for background integration step')


def send_f12(up=False):
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
    entry = Input(kind=1, payload=Payload(keyboard=Keyboard(vk=0x7B, flags=2 if up else 0)))
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    assert api.SendInput(1, ctypes.byref(entry), ctypes.sizeof(Input)) == 1


def helper(path):
    app = QApplication([])
    label = QLabel('Euphonia background test')
    label.setWindowFlag(Qt.WindowStaysOnTopHint)
    label.setWindowTitle('Euphonia test foreground process')
    label.setStyleSheet('background:white;color:black;font-size:36px;padding:20px;')
    screen = app.primaryScreen().geometry()
    label.setGeometry(screen.x() + 80, screen.y() + 80, 850, 220)
    label.show()
    label.raise_()
    label.activateWindow()
    def publish():
        origin = label.mapToGlobal(QPoint(0, 0))
        Path(path).write_text(json.dumps(dict(hwnd=int(label.winId()), x=origin.x(), y=origin.y(),
                                             width=label.width(), height=label.height())), encoding='utf-8')
    QTimer.singleShot(300, publish)
    QTimer.singleShot(60000, app.quit)
    app.exec()


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    original = foreground_window()
    cursor = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    window = None
    child = None
    with tempfile.TemporaryDirectory() as temp:
        try:
            settings = QSettings(temp + '/settings.ini', QSettings.IniFormat)
            with patch('euphonia.app.QSettings', return_value=settings):
                window = Window(auto_preload=False)
            assert window.tray is not None, 'Windows tray is unavailable'
            ready, errors = [], []
            window.screenshot_hotkey.ready.connect(lambda: ready.append(True))
            window.screenshot_hotkey.failed.connect(errors.append)
            wait_for(app, lambda: ready or errors)
            assert not errors, errors
            window.show()
            app.processEvents()
            window.hide_to_tray()
            assert not window.isVisible() and window.screenshot_hotkey.listener.isRunning()
            window.auto_read.setChecked(True)
            window.screens.setCurrentIndex(0)
            audio = DATA / 'sample-tests/rhiannon/english.wav'
            assert audio.is_file() and window.voices.currentData()
            window.engine.synthesize = Mock(return_value=str(audio))
            window.engine.recognize = Mock(return_value='Euphonia background test')
            window.audio.setVolume(0)
            states = []
            window.player.playbackStateChanged.connect(states.append)
            info_path = Path(temp) / 'foreground.json'
            child = subprocess.Popen([sys.executable, __file__, '--foreground', str(info_path)],
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            wait_for(app, info_path.exists)
            info = json.loads(info_path.read_text(encoding='utf-8'))
            activate_window(info['hwnd'])
            # A real click into our own test window grants normal foreground focus.
            ctypes.windll.user32.SetCursorPos(info['x'] + 30, info['y'] + 30)
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            wait_for(app, lambda: foreground_window() == info['hwnd'])
            send_f12()
            try:
                wait_for(app, lambda: window.capture is not None)
            finally:
                send_f12(up=True)
            overlay = window.capture
            print('Focus before selection:', foreground_window(), 'saved:', window.previous_foreground,
                  'expected:', info['hwnd'], 'overlay:', int(overlay.winId()), flush=True)
            origin = overlay.geometry().topLeft()
            first = QPoint(info['x'] + 5, info['y'] + 5) - origin
            last = first + QPoint(info['width'] - 10, info['height'] - 10)
            QTest.mousePress(overlay, Qt.LeftButton, pos=first)
            QTest.mouseMove(overlay, last)
            QTest.mouseRelease(overlay, Qt.LeftButton, pos=last)
            assert not window.isVisible(), 'Capture restored main window instead of staying in background'
            wait_for(app, lambda: window.thread is None and QMediaPlayer.PlayingState in states, 30)
            assert 'Euphonia' in window.editor.toPlainText(), window.editor.toPlainText()
            assert foreground_window() == info['hwnd'], f'Focus was not returned: actual={foreground_window()}, expected={info["hwnd"]}, saved={window.previous_foreground}'
            window.engine.synthesize.assert_called_once()
            assert not window.isVisible()
            report = dict(result='passed', native_f12=True, foreground_other_process=True,
                          explicit_background_to_tray=True, actual_capture_overlay=True, ocr='stubbed',
                          model_inference='stubbed with existing sample WAV', actual_silent_playback=True,
                          stayed_hidden=True, foreground_restored=True)
            (DATA / 'background-test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            if window:
                window.stop()
                wait_for(app, lambda: window.thread is None, 30)
                window.request_exit()
                assert not window.screenshot_hotkey.listener.isRunning()
            if child:
                child.terminate()
                child.wait(timeout=5)
            activate_window(original)
            ctypes.windll.user32.SetCursorPos(cursor.x, cursor.y)


if __name__ == '__main__':
    if '--foreground' in sys.argv:
        helper(sys.argv[-1])
    else:
        main()

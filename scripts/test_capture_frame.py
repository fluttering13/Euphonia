"""Native Windows test: persistent frame, click-through hole and fresh recapture."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_QPA_PLATFORM'] = 'windows'
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from euphonia.app import Window
from euphonia.core import DATA


def pump(ms=250):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


app = QApplication([])
app.setQuitOnLastWindowClosed(False)
with tempfile.TemporaryDirectory() as temp:
    with patch('euphonia.app.QSettings', return_value=QSettings(temp + '/settings.ini', QSettings.IniFormat)):
        window = Window(auto_preload=False)
    window.screenshot_hotkey.stop()
    panel = QWidget()
    panel.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    screen = app.primaryScreen()
    origin = screen.geometry().topLeft()
    panel.setGeometry(origin.x() + 100, origin.y() + 100, 600, 260)
    panel.setStyleSheet('background: #ff0000;')
    panel.show()
    outputs = []
    window.recognize = outputs.append
    try:
        pump()
        window.start_capture()
        pump(600)
        capture = window.capture
        assert capture is not None
        QTest.mousePress(capture, Qt.LeftButton, pos=QPoint(150, 160))
        QTest.mouseMove(capture, QPoint(550, 280))
        QTest.mouseRelease(capture, Qt.LeftButton, pos=QPoint(550, 280))
        pump()
        frame = window.capture_frame
        assert frame is not None and frame.isVisible()
        assert frame.windowFlags() & Qt.WindowStaysOnTopHint
        assert len(outputs) == 1
        assert outputs[0].toImage().pixelColor(30, 30).name() == '#ff0000'
        center = frame.region.center()
        assert not frame.mask().contains(frame.mapFromGlobal(center))
        api = ctypes.WinDLL('user32')
        api.WindowFromPoint.argtypes = [wintypes.POINT]
        api.WindowFromPoint.restype = wintypes.HWND
        assert api.WindowFromPoint(wintypes.POINT(center.x(), center.y())) == int(panel.winId())
        panel.setStyleSheet('background: #0000ff;')
        pump()
        QTest.mouseClick(frame.read_button, Qt.LeftButton)
        assert not frame.isVisible()
        pump(400)
        assert len(outputs) == 2 and frame.isVisible()
        assert outputs[1].size() == outputs[0].size()
        assert outputs[1].toImage().pixelColor(30, 30).name() == '#0000ff'
        window.busy(True)
        assert not frame.read_button.isEnabled()
        window.busy(False)
        assert frame.read_button.isEnabled()
        QTest.mouseClick(frame.select_button, Qt.LeftButton)
        pump(500)
        assert window.capture is not None and not frame.isVisible()
        QTest.keyClick(window.capture, Qt.Key_Escape)
        pump()
        assert frame.isVisible() and window.capture is None
        report = dict(always_on_top=True, native_click_through=True, fresh_pixels=True,
                      hidden_during_capture=True, busy_guard=True, cancel_reselect_keeps_region=True)
        (DATA / 'capture-frame-test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    finally:
        window.exit_requested = True
        window.close()
        panel.close()
        pump(100)

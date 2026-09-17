import os
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import Mock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QCloseEvent, QFontDatabase, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import numpy as np
import soundfile as sf

from euphonia.app import Capture, EmotionTemplateDialog, VoiceDialog, Window
from euphonia.core import VoiceStore

app = QApplication.instance() or QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/msjh.ttc')
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')


class FakeScreen:
    def grabWindow(self, _):
        pixmap = QPixmap(800, 600)
        pixmap.fill(Qt.white)
        pixmap.setDevicePixelRatio(2)
        return pixmap

    def geometry(self):
        return QRect(0, 0, 400, 300)


class UiTests(unittest.TestCase):
    def test_closing_main_window_exits_instead_of_hiding_to_tray(self):
        with patch('euphonia.app.ScreenshotHotkey.start'):
            window = Window(auto_preload=False)
        tray = Mock()
        window.tray = tray
        event = QCloseEvent()
        event.ignore()
        window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertTrue(window.exit_requested)
        tray.hide.assert_called_once()
        self.assertFalse(window.screenshot_hotkey.listener.isRunning())

    def test_new_voice_emotion_template_is_automatically_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'emotion.wav'
            sf.write(audio, np.sin(np.arange(16000) * .03) * .2, 16000)
            voice_dialog = VoiceDialog(VoiceStore(Path(tmp) / 'voices'), None)
            classifier = Mock(return_value=dict(label='joy', score=.91, seconds=.01,
                                                 scores={'joy': .91, 'neutral': .09}))
            dialog = EmotionTemplateDialog(classifier, voice_dialog)
            dialog.path = str(audio)
            dialog.transcript.setPlainText('I am delighted to see you!')
            dialog.detect_and_add()
            self.assertEqual(dialog.template['emotion'], 'joy')
            voice_dialog.emotion_templates.append(dialog.template)
            voice_dialog.refresh_emotions()
            self.assertEqual(voice_dialog.emotion_table.item(0, 0).text(), 'joy  91%')
            dialog.close()
            voice_dialog.close()

    def test_edit_voice_dialog_loads_existing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'voice.wav'
            sf.write(audio, np.sin(np.arange(64000) * .03) * .2, 16000)
            store = VoiceStore(root / 'voices')
            voice = store.save('Editable', 'Original transcript.', audio)
            dialog = VoiceDialog(store, None, voice)
            self.assertEqual(dialog.name.text(), 'Editable')
            self.assertEqual(dialog.transcript.toPlainText(), 'Original transcript.')
            self.assertIn(voice['id'], dialog.path)
            dialog.close()

    def test_retained_models_and_ocr_setting(self):
        window = Window(auto_preload=False)
        original = window.model_selector.currentData()
        fast = window.fast_ocr.isChecked()
        try:
            self.assertEqual(
                [window.model_selector.itemData(i) for i in range(window.model_selector.count())],
                ['faster', 'qwen_streaming', 'f5', 'zipvoice'])
            for backend in ('faster', 'qwen_streaming', 'f5', 'zipvoice'):
                index = window.model_selector.findData(backend)
                self.assertGreaterEqual(index, 0)
                window.model_selector.setCurrentIndex(index)
                self.assertEqual(window.engine.backend, backend)
            window.engine.ocr = object()
            window.fast_ocr.setChecked(not fast)
            self.assertIsNone(window.engine.ocr)
            self.assertEqual(window.engine.ocr_fast, not fast)
            window.busy(True)
            self.assertFalse(window.fast_ocr.isEnabled())
            window.busy(False)
        finally:
            window.fast_ocr.setChecked(fast)
            window.model_selector.setCurrentIndex(window.model_selector.findData(original))
            window.close()
    def test_model_is_preloaded_without_pressing_speak(self):
        with patch('euphonia.app.Engine.prepare', return_value=True) as prepare:
            window = Window()
            try:
                for _ in range(100):
                    app.processEvents()
                    time.sleep(0.01)
                    if prepare.called and window.thread is None:
                        break
                self.assertTrue(prepare.called)
                self.assertIn('保持載入', window.status.text())
            finally:
                window.close()

    def test_capture_during_job_cancels_and_runs_after_completion(self):
        window = Window(auto_preload=False)
        window.show_capture = Mock()
        def work(progress, cancelled):
            while not cancelled():
                time.sleep(0.005)
            return None
        try:
            window.run_job(work, lambda _: None)
            window.start_capture()
            self.assertTrue(window.worker.cancelled.is_set())
            self.assertTrue(window.capture_pending)
            for _ in range(100):
                app.processEvents()
                time.sleep(0.01)
                if window.show_capture.called:
                    break
            window.show_capture.assert_called_once()
            self.assertIsNone(window.thread)
            self.assertFalse(window._capture_after_job)
        finally:
            window.close()

    def test_model_switch_language_and_busy_state(self):
        window = Window(auto_preload=False)
        original = window.model_selector.currentData()
        language = window.language.currentData()
        try:
            window.model_selector.setCurrentIndex(window.model_selector.findData('f5'))
            self.assertEqual(window.engine.backend, 'f5')
            self.assertTrue(window.language.isEnabled())
            window.busy(True)
            self.assertFalse(window.model_selector.isEnabled())
            window.busy(False)
            self.assertTrue(window.model_selector.isEnabled())
            self.assertTrue(window.language.isEnabled())
            window.model_selector.setCurrentIndex(window.model_selector.findData('faster'))
            self.assertTrue(window.language.isEnabled())
            self.assertEqual(window.settings.value('model'), 'faster')
        finally:
            window.model_selector.setCurrentIndex(window.model_selector.findData(original))
            window.language.setCurrentIndex(window.language.findData(language))
            window.close()

    def test_capture_high_dpi_selection_and_escape(self):
        capture = Capture(FakeScreen())
        results = []
        cancelled = []
        capture.selected.connect(results.append)
        capture.cancelled.connect(lambda: cancelled.append(True))
        capture.show()
        QTest.mousePress(capture, Qt.LeftButton, pos=QPoint(20, 30))
        QTest.mouseMove(capture, QPoint(120, 90))
        QTest.mouseRelease(capture, Qt.LeftButton, pos=QPoint(120, 90))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].width(), 202)
        self.assertEqual(results[0].height(), 122)
        capture.show()
        QTest.keyClick(capture, Qt.Key_Escape)
        self.assertEqual(cancelled, [True])
        capture.close()

    def test_background_job_releases_ui_on_completion(self):
        window = Window(auto_preload=False)
        results = []
        window.run_job(lambda progress, cancelled: 'complete', results.append)
        self.assertFalse(window.speak_button.isEnabled())
        for _ in range(100):
            app.processEvents()
            time.sleep(0.02)
            if window.thread is None:
                break
        self.assertIsNone(window.thread)
        self.assertEqual(results, ['complete'])
        self.assertTrue(window.speak_button.isEnabled())
        window.close()


if __name__ == '__main__':
    unittest.main()

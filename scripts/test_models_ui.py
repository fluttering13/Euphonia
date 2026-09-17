"""Exercise actual model selection, worker handoff and silent Qt playback."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from euphonia.app import Window, STYLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', choices=['faster', 'qwen_streaming', 'f5', 'zipvoice'])
    args = parser.parse_args()
    app = QApplication([])
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/msjh.ttc')
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        settings = QSettings(str(Path(temporary) / 'settings.ini'), QSettings.IniFormat)
        with patch('euphonia.app.QSettings', return_value=settings):
            window = Window(auto_preload=False)
        window.audio.setVolume(0)
        errors = []
        window.failed = errors.append
        window.show()
        try:
            for model in args.models:
                window.model_selector.setCurrentIndex(window.model_selector.findData(model))
                window.language.setCurrentIndex(window.language.findData('English'))
                window.editor.setPlainText('The morning sky looks beautiful. Shall we explore the forest together?')
                window.output = None
                QTest.mouseClick(window.speak_button, Qt.LeftButton)
                deadline = time.monotonic() + 600
                while window.thread is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.02)
                if window.thread is not None or errors or not window.output:
                    raise RuntimeError(f'{model}: {errors or "Timed out / no output"}')
                assert Path(window.output).is_file()
                assert window.model_selector.isEnabled()
                results.append(dict(model=model, file=window.output, status=window.status.text(),
                                    metrics=window.engine.last_metrics))
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            window.grab().save(str(ROOT / 'data/model-selector.png'))
        finally:
            window.stop()
            while window.thread is not None:
                app.processEvents()
                time.sleep(0.02)
            window.close()
    (ROOT / 'data/model-ui-test.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

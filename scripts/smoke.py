"""Real OCR/UI checks, optional real model generation."""
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import soundfile as sf
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import QApplication
from euphonia.app import STYLE, Window
from euphonia.core import DATA, Engine, VoiceStore

app = QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/msjh.ttc')
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setStyle('Fusion')
app.setStyleSheet(STYLE)
window = Window()
window.show()
app.processEvents()
DATA.mkdir(exist_ok=True)
window.grab().save(str(DATA / 'ui-preview.png'))
canvas = QImage(1000, 200, QImage.Format_RGB888)
canvas.fill('white')
painter = QPainter(canvas)
painter.setPen('black')
painter.setFont(QFont('Microsoft JhengHei', 32))
painter.drawText(30, 85, 'Euphonia 文字辨識測試')
painter.end()
array = np.frombuffer(canvas.bits(), dtype=np.uint8).reshape(canvas.height(), canvas.bytesPerLine())[:, :canvas.width() * 3].reshape(canvas.height(), canvas.width(), 3)[:, :, ::-1].copy()
engine = Engine(window.store)
text = engine.recognize(array)
print('OCR:', text, flush=True)
assert 'Euphonia' in text, text
assert any('\u4e00' <= ch <= '\u9fff' for ch in text), text
print('UI and OCR passed', flush=True)
if '--tts' in sys.argv:
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / 'reference.wav'
        urllib.request.urlretrieve('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav', reference)
        store = VoiceStore(Path(tmp) / 'voices')
        voice = store.save('Official demo', 'Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.', reference)
        engine = Engine(store)
        output = engine.synthesize(voice, '你好，這是語音朗讀測試。', 'Chinese', lambda msg: print(msg, flush=True))
        audio, sr = sf.read(output)
        assert len(audio) > sr and np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.001
        print('TTS passed:', output, 'seconds:', len(audio) / sr, flush=True)
window.close()

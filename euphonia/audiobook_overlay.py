"""Click-through audiobook regions, excluded from desktop OCR captures."""
import ctypes
import logging
import sys
from contextlib import contextmanager
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

log = logging.getLogger('euphonia.audiobook')


class AudiobookBox(QWidget):
    def __init__(self, region, title, color):
        # No owner window: minimizing/hiding the main window must keep these visible.
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet('background: transparent;')
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowTitle(title)
        self.title, self.color = title, QColor(color)
        self.setGeometry(QRect(region['x'], region['y'], region['width'], region['height']))
        self.capture_excluded = False
        if sys.platform == 'win32' and QApplication.platformName() == 'windows':
            from ctypes import wintypes
            api = ctypes.WinDLL('user32', use_last_error=True)
            api.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
            api.SetWindowDisplayAffinity.restype = wintypes.BOOL
            self.capture_excluded = bool(api.SetWindowDisplayAffinity(int(self.winId()), 0x11))
            if not self.capture_excluded:
                log.warning('Overlay capture exclusion unavailable; using hide/flush fallback')

    def paintEvent(self, event):
        painter = QPainter(self)
        tint = QColor(self.color)
        tint.setAlpha(28)
        painter.fillRect(self.rect(), tint)
        border = QColor(self.color)
        border.setAlpha(210)
        painter.setPen(QPen(border, 2))
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)
        label = QRect(3, 3, min(self.width() - 6, painter.fontMetrics().horizontalAdvance(self.title) + 12), 20)
        painter.fillRect(label, QColor(12, 24, 32, 190))
        painter.setPen(QColor('#f2fbff'))
        painter.drawText(label, Qt.AlignCenter, self.title)


class AudiobookReadButton(QPushButton):
    """A non-activating action next to the dialogue region."""
    def __init__(self, region, callback):
        super().__init__('OCR ＋ 播放', None)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('立即辨識目前角色與對話，並使用對應語音播放')
        self.setStyleSheet('''
            QPushButton {
                color: #f2fbff; background: rgba(18, 54, 78, 235);
                border: 2px solid #72b7ff; border-radius: 6px;
                padding: 4px 10px; font-weight: 600;
            }
            QPushButton:hover { background: rgba(28, 82, 116, 245); }
            QPushButton:pressed { background: rgba(12, 38, 58, 250); }
            QPushButton:disabled { color: #b9c7d0; background: rgba(35, 48, 57, 220); }
        ''')
        self.setFixedSize(116, 32)
        self.clicked.connect(callback)

        desired = QPoint(region['x'] + region['width'] - self.width(), region['y'] - self.height() - 4)
        screen = QApplication.screenAt(QPoint(region['x'] + region['width'] // 2,
                                              region['y'] + region['height'] // 2))
        if screen is not None:
            bounds = screen.geometry()
            if desired.y() < bounds.top():
                desired.setY(min(bounds.bottom() - self.height() + 1,
                                 region['y'] + region['height'] + 4))
            desired.setX(max(bounds.left(), min(desired.x(), bounds.right() - self.width() + 1)))
        self.move(desired)

        self.capture_excluded = False
        if sys.platform == 'win32' and QApplication.platformName() == 'windows':
            from ctypes import wintypes
            api = ctypes.WinDLL('user32', use_last_error=True)
            api.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
            api.SetWindowDisplayAffinity.restype = wintypes.BOOL
            self.capture_excluded = bool(api.SetWindowDisplayAffinity(int(self.winId()), 0x11))
            if not self.capture_excluded:
                log.warning('Action capture exclusion unavailable; using hide/flush fallback')

    def set_busy(self, busy, label=None):
        self.setEnabled(not busy)
        self.setText((label or '辨識中…') if busy else 'OCR ＋ 播放')


class AudiobookEmotionPanel(QLabel):
    def __init__(self, region, read_button=None):
        super().__init__('情緒：等待 OCR\nSampling：尚未選擇', None)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet('''
            QLabel {
                color: #f2fbff; background: rgba(12, 24, 32, 225);
                border: 2px solid #b891ff; border-radius: 6px;
                padding: 5px 9px; font-size: 12px;
            }
        ''')
        self.setFixedSize(310, 52)
        if read_button is not None:
            # Keep the current routing information and its action together as one
            # toolbar, with the passive information immediately left of the button.
            desired = QPoint(read_button.x() - self.width() - 4,
                             read_button.y() + read_button.height() - self.height())
        else:
            desired = QPoint(region['x'], region['y'] + region['height'] + 4)
        screen = QApplication.screenAt(QPoint(region['x'] + region['width'] // 2,
                                              region['y'] + region['height'] // 2))
        if screen is not None:
            bounds = screen.geometry()
            if read_button is not None:
                available = read_button.x() - bounds.left() - 4
                if available >= 120:
                    self.setFixedWidth(min(self.width(), available))
                    desired.setX(read_button.x() - self.width() - 4)
                else:
                    # Very narrow layouts cannot fit both controls on one row.
                    desired = QPoint(region['x'], region['y'] + region['height'] + 4)
                desired.setY(max(bounds.top(), desired.y()))
            elif desired.y() + self.height() > bounds.bottom() + 1:
                desired.setY(max(bounds.top(), region['y'] - self.height() - 40))
            desired.setX(max(bounds.left(), min(desired.x(), bounds.right() - self.width() + 1)))
        self.move(desired)
        self.capture_excluded = False
        if sys.platform == 'win32' and QApplication.platformName() == 'windows':
            from ctypes import wintypes
            api = ctypes.WinDLL('user32', use_last_error=True)
            api.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
            api.SetWindowDisplayAffinity.restype = wintypes.BOOL
            self.capture_excluded = bool(api.SetWindowDisplayAffinity(int(self.winId()), 0x11))

    def show_emotion(self, result):
        if result is None:
            self.setText('情緒：分析中…\nSampling：等待選擇')
            return
        confidence = result.get('score', 0)
        emotion = result.get('routed') or result.get('label') or 'neutral'
        sample = result.get('sample_label') or result.get('sample_id') or '主要參考音檔'
        self.setText(f'情緒：{emotion}  {confidence:.0%}\nSampling：{sample}')


class AudiobookOverlays:
    def __init__(self, read_callback=None):
        self.boxes = []
        self.read_callback = read_callback
        self.read_button = None
        self.emotion_panel = None

    def show(self, regions):
        self.close()
        for key, title, color in [('character_name', '角色名稱 · 有聲小說 ON', '#58ddbe'),
                                  ('dialogue_text', '對話文字 · 有聲小說 ON', '#72b7ff')]:
            box = AudiobookBox(regions[key], title, color)
            self.boxes.append(box)
            box.show()
        if self.read_callback is not None:
            self.read_button = AudiobookReadButton(regions['dialogue_text'], self.read_callback)
            self.read_button.show()
        self.emotion_panel = AudiobookEmotionPanel(regions['dialogue_text'], self.read_button)
        self.emotion_panel.show()

    def close(self):
        for box in self.boxes:
            box.close()
            box.deleteLater()
        self.boxes.clear()
        if self.read_button is not None:
            self.read_button.close()
            self.read_button.deleteLater()
            self.read_button = None
        if self.emotion_panel is not None:
            self.emotion_panel.close()
            self.emotion_panel.deleteLater()
            self.emotion_panel = None

    def set_action_busy(self, busy, label=None):
        if self.read_button is not None:
            self.read_button.set_busy(busy, label)

    def set_emotion(self, result):
        if self.emotion_panel is not None:
            self.emotion_panel.show_emotion(result)

    @contextmanager
    def clean_capture(self):
        widgets = list(self.boxes)
        widgets += [widget for widget in (self.read_button, self.emotion_panel) if widget is not None]
        hidden = [widget for widget in widgets if widget.isVisible() and not widget.capture_excluded]
        for box in hidden:
            box.hide()
        if hidden and sys.platform == 'win32' and QApplication.platformName() == 'windows':
            # Wait for the compositor to remove fallback overlays before grabbing.
            ctypes.WinDLL('dwmapi').DwmFlush()
        try:
            yield
        finally:
            for box in hidden:
                box.show()

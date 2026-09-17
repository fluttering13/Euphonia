import shutil
import sys
import threading
import logging
import time
import json

import numpy as np
from PySide6.QtCore import QLockFile, QObject, QRect, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QRubberBand, QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget, QSystemTrayIcon, QMenu, QStyle,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from .core import DATA, Engine, VoiceStore
from .backends import BACKENDS
from .capture_frame import CaptureFrame
from .audiobook import load_config
from .audiobook_settings import AudiobookSettings
from .audiobook_monitor import AudiobookMonitor

log = logging.getLogger('euphonia.capture')
from .hotkeys import ScreenshotHotkey, foreground_window, activate_window


class Worker(QObject):
    done = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    finished = Signal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit, self.cancelled.is_set))
        except Exception as exc:
            self.error.emit(f'{type(exc).__name__}: {exc}')
        finally:
            self.finished.emit()


class Capture(QWidget):
    selected = Signal(object)
    cancelled = Signal()

    def __init__(self, screen):
        super().__init__()
        self.screen_source = screen
        self.selected_region = None
        self.snapshot = screen.grabWindow(0)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CrossCursor)
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.snapshot)
        painter.fillRect(self.rect(), QColor(8, 15, 25, 95))
        if self.rubber.isVisible():
            rect = self.rubber.geometry()
            painter.save()
            painter.setClipRect(rect)
            painter.drawPixmap(self.rect(), self.snapshot)
            painter.restore()
            painter.setPen(QPen(QColor('#58ddbe'), 2))
            painter.drawRect(rect)
        painter.setPen(QColor('white'))
        painter.drawText(24, 36, '拖曳框選文字區域 · Esc 取消')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.position().toPoint()
            self.rubber.setGeometry(QRect(self.origin, self.origin))
            self.rubber.show()

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            self.rubber.setGeometry(QRect(self.origin, event.position().toPoint()).normalized().intersected(self.rect()))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self.origin is None:
            return
        rect = self.rubber.geometry()
        if rect.width() < 8 or rect.height() < 8:
            self.origin = None
            self.rubber.hide()
            self.update()
            return
        ratio = self.snapshot.devicePixelRatio()
        self.selected_region = rect.translated(self.geometry().topLeft())
        crop = self.snapshot.copy(QRect(round(rect.x() * ratio), round(rect.y() * ratio), round(rect.width() * ratio), round(rect.height() * ratio)))
        self.hide()
        self.selected.emit(crop)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide()
            self.cancelled.emit()


class EmotionTemplateDialog(QDialog):
    def __init__(self, classify, parent=None):
        super().__init__(parent)
        self.classify = classify
        self.path = ''
        self.template = None
        self.setWindowTitle('新增情緒模板')
        self.resize(560, 360)
        layout = QVBoxLayout(self)
        note = QLabel('選擇一段只有此角色的語音，並輸入完全對應的逐字稿。\n程式會從文字自動判斷情緒分類。')
        note.setWordWrap(True)
        layout.addWidget(note)
        choose = QPushButton('選擇情緒語音…')
        choose.clicked.connect(self.choose)
        layout.addWidget(choose)
        self.file_label = QLabel('尚未選擇音檔')
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.transcript = QTextEdit()
        self.transcript.setPlaceholderText('輸入這份情緒語音中實際說出的完整文字…')
        layout.addWidget(self.transcript, 1)
        self.result = QLabel('')
        layout.addWidget(self.result)
        add = QPushButton('自動辨識情緒並加入')
        add.setObjectName('primary')
        add.clicked.connect(self.detect_and_add)
        layout.addWidget(add)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, '選擇情緒語音', '', '音訊 (*.wav *.flac *.mp3 *.ogg)')
        if path:
            self.path = path
            self.file_label.setText(path)

    def detect_and_add(self):
        try:
            transcript = self.transcript.toPlainText().strip()
            if not self.path or not transcript:
                raise ValueError('請選擇音檔並輸入對應逐字稿。')
            # Validate before loading the classifier so bad files fail quickly.
            VoiceStore.read_reference(self.path, minimum_seconds=.4)
            self.result.setText('正在辨識情緒…')
            QApplication.processEvents()
            detected = self.classify(transcript)
            self.template = dict(audio_path=self.path, transcript=transcript,
                                 emotion=detected['label'], score=detected['score'],
                                 scores=detected['scores'])
            self.result.setText(f"辨識結果：{detected['label']}（{detected['score']:.0%}）")
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, '無法加入情緒模板', str(exc))


class VoiceDialog(QDialog):
    def __init__(self, store, parent, voice=None):
        super().__init__(parent)
        self.store = store
        self.original_voice = voice
        self.voice = voice
        self.path = str(store.audio_path(voice['id'])) if voice else ''
        self.emotion_templates = store.emotion_templates(voice) if voice else []
        self.emotion_classifiers = {}
        self.setWindowTitle('編輯聲音角色' if voice else '建立聲音角色')
        self.resize(700, 650)
        layout = QVBoxLayout(self)
        hint = QLabel('匯入 3～30 秒清楚的語音，並填入音檔中實際說出的文字。\n建議只有一位說話者，且沒有背景音樂。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText('例如：我的旁白')
        if voice:
            self.name.setText(voice['name'])
        form.addRow('角色名稱', self.name)
        layout.addLayout(form)
        choose = QPushButton('選擇參考音檔…')
        choose.clicked.connect(self.choose)
        layout.addWidget(choose)
        self.file_label = QLabel(self.path or '尚未選擇音檔')
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.transcript = QTextEdit()
        self.transcript.setPlaceholderText('輸入參考音檔的完整逐字稿，請與語音內容一致…')
        if voice:
            self.transcript.setPlainText(voice['transcript'])
        layout.addWidget(self.transcript)
        layout.addWidget(QLabel('情緒模板（選用）'))
        self.emotion_table = QTableWidget(0, 3)
        self.emotion_table.setHorizontalHeaderLabels(['自動分類', '音檔', '對應逐字稿'])
        self.emotion_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.emotion_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.emotion_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.emotion_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.emotion_table.setMinimumHeight(130)
        layout.addWidget(self.emotion_table)
        emotion_actions = QHBoxLayout()
        add_emotion = QPushButton('新增情緒模板…')
        add_emotion.clicked.connect(self.add_emotion)
        emotion_actions.addWidget(add_emotion)
        remove_emotion = QPushButton('移除選取模板')
        remove_emotion.clicked.connect(self.remove_emotion)
        emotion_actions.addWidget(remove_emotion)
        layout.addLayout(emotion_actions)
        self.refresh_emotions()
        save = QPushButton('儲存變更' if voice else '儲存角色')
        save.setObjectName('primary')
        save.clicked.connect(self.save)
        layout.addWidget(save)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, '選擇參考音檔', '', '音訊 (*.wav *.flac *.mp3 *.ogg)')
        if path:
            self.path = path
            self.file_label.setText(path)

    def classify_emotion(self, text):
        from .emotion import EmotionClassifier, model_for_text
        model_dir = model_for_text(text)
        if model_dir not in self.emotion_classifiers:
            self.emotion_classifiers[model_dir] = EmotionClassifier(model_dir)
        return self.emotion_classifiers[model_dir].classify(text)

    def add_emotion(self):
        dialog = EmotionTemplateDialog(self.classify_emotion, self)
        if dialog.exec() == QDialog.Accepted and dialog.template:
            self.emotion_templates.append(dialog.template)
            self.refresh_emotions()

    def remove_emotion(self):
        rows = sorted({index.row() for index in self.emotion_table.selectedIndexes()}, reverse=True)
        for row in rows:
            del self.emotion_templates[row]
        self.refresh_emotions()

    def refresh_emotions(self):
        self.emotion_table.setRowCount(len(self.emotion_templates))
        for row, item in enumerate(self.emotion_templates):
            values = [f"{item['emotion']}  {item['score']:.0%}", item['audio_path'], item['transcript']]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip(value)
                self.emotion_table.setItem(row, column, cell)

    def save(self):
        try:
            if not self.path:
                raise ValueError('請先選擇參考音檔。')
            if self.original_voice:
                self.voice = self.store.update(self.original_voice['id'], self.name.text(),
                                               self.transcript.toPlainText(), self.path,
                                               self.emotion_templates)
            else:
                self.voice = self.store.save(self.name.text(), self.transcript.toPlainText(), self.path,
                                             self.emotion_templates)
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, '無法儲存', str(exc))


class Window(QMainWindow):
    def __init__(self, *, auto_preload=True):
        super().__init__()
        self.auto_preload = auto_preload
        self.pipeline = None
        self.store = VoiceStore()
        self.engine = Engine(self.store)
        self.thread = None
        self.worker = None
        self.capture = None
        self.capture_frame = None
        self.capture_pending = False
        self._region_target = None
        self.audiobook = None
        self.audiobook_dialog = None
        self._capture_after_job = False
        self.return_to_main = True
        self.previous_foreground = None
        self.exit_requested = False
        self.tray = None
        self.output = None
        self.settings = QSettings('Euphonia', 'Euphonia')
        self.setWindowTitle('Euphonia · 看見文字，聽見聲音')
        self.resize(1220, 820)
        self.setMinimumSize(980, 700)
        logo_path = DATA / 'logo/logo.png'
        if logo_path.is_file():
            self.setWindowIcon(QIcon(str(logo_path)))
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.errorOccurred.connect(lambda *_: self.status.setText('播放失敗：' + self.player.errorString()))
        root = QWidget()
        root.setObjectName('appRoot')
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(30, 22, 30, 24)
        layout.setSpacing(16)

        header = QWidget()
        header.setObjectName('brandHeader')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 24, 10)
        header_layout.setSpacing(18)
        logo = QLabel()
        logo.setObjectName('brandLogo')
        logo.setFixedSize(112, 112)
        logo.setAlignment(Qt.AlignCenter)
        if logo_path.is_file():
            logo.setPixmap(QPixmap(str(logo_path)).scaled(
                108, 108, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header_layout.addWidget(logo)
        brand = QVBoxLayout()
        brand.setSpacing(2)
        eyebrow = QLabel('VOICE  ·  OCR  ·  LOCAL AI')
        eyebrow.setObjectName('eyebrow')
        brand.addWidget(eyebrow)
        title = QLabel('EUPHONIA')
        title.setObjectName('title')
        brand.addWidget(title)
        subtitle = QLabel('把畫面上的文字，交給熟悉的聲音。')
        subtitle.setObjectName('subtitle')
        brand.addWidget(subtitle)
        header_layout.addLayout(brand, 1)
        seal = QLabel('ΜΟΥΣΑ')
        seal.setObjectName('seal')
        seal.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(seal)
        layout.addWidget(header)

        def section(text):
            label = QLabel(text)
            label.setObjectName('sectionTitle')
            return label

        split = QSplitter()
        split.setObjectName('mainSplit')
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)
        sidebar = QWidget()
        sidebar.setObjectName('sidebar')
        left = QVBoxLayout(sidebar)
        left.setContentsMargins(20, 18, 20, 18)
        left.setSpacing(9)
        left.addWidget(section('I  ·  聲音角色'))
        self.voices = QComboBox()
        self.voices.currentIndexChanged.connect(self.voice_changed)
        left.addWidget(self.voices)
        self.add_button = QPushButton('＋ 建立角色')
        self.add_button.clicked.connect(self.add_voice)
        left.addWidget(self.add_button)
        self.edit_button = QPushButton('編輯所選角色')
        self.edit_button.clicked.connect(self.edit_voice)
        left.addWidget(self.edit_button)
        self.delete_button = QPushButton('刪除所選角色')
        self.delete_button.setObjectName('dangerButton')
        self.delete_button.clicked.connect(self.delete_voice)
        left.addWidget(self.delete_button)
        self.voice_info = QLabel()
        self.voice_info.setObjectName('voiceInfo')
        self.voice_info.setWordWrap(True)
        left.addWidget(self.voice_info)
        self.reference_button = QPushButton('試聽參考音檔')
        self.reference_button.clicked.connect(self.preview_voice)
        left.addWidget(self.reference_button)
        left.addSpacing(12)
        left.addWidget(section('語音模型'))
        self.model_selector = QComboBox()
        for key, (label, _) in BACKENDS.items():
            self.model_selector.addItem(label, key)
        left.addWidget(self.model_selector)
        left.addSpacing(8)
        left.addWidget(section('II  ·  朗讀設定'))
        self.language = QComboBox()
        for label, value in [('中文', 'Chinese'), ('自動辨識', 'Auto'), ('英文', 'English'), ('日文', 'Japanese'), ('韓文', 'Korean')]:
            self.language.addItem(label, value)
        left.addWidget(self.language)
        self.auto_read = QCheckBox('截圖後自動辨識並播放')
        left.addWidget(self.auto_read)
        self.fast_ocr = QCheckBox('快速 OCR')
        self.fast_ocr.setToolTip('適合框選遊戲字幕；若小字辨識不完整，可取消勾選改用原設定。')
        self.fast_ocr.setChecked(self.settings.value('fast_ocr', True, type=bool))
        self.engine.ocr_fast = self.fast_ocr.isChecked()
        self.fast_ocr.toggled.connect(self.ocr_mode_changed)
        left.addWidget(self.fast_ocr)
        self.background_button = QPushButton('背景執行（遊戲模式）')
        self.background_button.clicked.connect(self.hide_to_tray)
        left.addWidget(self.background_button)
        left.addWidget(section('截圖螢幕'))
        self.screens = QComboBox()
        for i, screen in enumerate(QApplication.screens()):
            self.screens.addItem(f'{i + 1} · {screen.name()}', i)
        left.addWidget(self.screens)
        left.addStretch()
        note = QLabel('角色採樣可共用於不同模型。\n語音角色、OCR 與生成皆在本機處理。')
        note.setObjectName('sidebarNote')
        note.setWordWrap(True)
        note.setMinimumHeight(58)
        left.addWidget(note)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName('sidebarScroll')
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameStyle(0)
        sidebar_scroll.setWidget(sidebar)
        split.addWidget(sidebar_scroll)
        main = QWidget()
        main.setObjectName('workspace')
        right = QVBoxLayout(main)
        right.setContentsMargins(22, 18, 22, 18)
        right.setSpacing(11)
        row = QHBoxLayout()
        self.capture_button = QPushButton('框選截圖  F12')
        self.capture_button.setObjectName('captureButton')
        self.capture_button.setToolTip('F12 全域截圖；也可在工具視窗內按 Ctrl+Shift+S')
        self.capture_button.clicked.connect(self.start_capture)
        row.addWidget(self.capture_button)
        self.image_button = QPushButton('匯入圖片')
        self.image_button.clicked.connect(self.import_image)
        row.addWidget(self.image_button)
        right.addLayout(row)
        audiobook_row = QHBoxLayout()
        self.audiobook_settings_button = QPushButton('有聲小說設定')
        self.audiobook_settings_button.clicked.connect(self.open_audiobook_settings)
        audiobook_row.addWidget(self.audiobook_settings_button)
        self.monitor_button = QPushButton('背景監控 ON / OFF  F11')
        self.monitor_button.clicked.connect(lambda: self.audiobook.toggle())
        audiobook_row.addWidget(self.monitor_button)
        right.addLayout(audiobook_row)
        self.monitor_status = QLabel('背景監控 OFF · F11 開啟')
        self.monitor_status.setObjectName('monitorStatus')
        self.monitor_status.setWordWrap(True)
        right.addWidget(self.monitor_status)
        self.preview = QLabel('框選畫面或匯入圖片，自動辨識並播放')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(150)
        self.preview.setMaximumHeight(210)
        self.preview.setObjectName('preview')
        right.addWidget(self.preview)
        right.addWidget(section('III  ·  文字內容  /  可在朗讀前修正'))
        self.editor = QTextEdit()
        self.editor.setPlaceholderText('OCR 結果會顯示在這裡，也可以直接貼上想朗讀的文字…')
        right.addWidget(self.editor, 1)
        self.count = QLabel('0 / 5000 字')
        self.count.setObjectName('countLabel')
        self.editor.textChanged.connect(lambda: self.count.setText(f'{len(self.editor.toPlainText())} / 5000 字'))
        right.addWidget(self.count)
        actions = QHBoxLayout()
        self.speak_button = QPushButton('生成並朗讀')
        self.speak_button.setObjectName('primary')
        self.speak_button.clicked.connect(self.speak)
        actions.addWidget(self.speak_button)
        stop = QPushButton('停止')
        stop.clicked.connect(self.stop)
        actions.addWidget(stop)
        self.replay_button = QPushButton('重播')
        self.replay_button.clicked.connect(lambda: self.play(self.output))
        self.replay_button.setEnabled(False)
        actions.addWidget(self.replay_button)
        self.export_button = QPushButton('匯出 WAV')
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        actions.addWidget(self.export_button)
        right.addLayout(actions)
        split.addWidget(main)
        split.setSizes([285, 760])
        self.status = QLabel('準備就緒 · 先建立聲音角色，或擷取文字。')
        self.status.setObjectName('statusBar')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.shortcut = QShortcut(QKeySequence('Ctrl+Shift+S'), self)
        self.shortcut.activated.connect(self.start_capture)
        self.refresh_voices(self.settings.value('voice', ''))
        index = self.language.findData(self.settings.value('language', 'Chinese'))
        self.language.setCurrentIndex(max(index, 0))
        # Replace the previous opt-in preference with automatic playback by default.
        self.auto_read.setChecked(self.settings.value('auto_play_after_ocr', True, type=bool))
        self.auto_read.toggled.connect(lambda enabled: self.settings.setValue('auto_play_after_ocr', enabled))
        self.screens.setCurrentIndex(min(self.settings.value('screen', 0, type=int), self.screens.count() - 1))
        self.setup_tray()
        saved_model = self.settings.value('model', 'faster')
        self.model_selector.setCurrentIndex(max(0, self.model_selector.findData(saved_model)))
        self.model_selector.currentIndexChanged.connect(self.model_changed)
        self.model_changed()
        self.screenshot_hotkey = ScreenshotHotkey(self)
        self.screenshot_hotkey.pressed.connect(self.start_capture)
        self.screenshot_hotkey.failed.connect(self.notify_error)
        self.screenshot_hotkey.start()
        self.audiobook = AudiobookMonitor(self, load_config(self.settings))
        self.audiobook.changed.connect(self.audiobook_status_changed)
        self.monitor_hotkey = ScreenshotHotkey(self, virtual_key=0x7A)
        self.monitor_hotkey.pressed.connect(self.audiobook.toggle)
        self.monitor_hotkey.failed.connect(self.notify_error)
        self.monitor_hotkey.start()
        QApplication.instance().aboutToQuit.connect(self.screenshot_hotkey.stop)
        QApplication.instance().aboutToQuit.connect(self.monitor_hotkey.stop)
        QApplication.instance().aboutToQuit.connect(self.audiobook.shutdown)
        QApplication.instance().aboutToQuit.connect(self.engine.close)

    def audiobook_status_changed(self, text):
        self.monitor_status.setText(text)
        if self.tray:
            self.tray.setToolTip('Euphonia · F12 截圖 · ' + text[:90])

    def open_audiobook_settings(self):
        if self.audiobook_dialog is not None:
            self.audiobook_dialog.show()
            self.audiobook_dialog.raise_()
            return
        self.audiobook.set_enabled(False)
        dialog = AudiobookSettings(self.audiobook.config, self.store.list(), self.settings, self)
        self.audiobook_dialog = dialog
        def select_region(key):
            dialog.hide()
            self._region_target = (dialog, key)
            self.start_capture()
        dialog.select_region.connect(select_region)
        def finished(result):
            if result == QDialog.Accepted:
                self.audiobook.config = dialog.config
                self.audiobook.status('設定已儲存')
            self.audiobook_dialog = None
            dialog.deleteLater()
        dialog.finished.connect(finished)
        # Hiding a dialog for region selection terminates exec()'s nested loop.
        # open() keeps its lifetime explicit until Save/Cancel, across both picks.
        dialog.open()

    def model_changed(self):
        backend = self.model_selector.currentData()
        self.engine.set_backend(backend)
        self.settings.setValue('model', backend)
        limited = ('Chinese', 'English', 'Auto') if backend in ('f5', 'zipvoice') else None
        if limited is not None and self.language.currentData() not in limited:
            self.language.setCurrentIndex(self.language.findData('English'))
        for index in range(self.language.count()):
            self.language.model().item(index).setEnabled(
                limited is None or self.language.itemData(index) in limited)
        self.language.setEnabled(True)
        self.status.setText(f'已選擇 {BACKENDS[backend][0]}；正在準備常駐模型。')
        if self.auto_preload:
            QTimer.singleShot(0, self.preload_model)

    def preload_model(self):
        if self.audiobook is not None and self.audiobook.enabled:
            return
        voice = self.voices.currentData()
        if self.thread is not None or self.capture_pending or not voice or self.exit_requested:
            return
        language = self.language.currentData()
        self.run_job(lambda progress, cancelled: self.engine.prepare(voice, language, progress, cancelled),
                     lambda ready: self.status.setText('模型與角色已就緒，保持載入，可連續截圖朗讀。' if ready else '已取消暖機；已載入的模型仍保留。'))

    def setup_tray(self):
        if QApplication.platformName() != 'windows' or not QSystemTrayIcon.isSystemTrayAvailable():
            return
        logo_path = DATA / 'logo/logo.png'
        icon = QIcon(str(logo_path)) if logo_path.is_file() else self.style().standardIcon(QStyle.SP_MediaVolume)
        self.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip('Euphonia · F12 截圖並朗讀')
        self.tray_menu = QMenu(self)
        self.tray_menu.addAction('開啟主視窗', self.show_main)
        self.tray_menu.addAction('截圖並朗讀（F12）', self.start_capture)
        self.tray_menu.addAction('背景監控 ON / OFF（F11）', lambda: self.audiobook.toggle())
        self.tray_menu.addAction('停止朗讀', self.stop)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction('結束 Euphonia', self.request_exit)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self.tray_activated)
        self.tray.show()

    def tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_main()

    def show_main(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def save_settings(self):
        voice = self.voices.currentData()
        self.settings.setValue('voice', voice['id'] if voice else '')
        self.settings.setValue('language', self.language.currentData())
        self.settings.setValue('auto_play_after_ocr', self.auto_read.isChecked())
        self.settings.setValue('screen', self.screens.currentIndex())
        self.settings.sync()

    def hide_to_tray(self):
        self.save_settings()
        if self.tray is not None:
            self.hide()
        else:
            self.showMinimized()

    def request_exit(self):
        self.exit_requested = True
        self.close()

    def notify_error(self, message):
        self.status.setText(message)
        if self.tray is not None:
            self.tray.showMessage('Euphonia', message, QSystemTrayIcon.Warning)

    def refresh_voices(self, selected=None):
        self.voices.clear()
        for voice in self.store.list():
            self.voices.addItem(voice['name'], voice)
            if voice['id'] == selected:
                self.voices.setCurrentIndex(self.voices.count() - 1)
        self.voice_changed()

    def voice_changed(self):
        voice = self.voices.currentData()
        if voice:
            emotions = self.engine.emotion_variants(voice)
            suffix = f"\n情緒模板：{', '.join(emotions)}" if emotions else ''
            self.voice_info.setText(f"{voice['duration']} 秒參考語音\n{voice['transcript']}{suffix}")
        else:
            self.voice_info.setText('尚無角色，點選「建立角色」加入聲音。')
        if self.auto_preload and hasattr(self, 'status'):
            QTimer.singleShot(0, self.preload_model)

    def add_voice(self):
        dialog = VoiceDialog(self.store, self)
        if dialog.exec():
            self.refresh_voices(dialog.voice['id'])

    def edit_voice(self):
        voice = self.voices.currentData()
        if not voice:
            return
        dialog = VoiceDialog(self.store, self, voice)
        if dialog.exec():
            self.engine.prepared = {key for key in self.engine.prepared if key[1] != voice['id']}
            for key in [key for key in self.engine.prompts
                        if key == voice['id'] or key.startswith(voice['id'] + ':')]:
                self.engine.prompts.pop(key, None)
            self.refresh_voices(dialog.voice['id'])

    def delete_voice(self):
        voice = self.voices.currentData()
        if voice and QMessageBox.question(self, '刪除角色', f"確定刪除「{voice['name']}」及其參考音檔？") == QMessageBox.Yes:
            self.player.stop()
            self.player.setSource(QUrl())
            try:
                self.store.delete(voice['id'])
                for key in [key for key in self.engine.prompts
                            if key == voice['id'] or key.startswith(voice['id'] + ':')]:
                    self.engine.prompts.pop(key, None)
                self.refresh_voices()
            except OSError as exc:
                self.failed(str(exc))

    def preview_voice(self):
        voice = self.voices.currentData()
        if voice:
            self.play(str(self.store.audio_path(voice['id'])))

    def busy(self, value):
        self.audiobook_settings_button.setEnabled(not value)
        value = value or (self.audiobook is not None and self.audiobook.enabled)
        self.reference_button.setEnabled(not value)
        self.replay_button.setEnabled(not value and bool(self.output))
        self.fast_ocr.setEnabled(not value)
        if self.capture_frame is not None:
            self.capture_frame.set_busy(value)
        self.model_selector.setEnabled(not value)
        for control in (self.add_button, self.edit_button, self.delete_button, self.capture_button, self.image_button, self.speak_button, self.voices, self.language, self.auto_read, self.editor):
            control.setEnabled(not value)
        self.language.setEnabled(not value)

    def ocr_mode_changed(self, fast):
        self.settings.setValue('fast_ocr', fast)
        with self.engine.ocr_lock:
            self.engine.ocr_fast = fast
            self.engine.ocr = None

    def run_job(self, fn, callback, error_callback=None, progress_callback=None):
        if self.thread is not None:
            return
        self.busy(True)
        self.worker = Worker(fn)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.status.setText)
        if progress_callback is not None:
            self.worker.progress.connect(progress_callback)
        self.worker.done.connect(callback)
        self.worker.error.connect(error_callback or self.failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.job_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def job_finished(self):
        self.thread = None
        self.worker = None
        self.busy(False)
        if self.audiobook is not None:
            self.audiobook.job_finished()
        if self.exit_requested:
            self.close()
            return
        if self._capture_after_job:
            self._capture_after_job = False
            self._pending_read = False
            log.info('starting capture queued during inference')
            self.hide()
            QTimer.singleShot(300, self.show_capture)
            return
        if getattr(self, '_pending_read', False):
            self._pending_read = False
            self.speak()

    def failed(self, message):
        self.pipeline = None
        self.status.setText('操作失敗，可修正後重試。')
        if self.isVisible() and not self.isMinimized():
            QMessageBox.warning(self, 'Euphonia', message)
        else:
            self.notify_error(message)

    def start_capture(self):
        if self.audiobook is not None:
            self.audiobook.set_enabled(False)
        log.info('capture requested; busy=%s overlay=%s pending=%s modal=%s',
                 self.thread is not None, self.capture is not None, self.capture_pending,
                 QApplication.activeModalWidget() is not None)
        if self.capture is not None or self.capture_pending or QApplication.activeModalWidget() is not None:
            return
        self.capture_pending = True
        if self.capture_frame is not None:
            self.capture_frame.hide()
        self.return_to_main = self.isActiveWindow() and not self.isMinimized()
        self.previous_foreground = foreground_window()
        self.player.stop()
        if self.thread is not None:
            self._capture_after_job = True
            self._pending_read = False
            self.worker.cancelled.set()
            self.status.setText('已收到截圖要求，正在停止目前工作後開啟框選…')
            return
        self.hide()
        QTimer.singleShot(300, self.show_capture)

    def show_capture(self):
        if not self.capture_pending:
            return
        try:
            screens = QApplication.screens()
            screen = screens[min(max(self.screens.currentIndex(), 0), len(screens) - 1)]
            self.capture = Capture(screen)
            self.capture_pending = False
            self.capture.selected.connect(self.captured)
            self.capture.cancelled.connect(self.restore)
            self.capture.show()
            log.info('capture overlay shown')
            self.capture.raise_()
            self.capture.activateWindow()
            activate_window(int(self.capture.winId()))
        except Exception as exc:
            self.restore()
            self.failed(str(exc))

    def restore(self):
        log.info('capture overlay dismissed')
        self.capture_pending = False
        if self.capture:
            self.capture.hide()
            self.capture.deleteLater()
            self.capture = None
        if self.capture_frame is not None:
            self.capture_frame.show()
        if self.return_to_main:
            self.show_main()
        else:
            # Let Qt finish disposing the overlay before restoring native focus.
            previous = self.previous_foreground
            QTimer.singleShot(50, lambda: activate_window(previous))
        if self._region_target:
            dialog, _ = self._region_target
            self._region_target = None
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

    def captured(self, pixmap):
        if self._region_target:
            dialog, key = self._region_target
            if self.capture is not None and self.capture.selected_region is not None:
                dialog.set_region(key, self.capture.selected_region)
            self.restore()
            return
        if self.capture is not None and self.capture.selected_region is not None:
            if self.capture_frame is not None:
                self.capture_frame.close()
                self.capture_frame.deleteLater()
            self.capture_frame = CaptureFrame(self.capture.screen_source, self.capture.selected_region, self)
            self.capture_frame.requested.connect(self.repeat_capture)
            self.capture_frame.reselect.connect(self.start_capture)
        self.restore()
        self.recognize(pixmap)

    def repeat_capture(self):
        if self.thread is not None or self.capture_pending or self.capture is not None or QApplication.activeModalWidget() is not None:
            return
        frame = self.capture_frame
        if frame is None:
            return
        capture_started = time.perf_counter()
        self.capture_pending = True
        self.return_to_main = self.isActiveWindow() and not self.isMinimized()
        self.previous_foreground = foreground_window()
        self.player.stop()
        frame.hide()
        self.hide()
        # Let the compositor remove our border/toolbar before taking fresh pixels.
        def grab():
            if not self.capture_pending or self.exit_requested:
                return
            try:
                screen = frame.screen_source
                if screen not in QApplication.screens():
                    raise RuntimeError('原截圖螢幕已移除，請按 F12 重新框選。')
                snapshot = screen.grabWindow(0)
                region = frame.region.translated(-screen.geometry().topLeft())
                ratio = snapshot.devicePixelRatio()
                crop = snapshot.copy(QRect(round(region.x() * ratio), round(region.y() * ratio),
                                           round(region.width() * ratio), round(region.height() * ratio)))
                self.repeat_capture_seconds = time.perf_counter() - capture_started
                self.restore()
                self.recognize(crop)
            except Exception as exc:
                self.restore()
                self.failed(str(exc))
        QTimer.singleShot(150, grab)

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '選擇圖片', '', '圖片 (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            self.recognize(QPixmap(path))

    def recognize(self, pixmap):
        self.pipeline = {'started': time.perf_counter(), 'backend': self.engine.backend}
        if pixmap.isNull():
            self.failed('無法讀取圖片。')
            return
        from PySide6.QtGui import QImage
        self.preview.setPixmap(pixmap.scaled(700, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        array = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
        bgr = array[:, :, ::-1].copy()
        self.status.setText('正在辨識圖片中的文字…')
        self.run_job(lambda progress, cancelled: self.engine.recognize(bgr), self.ocr_done)

    def ocr_done(self, text):
        if self.pipeline is not None:
            self.pipeline['ocr_seconds'] = self.engine.last_ocr_seconds
            self.pipeline['ocr_pipeline_seconds'] = time.perf_counter() - self.pipeline['started']
        if self.worker.cancelled.is_set():
            self.pipeline = None
            self.status.setText('已取消辨識。')
            return
        if not text:
            self.pipeline = None
            self.notify_error('未辨識到文字，請選擇更清楚的文字區域。')
            return
        self.editor.setPlainText(text)
        self._pending_read = self.auto_read.isChecked()
        if not self._pending_read:
            self.pipeline = None
        self.status.setText('文字辨識完成，準備自動生成並播放…' if self._pending_read else '文字辨識完成，可以修正後朗讀。')

    def speak(self):
        voice = self.voices.currentData()
        text = self.editor.toPlainText().strip()
        if not voice or not text:
            self.failed('請先建立並選擇聲音角色，並輸入要朗讀的文字。')
            return
        if len(text) > 5000:
            self.failed('每次最多 5000 字，請分次朗讀。')
            return
        self.player.stop()
        language = self.language.currentData()
        self.status.setText('準備生成語音…')
        self._tts_started = time.perf_counter()
        self.run_job(lambda progress, cancelled: self.engine.synthesize(voice, text, language, progress, cancelled), self.speech_done)

    def speech_done(self, path):
        if not path or self.worker.cancelled.is_set():
            self.pipeline = None
            self.status.setText('已停止生成。')
            return
        self.output = path
        self.export_button.setEnabled(True)
        self.replay_button.setEnabled(True)
        self.status.setText('語音已生成，正在播放；可重播或匯出 WAV。')
        if self.engine.last_metrics:
            metrics = self.engine.last_metrics
            elapsed = metrics.get('request_seconds', metrics['total_seconds'])
            self.status.setText(f"本次生成 {elapsed:.2f} 秒（含模型準備） · 音訊 {metrics['audio_seconds']:.2f} 秒；正在播放。")
        self.play(path)

        if self.pipeline is not None:
            from .core import DATA
            record = dict(self.pipeline, tts_seconds=time.perf_counter() - self._tts_started,
                          to_play_request_seconds=time.perf_counter() - self.pipeline['started'],
                          tts_details=self.engine.last_metrics)
            record.pop('started', None)
            self.status.setText(f"OCR {record.get('ocr_seconds') or 0:.2f} 秒 · TTS {record['tts_seconds']:.2f} 秒 · 合計 {record['to_play_request_seconds']:.2f} 秒；正在播放。")
            (DATA / 'last-pipeline.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            self.pipeline = None

    def play(self, path):
        if path:
            url = QUrl.fromLocalFile(str(path))
            # Force Qt/FFmpeg to reopen the WAV. This avoids a Windows backend
            # race when audiobook jobs replace a recently finished source.
            self.player.stop()
            self.player.setSource(QUrl())
            device = QMediaDevices.defaultAudioOutput()
            if not device.isNull() and self.audio.device() != device:
                self.audio.setDevice(device)
            self.player.setSource(url)
            self.player.play()
            log.info('play requested file=%s device=%s volume=%.2f', path,
                     self.audio.device().description(), self.audio.volume())
            QTimer.singleShot(200, lambda: self.player.play()
                              if self.player.source() == url
                              and self.player.mediaStatus() in (QMediaPlayer.LoadingMedia,
                                                                QMediaPlayer.LoadedMedia,
                                                                QMediaPlayer.BufferingMedia)
                              else None)

    def stop(self):
        if self.audiobook is not None:
            self.audiobook.set_enabled(False)
        if self._capture_after_job:
            self._capture_after_job = False
            self.capture_pending = False
        self.player.stop()
        self._pending_read = False
        if self.worker:
            self.worker.cancelled.set()
            self.status.setText('正在停止，會在目前載入或句子完成後結束…')
        else:
            self.status.setText('已停止播放。')

    def export(self):
        if self.output:
            path, _ = QFileDialog.getSaveFileName(self, '匯出語音', 'euphonia.wav', 'WAV (*.wav)')
            if path:
                if not path.lower().endswith('.wav'):
                    path += '.wav'
                try:
                    shutil.copy2(self.output, path)
                    self.status.setText('已匯出：' + path)
                except OSError as exc:
                    self.failed(str(exc))

    def closeEvent(self, event):
        # Closing the main window means closing the one Euphonia application.
        # Only the explicit background button hides this same instance to tray.
        self.exit_requested = True
        if self.thread is not None:
            self.stop()
            self.status.setText('正在結束背景工作，完成後即可關閉視窗。')
            event.ignore()
        else:
            if self.audiobook is not None:
                self.audiobook.shutdown()
                self.monitor_hotkey.stop()
            self.player.stop()
            self.capture_pending = False
            self.engine.close()
            if self.capture_frame is not None:
                self.capture_frame.close()
                self.capture_frame.deleteLater()
                self.capture_frame = None
            self.screenshot_hotkey.stop()
            if self.capture is not None:
                self.capture.close()
                self.capture.deleteLater()
                self.capture = None
            self.save_settings()
            if self.tray is not None:
                self.tray.hide()
            event.accept()
            QApplication.instance().quit()


STYLE = '''
QWidget {
    background: #071726;
    color: #eee8dc;
    font-family: "Microsoft JhengHei UI";
    font-size: 14px;
}
QWidget#appRoot { background: #061421; }
QWidget#brandHeader {
    background: #0a2745;
    border: 1px solid #8f6a2a;
    border-radius: 14px;
}
QLabel { background: transparent; }
QLabel#brandLogo { border: none; }
QLabel#eyebrow {
    color: #cda652;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#title {
    color: #f1d58b;
    font-family: "Palatino Linotype", "Georgia";
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 4px;
}
QLabel#subtitle { color: #c8d5df; font-size: 14px; }
QLabel#seal {
    color: #e3bd63;
    background: #071d34;
    border: 1px solid #9f7832;
    border-radius: 18px;
    padding: 10px 16px;
    font-family: "Palatino Linotype";
    font-size: 12px;
    font-weight: 700;
}
QWidget#sidebar, QWidget#workspace {
    background: #0a2036;
    border: 1px solid #29445b;
    border-radius: 12px;
}
QScrollArea#sidebarScroll { background: transparent; border: none; }
QLabel#sectionTitle {
    color: #d8b35a;
    border-bottom: 1px solid #5d4d2c;
    padding: 4px 0 7px 0;
    font-family: "Palatino Linotype", "Microsoft JhengHei UI";
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#voiceInfo { color: #aebfcb; padding: 4px 2px; }
QLabel#sidebarNote {
    color: #9fb0bc;
    background: #081a2c;
    border-left: 3px solid #b88a36;
    border-radius: 5px;
    padding: 10px;
}
QPushButton {
    background: #123451;
    color: #eee8dc;
    border: 1px solid #3b5870;
    border-radius: 7px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #19496f;
    border-color: #c9a24f;
    color: #fff7e5;
}
QPushButton:pressed { background: #0b2943; padding-top: 11px; padding-bottom: 9px; }
QPushButton#primary {
    background: #d5ad55;
    color: #10243a;
    border: 1px solid #f0d58f;
    font-weight: 800;
}
QPushButton#primary:hover { background: #e5c36f; border-color: #fff0bd; }
QPushButton#captureButton {
    background: #114f7a;
    border: 1px solid #d0aa54;
    color: #fff5dc;
}
QPushButton#dangerButton { color: #e7bbb0; border-color: #68413e; }
QPushButton#dangerButton:hover { background: #5b302f; border-color: #b87967; }
QPushButton:disabled { color: #64798a; background: #0b2236; border-color: #233b4e; }
QTextEdit, QLineEdit, QComboBox {
    background: #071a2d;
    color: #f0ece3;
    border: 1px solid #405b70;
    border-radius: 7px;
    padding: 9px;
    selection-background-color: #9c762f;
}
QTextEdit:focus, QLineEdit:focus, QComboBox:focus { border: 1px solid #d4ad55; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #0b2741;
    color: #eee8dc;
    border: 1px solid #9c7837;
    selection-background-color: #174c72;
    padding: 4px;
}
QTableWidget {
    background: #071a2d;
    alternate-background-color: #0b2338;
    color: #eee8dc;
    border: 1px solid #405b70;
    border-radius: 7px;
    gridline-color: #29445b;
    selection-background-color: #174c72;
}
QHeaderView::section {
    background: #11324f;
    color: #e2c474;
    border: none;
    border-right: 1px solid #29445b;
    border-bottom: 1px solid #8f7137;
    padding: 8px;
    font-weight: 700;
}
QMenu {
    background: #0b2741;
    color: #eee8dc;
    border: 1px solid #9c7837;
    padding: 5px;
}
QMenu::item { padding: 7px 24px 7px 12px; border-radius: 4px; }
QMenu::item:selected { background: #174c72; color: #fff5dc; }
QMenu::separator { height: 1px; background: #3b5264; margin: 5px 8px; }
QTextEdit { font-size: 16px; }
QCheckBox { color: #d9e1e5; spacing: 8px; padding: 3px 0; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #60798b;
    border-radius: 3px;
    background: #071a2d;
}
QCheckBox::indicator:checked { background: #d1a74d; border-color: #efd58e; }
QLabel#preview {
    background: #071a2d;
    border: 1px dashed #8f7137;
    border-radius: 10px;
    color: #9eb2c1;
    padding: 10px;
}
QLabel#monitorStatus {
    color: #c8d5dd;
    background: #0b2a45;
    border-radius: 6px;
    padding: 8px 10px;
}
QLabel#countLabel { color: #9faeba; font-size: 12px; padding-right: 4px; }
QLabel#statusBar {
    color: #e6d7ad;
    background: #0a2742;
    border: 1px solid #365269;
    border-left: 4px solid #d1a64c;
    border-radius: 7px;
    padding: 10px 12px;
}
QSplitter#mainSplit::handle { background: transparent; width: 12px; }
QToolTip {
    background: #102e49;
    color: #fff4d8;
    border: 1px solid #b58a39;
    padding: 6px;
}
QScrollBar:vertical { background: #081b2c; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #36556c; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
'''


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    instance_lock = QLockFile(str(DATA / 'euphonia-instance.lock'))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        log.info('Euphonia is already running; refusing a second instance')
        return
    # Keep the lock owned until QApplication and all background services exit.
    app._euphonia_instance_lock = instance_lock
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    sys.exit(app.exec())

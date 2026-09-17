"""Audiobook preferences using the application's existing voices and QSettings."""
import copy
import uuid
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)
from .audiobook import REGIONS, save_config


class AudiobookSettings(QDialog):
    select_region = Signal(str)

    def __init__(self, config, voices, settings, parent=None):
        super().__init__(parent)
        self.config = copy.deepcopy(config)
        self.voices = voices
        self.settings = settings
        self.setWindowTitle('Audiobook Mode · 有聲小說設定')
        self.resize(750, 650)
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox('啟用有聲小說模式（總開關）')
        self.enabled.setChecked(config['enabled'])
        layout.addWidget(self.enabled)
        hint = QLabel('F11 開關背景監控。角色共用主視窗選定的 TTS 模型及語言，Voice 各自指定。\n'
                      '請分別框選姓名與台詞；設定儲存後仍需按 F11 開始監控。\n'
                      '上方總開關與下方角色啟用是獨立設定，兩者都必須勾選。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.region_labels = {}
        for key, label in zip(REGIONS, ('角色名稱區域', '對話文字區域')):
            row = QHBoxLayout()
            self.region_labels[key] = QLabel()
            row.addWidget(self.region_labels[key], 1)
            button = QPushButton('框選區域')
            button.clicked.connect(lambda checked=False, k=key: self.select_region.emit(k))
            row.addWidget(button)
            form.addRow(label, row)
        self.interval = QSpinBox()
        self.interval.setRange(100, 5000)
        self.interval.setSuffix(' ms')
        self.interval.setValue(config['ocr_interval_ms'])
        form.addRow('OCR 間隔', self.interval)
        self.stable = QSpinBox()
        self.stable.setRange(300, 5000)
        self.stable.setSuffix(' ms')
        self.stable.setValue(config['stable_duration_ms'])
        form.addRow('文字穩定時間', self.stable)
        self.streaming_threshold = QDoubleSpinBox()
        self.streaming_threshold.setRange(-80, -10)
        self.streaming_threshold.setDecimals(0)
        self.streaming_threshold.setSingleStep(1)
        self.streaming_threshold.setSuffix(' dB')
        self.streaming_threshold.setValue(config['streaming_silence_threshold_db'])
        self.streaming_threshold.setToolTip(
            '相對每段峰值的靜音判定。越接近 -10 裁切越多；越接近 -80 保留越多微弱聲音。')
        form.addRow('Streaming 靜音閾值', self.streaming_threshold)
        self.streaming_gap = QSpinBox()
        self.streaming_gap.setRange(0, 1000)
        self.streaming_gap.setSingleStep(10)
        self.streaming_gap.setSuffix(' ms')
        self.streaming_gap.setValue(config['streaming_interval_ms'])
        self.streaming_gap.setToolTip('每個串流片段在有效聲音結束後保留的停頓時間。')
        form.addRow('Streaming 片段間隔', self.streaming_gap)
        self.fuzzy = QCheckBox('啟用模糊姓名匹配（短姓名只用正規化精確匹配）')
        self.fuzzy.setChecked(config['fuzzy_match'])
        form.addRow(self.fuzzy)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.75, 1)
        self.threshold.setSingleStep(.01)
        self.threshold.setValue(config['match_threshold'])
        form.addRow('姓名相似度門檻', self.threshold)
        layout.addLayout(form)
        layout.addWidget(QLabel('Character Voice Mapping · 雙擊角色名稱可編輯'))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['角色名稱', 'Voice', '啟用'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, 1)
        for character in config['characters']:
            self.add_character(character)
        row = QHBoxLayout()
        add = QPushButton('＋ 新增角色')
        add.clicked.connect(lambda: self.add_character())
        row.addWidget(add)
        delete = QPushButton('刪除所選角色')
        delete.clicked.connect(self.delete_selected)
        row.addWidget(delete)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('儲存')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_regions()

    def update_regions(self):
        for key, label in self.region_labels.items():
            r = self.config['ocr_regions'][key]
            label.setText(f"({r['x']}, {r['y']}) · {r['width']} × {r['height']}" if r else '尚未設定')

    def set_region(self, key, rect):
        self.config['ocr_regions'][key] = dict(x=rect.x(), y=rect.y(), width=rect.width(), height=rect.height())
        self.update_regions()

    def add_character(self, character=None):
        character = character or dict(id=uuid.uuid4().hex, name='', voice_id='', enabled=True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(character['name'])
        name.setData(Qt.UserRole, character['id'])
        self.table.setItem(row, 0, name)
        voice = QComboBox()
        for item in self.voices:
            voice.addItem(item['name'], item['id'])
        index = voice.findData(character['voice_id'])
        if character['voice_id'] and index < 0:
            voice.addItem('缺少 Voice：' + character['voice_id'], character['voice_id'])
            index = voice.count() - 1
        voice.setCurrentIndex(max(0, index))
        self.table.setCellWidget(row, 1, voice)
        enabled = QTableWidgetItem()
        enabled.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
        enabled.setCheckState(Qt.Checked if character['enabled'] else Qt.Unchecked)
        self.table.setItem(row, 2, enabled)

    def delete_selected(self):
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def save(self):
        value = copy.deepcopy(self.config)
        value.update(enabled=self.enabled.isChecked(), ocr_interval_ms=self.interval.value(),
                     stable_duration_ms=self.stable.value(), fuzzy_match=self.fuzzy.isChecked(),
                     match_threshold=self.threshold.value(),
                     streaming_silence_threshold_db=self.streaming_threshold.value(),
                     streaming_interval_ms=self.streaming_gap.value())
        value['characters'] = [dict(id=self.table.item(r, 0).data(Qt.UserRole),
            name=self.table.item(r, 0).text(), voice_id=self.table.cellWidget(r, 1).currentData(),
            enabled=self.table.item(r, 2).checkState() == Qt.Checked) for r in range(self.table.rowCount())]
        try:
            available = {v['id'] for v in self.voices}
            if any(c['enabled'] and c['voice_id'] not in available for c in value['characters']):
                raise ValueError('啟用的角色必須選擇仍存在的 Voice。')
            self.config = save_config(self.settings, value)
            self.accept()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, '無法儲存有聲小說設定', str(exc))

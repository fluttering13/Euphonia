"""Persistent desktop region with a clickable toolbar and an input-transparent hole."""
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QRegion
from PySide6.QtWidgets import QWidget, QPushButton, QHBoxLayout


class CaptureFrame(QWidget):
    requested = Signal()
    reselect = Signal()

    def __init__(self, screen, region, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.screen_source = screen
        self.region = QRect(region)  # Global logical desktop coordinates.
        self.toolbar = QWidget(self)
        self.toolbar.setStyleSheet('background:#182630;color:white;')
        layout = QHBoxLayout(self.toolbar)
        layout.setContentsMargins(4, 4, 4, 4)
        self.read_button = QPushButton('截圖朗讀')
        self.select_button = QPushButton('重新框選')
        close_button = QPushButton('關閉框框')
        for button in (self.read_button, self.select_button, close_button):
            button.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(button)
        self.read_button.clicked.connect(self.requested)
        self.select_button.clicked.connect(self.reselect)
        close_button.clicked.connect(self.hide)
        size = self.toolbar.sizeHint()
        bounds = screen.geometry()
        x = max(bounds.left(), min(region.left(), bounds.right() - size.width() + 1))
        y = region.top() - size.height() - 3
        if y < bounds.top():
            y = min(region.bottom() + 4, bounds.bottom() - size.height() + 1)
        toolbar_rect = QRect(x, y, size.width(), size.height())
        outer = region.adjusted(-3, -3, 3, 3)
        self.setGeometry(outer.united(toolbar_rect))
        self.inner = region.translated(-self.pos())
        self.toolbar.setGeometry(toolbar_rect.translated(-self.pos()))
        # A native window mask also lets mouse input through the selected area.
        ring = QRegion(outer.translated(-self.pos())).subtracted(QRegion(self.inner))
        self.setMask(ring.united(QRegion(self.toolbar.geometry())))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor('#58ddbe'), 3))
        painter.drawRect(self.inner.adjusted(-2, -2, 1, 1))

    def set_busy(self, busy):
        self.read_button.setEnabled(not busy)
        self.select_button.setEnabled(not busy)
        self.read_button.setText('處理中…' if busy else '截圖朗讀')

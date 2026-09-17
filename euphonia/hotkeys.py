"""Configurable Windows function-key hook; no keyboard content is stored."""
import ctypes
from ctypes import wintypes
import sys
import threading
import time
import logging

from PySide6.QtCore import QObject, QThread, Signal

log = logging.getLogger('euphonia.hotkey')


class KeyboardData(ctypes.Structure):
    _fields_ = [('vkCode', wintypes.DWORD), ('scanCode', wintypes.DWORD),
                ('flags', wintypes.DWORD), ('time', wintypes.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]


def user_api():
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetForegroundWindow.restype = wintypes.HWND
    api.SetForegroundWindow.argtypes = [wintypes.HWND]
    api.SetForegroundWindow.restype = wintypes.BOOL
    return api


def foreground_window():
    return user_api().GetForegroundWindow() if sys.platform == 'win32' else None


def activate_window(handle):
    if sys.platform == 'win32' and handle:
        return bool(user_api().SetForegroundWindow(handle))
    return False


class HookThread(QThread):
    pressed = Signal()
    ready = Signal()
    failed = Signal(str)

    def __init__(self, parent=None, *, virtual_key=0x7B):
        super().__init__(parent)
        self.stopping = threading.Event()
        self.virtual_key = virtual_key
        self.native_hook = None
        self.thread_id = None
        self.api = None
        self.down = False
        self.claimed = False
        self.last_down = 0.0
        self.last_trigger = -10.0
        self.polled_down = False

    def handle_key(self, vk, message, modified=False, now=None):
        if vk != self.virtual_key:
            return False
        if message in (0x100, 0x104):
            now = time.monotonic() if now is None else now
            # Recover a lost key-up. Normal held-key repeat keeps this fresh.
            if self.down and now - self.last_down > 1.5:
                self.down = self.claimed = False
            self.last_down = now
            if not self.down:
                self.down = True
                self.claimed = not modified
                if self.claimed:
                    self.last_trigger = now
                    self.pressed.emit()
            return self.claimed
        if message in (0x101, 0x105):
            claimed = self.claimed
            self.down = self.claimed = False
            return claimed
        return False

    def poll_key(self, down, modified=False, now=None):
        """Physical-state backup when Windows silently removes the hook."""
        now = time.monotonic() if now is None else now
        edge = down and not self.polled_down
        self.polled_down = down
        if edge and not modified and now - self.last_trigger > 0.12:
            self.last_trigger = now
            self.pressed.emit()
            return True
        return False

    def run(self):
        api = self.api = user_api()
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentThreadId.restype = wintypes.DWORD
        kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel.GetModuleHandleW.restype = wintypes.HMODULE
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)
        api.SetWindowsHookExW.argtypes = [ctypes.c_int, callback_type, wintypes.HINSTANCE, wintypes.DWORD]
        api.SetWindowsHookExW.restype = wintypes.HANDLE
        api.CallNextHookEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t]
        api.CallNextHookEx.restype = ctypes.c_ssize_t
        api.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]
        api.GetAsyncKeyState.argtypes = [ctypes.c_int]
        api.GetAsyncKeyState.restype = ctypes.c_short
        api.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        api.GetMessageW.restype = wintypes.BOOL
        api.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        api.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        api.DispatchMessageW.restype = ctypes.c_ssize_t
        api.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
        api.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t]

        def callback(code, message, pointer):
            if code >= 0:
                data = ctypes.cast(pointer, ctypes.POINTER(KeyboardData)).contents
                if data.vkCode == self.virtual_key:
                    modified = any(api.GetAsyncKeyState(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C))
                    if self.handle_key(data.vkCode, message, modified):
                        return 1
            return api.CallNextHookEx(None, code, message, pointer)

        # Keep the callback alive until after the native hook is removed.
        native_callback = callback_type(callback)
        msg = wintypes.MSG()
        api.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        self.thread_id = kernel.GetCurrentThreadId()
        hook = api.SetWindowsHookExW(13, native_callback, kernel.GetModuleHandleW(None), 0)
        self.native_hook = hook
        if not hook:
            self.failed.emit(f'F{self.virtual_key - 0x6F} 鍵盤 hook 無法啟動（Windows 錯誤 {ctypes.get_last_error()}），已使用按鍵狀態備援。')
        self.down = self.claimed = self.polled_down = False
        recover = not bool(hook)
        try:
            log.info('listener started; hook=%s', bool(hook))
            self.ready.emit()
            while not self.stopping.is_set():
                # Pump callbacks and poll outside the hook, after Windows updates
                # async state. Use only the high bit; the low bit is not reliable.
                for _ in range(64):
                    if not api.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                        break
                    if msg.message == 0x12:
                        self.stopping.set()
                        break
                    api.TranslateMessage(ctypes.byref(msg))
                    api.DispatchMessageW(ctypes.byref(msg))
                down = bool(api.GetAsyncKeyState(self.virtual_key) & 0x8000)
                modified = any(api.GetAsyncKeyState(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C))
                if self.poll_key(down, modified):
                    log.warning('key=%s received through fallback; scheduling hook recovery', self.virtual_key)
                    recover = True
                if recover and not down:
                    replacement = api.SetWindowsHookExW(13, native_callback, kernel.GetModuleHandleW(None), 0)
                    if replacement:
                        if hook:
                            api.UnhookWindowsHookEx(hook)
                        hook = replacement
                        self.native_hook = hook
                        self.down = self.claimed = False
                        log.info('keyboard hook recovered')
                    recover = False
                self.stopping.wait(0.008)
        finally:
            if hook:
                api.UnhookWindowsHookEx(hook)
            log.info('listener stopped')
            self.native_hook = None
            self.thread_id = None

    def stop(self):
        self.stopping.set()
        if self.api is not None and self.thread_id is not None:
            self.api.PostThreadMessageW(self.thread_id, 0x12, 0, 0)
        self.wait()


class ScreenshotHotkey(QObject):
    pressed = Signal()
    ready = Signal()
    failed = Signal(str)

    def __init__(self, parent=None, *, virtual_key=0x7B):
        super().__init__(parent)
        self.listener = HookThread(self, virtual_key=virtual_key)
        self.listener.pressed.connect(self.pressed)
        self.listener.ready.connect(self.ready)
        self.listener.failed.connect(self.failed)

    def start(self):
        if sys.platform == 'win32' and not self.listener.isRunning():
            self.listener.stopping.clear()
            self.listener.start()

    def stop(self):
        self.listener.stop()

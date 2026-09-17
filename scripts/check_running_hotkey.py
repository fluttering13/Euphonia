"""Check F12 against an already running app; cancel without capturing content."""
import ctypes
from ctypes import wintypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_background import send_f12


def main():
    pid = int(sys.argv[1])
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    api.IsWindowVisible.argtypes = [wintypes.HWND]
    api.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    api.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    def windows():
        found = []
        def visit(hwnd, _):
            owner = wintypes.DWORD()
            api.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid:
                title = ctypes.create_unicode_buffer(512)
                api.GetWindowTextW(hwnd, title, len(title))
                rect = wintypes.RECT()
                api.GetWindowRect(hwnd, ctypes.byref(rect))
                found.append(dict(hwnd=hwnd, title=title.value, width=rect.right-rect.left,
                                  height=rect.bottom-rect.top, visible=bool(api.IsWindowVisible(hwnd))))
            return True
        api.EnumWindows(callback_type(visit), 0)
        return found
    initial = windows()
    main_window = next((w for w in initial if w['title'].startswith('Euphonia')), None)
    if main_window is None:
        raise RuntimeError(f'No running main window found for PID {pid}: {initial}')
    for w in initial:
        if w['visible'] and w['hwnd'] != main_window['hwnd'] and w['width'] > 500 and w['height'] > 400:
            api.PostMessageW(w['hwnd'], 0x100, 0x1B, 1)
    time.sleep(0.3)
    api.ShowWindow(main_window['hwnd'], 6)
    time.sleep(0.5)
    send_f12()
    send_f12(up=True)
    time.sleep(1)
    after = windows()
    overlays = [w for w in after if w['visible'] and w['hwnd'] != main_window['hwnd'] and w['width'] > 500 and w['height'] > 400]
    try:
        assert overlays, f'F12 did not open capture overlay: {after}'
    finally:
        # Escape cancels the selection; no screenshot is saved or OCR/TTS requested.
        for w in overlays:
            api.PostMessageW(w['hwnd'], 0x100, 0x1B, 1)
    time.sleep(0.3)
    print(json.dumps(dict(pid=pid, background_f12=bool(overlays), overlay_closed=not any(
        w['visible'] and w['hwnd'] in {o['hwnd'] for o in overlays} for w in windows()))))


if __name__ == '__main__':
    main()

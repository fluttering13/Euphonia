"""Native hook removal/recovery test, using F24 to leave the user's F12 alone."""
import ctypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QCoreApplication
from euphonia.hotkeys import HookThread


def main():
    app = QCoreApplication([])
    hook = HookThread(virtual_key=0x87)
    events, ready = [], []
    hook.pressed.connect(lambda: events.append(time.monotonic()))
    hook.ready.connect(lambda: ready.append(True))
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_size_t]
    api.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
    def pump(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(0.005)
    def tap():
        api.keybd_event(0x87, 0, 0, 0)
        pump(0.08)
        api.keybd_event(0x87, 0, 2, 0)
        pump(0.2)
    hook.start()
    try:
        pump(0.3)
        assert ready and hook.native_hook, 'Hook did not start'
        tap()
        assert len(events) == 1, f'Normal press count: {len(events)}'
        old = hook.native_hook
        assert api.UnhookWindowsHookEx(old), 'Cannot simulate hook removal'
        tap()
        assert len(events) == 2, f'Fallback press count: {len(events)}'
        assert hook.native_hook and hook.native_hook != old, 'Hook was not replaced'
        tap()
        assert len(events) == 3, f'Recovered press count: {len(events)}'
        result = dict(normal_press=True, removed_hook_fallback=True, hook_reinstalled=True,
                      recovered_press=True, duplicate_events=False, test_key='F24')
        destination = Path(__file__).resolve().parents[1] / 'data/hotkey-recovery-test.json'
        destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        api.keybd_event(0x87, 0, 2, 0)
        hook.stop()


if __name__ == '__main__':
    main()

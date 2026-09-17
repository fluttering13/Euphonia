import unittest
from euphonia.hotkeys import HookThread


class HotkeyTests(unittest.TestCase):
    def test_fallback_edges_and_deduplication(self):
        hook = HookThread()
        events = []
        hook.pressed.connect(lambda: events.append(True))
        hook.handle_key(0x7B, 0x100, now=10)
        self.assertFalse(hook.poll_key(True, now=10.01))
        hook.poll_key(False, now=10.1)
        self.assertTrue(hook.poll_key(True, now=11))
        self.assertFalse(hook.poll_key(True, now=12))
        hook.poll_key(False, now=13)
        self.assertFalse(hook.poll_key(True, modified=True, now=14))
        self.assertEqual(len(events), 2)

    def test_missing_key_up_recovers_without_repeating_held_key(self):
        hook = HookThread()
        events = []
        hook.pressed.connect(lambda: events.append(True))
        for now in (10, 11, 11.2, 11.4):
            hook.handle_key(0x7B, 0x100, now=now)
        self.assertEqual(len(events), 1)
        hook.handle_key(0x7B, 0x100, now=15)
        self.assertEqual(len(events), 2)

    def test_f12_edges_and_modifier_combinations(self):
        hook = HookThread()
        events = []
        hook.pressed.connect(lambda: events.append('capture'))
        self.assertFalse(hook.handle_key(0x41, 0x100))
        self.assertTrue(hook.handle_key(0x7B, 0x100))
        self.assertTrue(hook.handle_key(0x7B, 0x100))
        self.assertEqual(events, ['capture'])
        self.assertTrue(hook.handle_key(0x7B, 0x101))
        self.assertFalse(hook.handle_key(0x7B, 0x100, modified=True))
        self.assertFalse(hook.handle_key(0x7B, 0x101))
        self.assertTrue(hook.handle_key(0x7B, 0x100))
        self.assertEqual(events, ['capture', 'capture'])

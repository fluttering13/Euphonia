import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
from PySide6.QtCore import QPoint, QRect, QSettings, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication
from euphonia.audiobook import (DialogueState, StreamingDialogueState, defaults, load_config, match_character,
                               normalize, save_config, speech_text, validate_config)
from euphonia.audiobook_monitor import OCRWorker, capture_regions
from euphonia.audiobook_overlay import AudiobookEmotionPanel, AudiobookReadButton
from euphonia.audiobook_settings import AudiobookSettings
from euphonia.app import Window
from euphonia.hotkeys import HookThread

app = QApplication.instance() or QApplication([])


def config():
    value = defaults()
    value.update(enabled=True, ocr_interval_ms=100, stable_duration_ms=300)
    value['characters'] = [dict(id='alice', name='Alice', voice_id='voice-a', enabled=True),
                           dict(id='bob', name='Bob', voice_id='voice-b', enabled=True)]
    value['ocr_regions'] = dict(character_name=dict(x=10, y=10, width=80, height=30),
                                dialogue_text=dict(x=10, y=50, width=200, height=60))
    return value


def spin(condition, timeout=3):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    if not condition():
        raise AssertionError('Qt condition timed out')


class StateTests(unittest.TestCase):
    def test_streaming_commits_punctuation_then_stable_tail_once(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        self.assertEqual(state.observe(alice, 'Hello', 0), [])
        first = state.observe(alice, 'Hello world. More', .2)
        self.assertEqual([item['text'] for item in first], ['Hello world.'])
        self.assertEqual(state.observe(alice, 'Hello world. More text', .4), [])
        self.assertEqual(state.observe(alice, 'Hello world. More text', .7), [])
        tail = state.observe(alice, 'Hello world. More text', 1.0)
        self.assertEqual([item['text'] for item in tail], ['More text'])
        self.assertEqual(state.observe(alice, 'Hello world. More text', 2), [])

    def test_streaming_commits_multiple_new_sentences_in_order(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        items = state.observe(alice, '第一段，第二句。還在輸入', 0)
        self.assertEqual([item['text'] for item in items], ['第一段，', '第二句。'])

    def test_streaming_late_growth_does_not_replay_finished_prefix(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        self.assertEqual([x['text'] for x in state.observe(alice, 'Hello.', 0)], ['Hello.'])
        self.assertEqual(state.observe(alice, 'Hello.', .6), [])
        self.assertEqual(state.observe(alice, 'Hello. New', 1.0), [])
        items = state.observe(alice, 'Hello. New sentence.', 1.2)
        self.assertEqual([item['text'] for item in items], ['New sentence.'])

    def test_streaming_allowed_punctuation_boundaries(self):
        alice = config()['characters'][0]
        for punctuation in (',', '，', '.', '．', '。', '!', '！', '?', '？', ';', '；'):
            with self.subTest(punctuation=punctuation):
                state = StreamingDialogueState(500)
                items = state.observe(alice, 'first' + punctuation + 'second', 0)
                self.assertEqual([item['text'] for item in items], ['first' + punctuation])

    def test_streaming_newline_is_not_a_boundary(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        self.assertEqual(state.observe(alice, 'first\nsecond', 0), [])

    def test_streaming_decimal_dot_is_not_a_boundary(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        self.assertEqual(state.observe(alice, 'Value 3.14 is exact', 0), [])

    def test_streaming_titles_and_abbreviations_do_not_split(self):
        alice = config()['characters'][0]
        examples = (
            ('Ms.', 'Ms. Rhiannon,', 'Ms. Rhiannon,'),
            ('Mr．', 'Mr． Stranger，', 'Mr． Stranger，'),
            ('Dr.', 'Dr. Watson.', 'Dr. Watson.'),
            ('A.', 'A. Smith;', 'A. Smith;'),
            ('e.g.', 'e.g. apples,', 'e.g. apples,'),
            ('i.e.', 'i.e. exactly;', 'i.e. exactly;'),
            ('U.S.', 'U.S. forces.', 'U.S. forces.'),
        )
        for initial, completed, expected in examples:
            with self.subTest(initial=initial):
                state = StreamingDialogueState(500)
                self.assertEqual(state.observe(alice, initial, 0), [])
                items = state.observe(alice, completed, .2)
                self.assertEqual([item['text'] for item in items], [expected])

    def test_streaming_real_period_still_splits(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        items = state.observe(alice, 'This is complete. Next', 0)
        self.assertEqual([item['text'] for item in items], ['This is complete.'])

    def test_streaming_interval_flushes_pending_safe_text(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        items = state.observe(alice, 'First,second word grow', 0)
        self.assertEqual([item['text'] for item in items], ['First,'])
        self.assertEqual(state.flush_pending_for_interval()['text'], 'second word')
        self.assertIsNone(state.flush_pending_for_interval())
        state.observe(alice, 'First,second word growing now', .2)
        self.assertEqual(state.flush_pending_for_interval()['text'], 'growing')

        cjk = StreamingDialogueState(500)
        cjk.observe(alice, '第一段，後續文字還在跑', 0)
        self.assertEqual(cjk.flush_pending_for_interval()['text'], '後續文字還在跑')

    def test_streaming_manual_commit_is_immediate_and_suppresses_replay(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        item = state.commit_immediately(alice, 'Read this now.', 1)
        self.assertEqual(item['text'], 'Read this now.')
        self.assertEqual(state.observe(alice, 'Read this now.', 2), [])

    def test_streaming_quotes_are_text_changes_not_boundaries(self):
        alice = config()['characters'][0]
        for quote in ('"', "'", '“', '”', '‘', '’'):
            with self.subTest(quote=quote):
                state = StreamingDialogueState(500)
                self.assertEqual(state.observe(alice, 'He said', 0), [])
                self.assertEqual(state.observe(alice, 'He said', .4), [])
                # A quote resets the stability timer but emits no segment.
                text = 'He said' + quote
                self.assertEqual(state.observe(alice, text, .49), [])
                self.assertEqual(state.observe(alice, text, .8), [])
                items = state.observe(alice, text, 1.0)
                self.assertEqual([item['text'] for item in items], ['He said'])

    def test_streaming_apostrophe_inside_word_never_splits(self):
        state = StreamingDialogueState(500)
        alice = config()['characters'][0]
        self.assertEqual(state.observe(alice, "I don'", 0), [])
        self.assertEqual(state.observe(alice, "I don't want", .2), [])
        self.assertEqual(state.observe(alice, "I don't want", .5), [])
        items = state.observe(alice, "I don't want", .8)
        self.assertEqual([item['text'] for item in items], ['I dont want'])

    def test_quotes_are_removed_only_from_spoken_text(self):
        self.assertEqual(speech_text('\"She said \'hello\'.\"'), 'She said hello.')
        self.assertEqual(speech_text('“你好，‘旅人’。”'), '你好，旅人。')
        self.assertEqual(speech_text('「＂測試＇」'), '測試')
        alice = config()['characters'][0]
        state = DialogueState(500)
        item = state.commit_immediately(alice, '\"Read this.\"', 0)
        self.assertEqual(item['text'], 'Read this.')

    def test_normalization_matching_disabled_short_and_ambiguous(self):
        c = config()
        for name in ('Alice', 'ALICE', 'Alice：', 'Alice:', ' Alice ', 'Ａｌｉｃｅ'):
            self.assertEqual(match_character(name, c)['id'], 'alice')
        self.assertIsNone(match_character('Unknown NPC', c))
        self.assertIsNone(match_character('Bod', c))
        c['match_threshold'] = .85
        self.assertEqual(match_character('A1ice', c), None)  # Too dissimilar.
        self.assertEqual(match_character('Alicee', c)['id'], 'alice')
        c['characters'].append(dict(id='other', name='Alicec', voice_id='c', enabled=True))
        self.assertIsNone(match_character('Aliced', c))
        c['characters'][0]['enabled'] = False
        self.assertIsNone(match_character('ALICE:', c))
        c['fuzzy_match'] = False
        self.assertIsNone(match_character('Alicee', c))
        self.assertEqual(normalize(' 愛麗絲： '), '愛麗絲')

    def test_typewriter_only_commits_full_line_once_for_ten_seconds(self):
        state = DialogueState(700)
        alice = config()['characters'][0]
        for i, text in enumerate(('你', '你好', '你好，', '你好，冒險者。')):
            self.assertIsNone(state.observe(alice, text, i * .2))
        self.assertIsNone(state.observe(alice, '你好，冒險者。', 1.0))
        item = state.observe(alice, '你好，冒險者。', 1.31)
        self.assertEqual(item['text'], '你好，冒險者。')
        for now in range(2, 50):
            self.assertIsNone(state.observe(alice, '你好，冒險者！', now))

    def test_character_switch_and_expiring_history(self):
        s = DialogueState(500)
        a, b = config()['characters']
        for actor, text, now in ((a, 'Hello.', 0), (b, 'Welcome.', 2), (a, 'Let us go.', 4)):
            self.assertIsNone(s.observe(actor, text, now))
            self.assertEqual(s.observe(actor, text, now + .6)['voice_id'], actor['voice_id'])
        s.observe(a, 'Hello.', 6)
        self.assertIsNone(s.observe(a, 'Hello.', 6.6))
        s.observe(b, 'A different scene.', 34)
        self.assertIsNotNone(s.observe(b, 'A different scene.', 34.6))
        s.observe(a, 'Hello.', 36)
        self.assertIsNotNone(s.observe(a, 'Hello.', 36.6))

    def test_manual_commit_is_immediate_and_suppresses_automatic_replay(self):
        s = DialogueState(700)
        alice = config()['characters'][0]
        item = s.commit_immediately(alice, 'Read this now.', 1)
        self.assertEqual(item['text'], 'Read this now.')
        self.assertIsNone(s.observe(alice, 'Read this now.', 2))

    def test_jitter_empty_invalid_and_late_growth(self):
        s = DialogueState(500)
        a = config()['characters'][0]
        for text, now in [('Hello traveller.', 0), ('Hello trave1ler.', .3), ('Hello traveller.', .4)]:
            self.assertIsNone(s.observe(a, text, now))
        self.assertIsNotNone(s.observe(a, 'Hello traveller.', 1))
        for text in ('Hello trave1ler.', 'Hello traveller. Welcome!', 'Hello traveller.'):
            self.assertIsNone(s.observe(a, text, 2))
        for text in ('', '！？', 'x' * 5001):
            self.assertIsNone(s.observe(a, text, 3))
        self.assertIsNone(s.observe(None, 'A valid line.', 4))

    def test_settings_roundtrip_validation_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'settings.ini')
            c = config()
            settings = QSettings(path, QSettings.IniFormat)
            save_config(settings, c)
            self.assertEqual(load_config(QSettings(path, QSettings.IniFormat)), c)
            broken = copy.deepcopy(c)
            broken['characters'][1]['name'] = 'ALICE:'
            with self.assertRaises(ValueError):
                validate_config(broken)
            broken = copy.deepcopy(c)
            broken['streaming_silence_threshold_db'] = -100
            with self.assertRaises(ValueError):
                validate_config(broken)
            broken = copy.deepcopy(c)
            broken['streaming_interval_ms'] = 1001
            with self.assertRaises(ValueError):
                validate_config(broken)
            settings.setValue('audiobook_mode', '{broken')
            with self.assertLogs('euphonia.audiobook', level='ERROR'):
                self.assertFalse(load_config(settings)['enabled'])
            self.assertEqual(settings.value('audiobook_mode'), '{broken')

    def test_unknown_character_does_not_ocr_dialogue(self):
        engine = Mock()
        engine.recognize.return_value = 'Unknown'
        worker = OCRWorker(engine, threading.Event())
        results = []
        worker.result.connect(results.append)
        worker.scan((1, config(), dict(character_name='name-image', dialogue_text='dialogue-image')))
        engine.recognize.assert_called_once_with('name-image')
        self.assertIsNone(results[0]['character'])

    def test_two_regions_share_snapshot_and_scale_at_high_dpi(self):
        class Screen:
            def __init__(self):
                self.calls = 0
            def geometry(self):
                return QRect(-200, 0, 200, 150)
            def grabWindow(self, _):
                self.calls += 1
                p = QPixmap(400, 300)
                p.setDevicePixelRatio(2)
                p.fill(Qt.white)
                return p
        screen = Screen()
        regions = dict(character_name=dict(x=-180, y=10, width=40, height=20),
                       dialogue_text=dict(x=-180, y=50, width=100, height=40))
        images = capture_regions(regions, [screen])
        self.assertEqual(screen.calls, 1)
        self.assertEqual(images['character_name'].shape, (40, 80, 3))
        self.assertEqual(images['dialogue_text'].shape, (80, 200, 3))
        regions['character_name']['x'] = 200
        with self.assertRaises(ValueError):
            capture_regions(regions, [screen])

    def test_f11_hold_and_f12_are_independent(self):
        hook = HookThread(virtual_key=0x7A)
        events = []
        hook.pressed.connect(lambda: events.append(True))
        self.assertFalse(hook.handle_key(0x7B, 0x100, now=1))
        hook.handle_key(0x7A, 0x100, now=2)
        hook.handle_key(0x7A, 0x100, now=2.1)
        hook.poll_key(True, now=2.11)
        self.assertEqual(len(events), 1)
        hook.handle_key(0x7A, 0x101)
        hook.poll_key(False, now=2.2)
        hook.handle_key(0x7A, 0x100, now=3)
        self.assertEqual(len(events), 2)


class MonitorTests(unittest.TestCase):
    def test_emotion_panel_is_left_of_manual_ocr_button(self):
        region = dict(x=500, y=200, width=260, height=80)
        button = AudiobookReadButton(region, lambda: None)
        panel = AudiobookEmotionPanel(region, button)
        self.addCleanup(button.deleteLater)
        self.addCleanup(panel.deleteLater)
        self.assertLess(panel.geometry().right(), button.geometry().left())
        self.assertEqual(panel.geometry().bottom(), button.geometry().bottom())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = QSettings(str(Path(self.temp.name) / 'settings.ini'), QSettings.IniFormat)
        with patch('euphonia.app.QSettings', return_value=self.settings), patch('euphonia.app.ScreenshotHotkey.start'):
            self.window = Window(auto_preload=False)
        self.monitor = self.window.audiobook
        self.monitor.config = config()
        self.window.store.list = Mock(return_value=[dict(id='voice-a', name='A', transcript='A'),
                                                   dict(id='voice-b', name='B', transcript='B')])
        self.window.play = Mock()

    def tearDown(self):
        self.monitor.set_enabled(False)
        spin(lambda: self.window.thread is None)
        self.monitor.shutdown()
        self.window.close()
        app.processEvents()
        self.temp.cleanup()

    def test_f11_toggle_single_worker_and_stale_result(self):
        gate = threading.Event()
        entered = threading.Event()
        def recognize(image):
            entered.set()
            gate.wait(2)
            return 'Alice'
        self.window.engine.recognize = Mock(side_effect=recognize)
        with patch('euphonia.audiobook_monitor.capture_regions', return_value=dict(character_name=1, dialogue_text=2)):
            self.window.monitor_hotkey.pressed.emit()
            old = self.monitor.thread
            self.monitor.tick()
            spin(entered.is_set)
            for _ in range(5):
                self.monitor.tick()
                self.monitor.set_enabled(True)
            self.assertIs(self.monitor.thread, old)
            self.assertEqual(self.window.engine.recognize.call_count, 1)
            self.window.monitor_hotkey.pressed.emit()
            self.assertFalse(self.monitor.enabled)
            self.assertFalse(self.monitor.timer.isActive())
            self.window.monitor_hotkey.pressed.emit()
            self.assertIs(self.monitor.thread, old)
            gate.set()
            spin(lambda: self.monitor.thread is not old)
            self.assertFalse(self.window.play.called)
            self.assertEqual(len(self.monitor.queue), 0)

    def test_voice_routing_queue_and_cancellation(self):
        self.monitor.set_enabled(True)
        self.monitor.timer.stop()
        calls = []
        def synth(voice, text, language, progress, cancelled):
            calls.append((voice['id'], text))
            return 'test.wav'
        self.window.engine.synthesize = synth
        self.monitor.queue.extend([dict(character_name='Alice', voice_id='voice-a', text='Hello'),
                                   dict(character_name='Bob', voice_id='voice-b', text='Welcome')])
        self.window.engine.backend = 'qwen_streaming'
        self.monitor.pump()
        spin(lambda: len(calls) == 2 and self.window.thread is None)
        self.assertEqual(calls, [('voice-a', 'Hello'), ('voice-b', 'Welcome')])
        self.assertTrue(self.monitor.playing)
        self.assertEqual(self.window.play.call_count, 1)
        self.assertEqual(len(self.monitor.ready_audio), 1)
        self.monitor.media_status(QMediaPlayer.EndOfMedia)
        self.assertEqual(self.window.play.call_count, 2)
        self.assertTrue(self.monitor.playing)
        self.assertEqual(len(self.monitor.ready_audio), 0)
        self.monitor.set_enabled(False)
        self.monitor.set_enabled(True)
        self.monitor.timer.stop()
        def slow(voice, text, language, progress, cancelled):
            while not cancelled():
                time.sleep(.005)
            return 'late.wav'
        self.window.engine.synthesize = slow
        self.monitor.queue.append(dict(character_name='Alice', voice_id='voice-a', text='Late'))
        self.monitor.pump()
        self.monitor.set_enabled(False)
        spin(lambda: self.window.thread is None)
        self.assertEqual(self.window.play.call_count, 2)
        self.assertFalse(self.monitor.queue)

    def test_playback_end_flushes_text_that_is_still_growing(self):
        self.window.engine.set_backend('qwen_streaming')
        calls = []
        def synth(voice, text, language, progress, cancelled):
            calls.append(text)
            return f'part-{len(calls)}.wav'
        self.window.engine.synthesize = synth
        self.monitor.set_enabled(True)
        self.monitor.timer.stop()
        alice = config()['characters'][0]
        self.monitor.queue.extend(
            self.monitor.state.observe(alice, 'First,second word grow', 0))
        self.monitor.pump()
        spin(lambda: self.window.thread is None and self.monitor.playing)
        self.assertEqual(calls, ['First,'])
        self.monitor.media_status(QMediaPlayer.EndOfMedia)
        spin(lambda: len(calls) == 2 and self.window.thread is None)
        self.assertEqual(calls, ['First,', 'second word'])
        self.assertEqual(self.window.play.call_count, 2)

    def test_overlay_button_immediately_ocrs_and_plays(self):
        self.window.engine.recognize = Mock(return_value='Read this immediately.')
        self.window.engine.synthesize = Mock(return_value='manual.wav')
        with patch('euphonia.audiobook_monitor.capture_regions',
                   return_value=dict(dialogue_text=2)) as capture:
            self.monitor.set_enabled(True)
            self.monitor.timer.stop()
            self.assertIsNotNone(self.monitor.overlays.read_button)
            self.assertIsNotNone(self.monitor.overlays.emotion_panel)
            self.assertFalse(hasattr(self.monitor.overlays, 'text_panel'))
            self.monitor.overlays.read_button.click()
            spin(lambda: self.window.play.called and self.window.thread is None)
        self.assertEqual(self.window.editor.toPlainText(), 'Read this immediately.')
        capture.assert_called_once_with(self.monitor.config['ocr_regions'], QApplication.screens(),
                                        ('dialogue_text',))
        self.window.engine.recognize.assert_called_once_with(2)
        self.window.engine.synthesize.assert_called_once()
        self.window.play.assert_called_once_with('manual.wav')
        self.assertIn('Sampling：主要參考音檔', self.monitor.overlays.emotion_panel.text())

    def test_overlay_button_does_not_require_character_name_ocr(self):
        self.window.engine.recognize = Mock(return_value='Dialogue remains visible.')
        self.window.engine.synthesize = Mock(return_value='dialogue-only.wav')
        with patch('euphonia.audiobook_monitor.capture_regions', return_value=dict(dialogue_text=2)):
            self.monitor.set_enabled(True)
            self.monitor.timer.stop()
            self.monitor.overlays.read_button.click()
            spin(lambda: self.window.play.called and self.window.thread is None)
        self.assertEqual(self.window.editor.toPlainText(), 'Dialogue remains visible.')
        self.window.engine.recognize.assert_called_once_with(2)
        self.window.play.assert_called_once_with('dialogue-only.wav')

    def test_streaming_overlay_button_finishes_recognition_and_plays(self):
        self.window.engine.set_backend('qwen_streaming')
        self.window.engine.recognize = Mock(return_value='Streaming button dialogue.')
        self.window.engine.synthesize = Mock(return_value='streaming-manual.wav')
        with patch('euphonia.audiobook_monitor.capture_regions', return_value=dict(dialogue_text=2)):
            self.monitor.set_enabled(True)
            self.monitor.timer.stop()
            self.assertIsInstance(self.monitor.state, StreamingDialogueState)
            self.monitor.overlays.read_button.click()
            spin(lambda: self.window.play.called and self.window.thread is None)
        self.assertFalse(self.monitor.inflight)
        self.assertTrue(self.monitor.playing)
        self.assertFalse(self.monitor.overlays.read_button.isEnabled())
        self.assertEqual(self.monitor.overlays.read_button.text(), '播放中…')
        self.window.play.assert_called_once_with('streaming-manual.wav')
        self.monitor.media_status(QMediaPlayer.EndOfMedia)
        self.assertTrue(self.monitor.overlays.read_button.isEnabled())
        self.assertEqual(self.monitor.overlays.read_button.text(), 'OCR ＋ 播放')

    def test_manual_result_processing_error_never_leaves_button_stuck(self):
        self.monitor.set_enabled(True)
        self.monitor.timer.stop()
        self.monitor.inflight = True
        self.monitor.overlays.set_action_busy(True)
        self.monitor.state.commit_immediately = Mock(side_effect=RuntimeError('broken state'))
        character = self.monitor.manual_character()
        with self.assertLogs('euphonia.audiobook', level='ERROR'):
            self.monitor.observed(dict(epoch=self.monitor.epoch, error=None, manual=True,
                                       character=character, text='Hello.', seconds=.01))
        self.assertFalse(self.monitor.inflight)
        self.assertTrue(self.monitor.overlays.read_button.isEnabled())
        self.assertEqual(self.monitor.overlays.read_button.text(), 'OCR ＋ 播放')
        self.assertIn('文字處理錯誤', self.window.monitor_status.text())

    def test_settings_dialog_edit_add_delete_and_restart(self):
        dialog = AudiobookSettings(config(), self.window.store.list(), self.settings)
        dialog.streaming_threshold.setValue(-45)
        dialog.streaming_gap.setValue(40)
        dialog.table.item(0, 0).setText('Alicia')
        dialog.table.cellWidget(0, 1).setCurrentIndex(1)
        dialog.table.item(1, 2).setCheckState(Qt.Unchecked)
        dialog.add_character(dict(id='c', name='Carol', voice_id='voice-a', enabled=True))
        dialog.table.selectRow(2)
        dialog.delete_selected()
        dialog.save()
        saved = load_config(QSettings(self.settings.fileName(), QSettings.IniFormat))
        self.assertEqual(len(saved['characters']), 2)
        self.assertEqual(saved['characters'][0]['name'], 'Alicia')
        self.assertEqual(saved['characters'][0]['voice_id'], 'voice-b')
        self.assertFalse(saved['characters'][1]['enabled'])
        self.assertEqual(saved['streaming_silence_threshold_db'], -45)
        self.assertEqual(saved['streaming_interval_ms'], 40)
        dialog.close()

    def test_region_selection_reuses_capture_and_cancel_restores_dialog(self):
        outcomes = []
        errors = []
        self.window.engine.recognize = Mock()
        self.window.failed = errors.append
        def interact():
            dialog = next(w for w in app.topLevelWidgets() if isinstance(w, AudiobookSettings))
            original = copy.deepcopy(dialog.config['ocr_regions'])
            dialog.select_region.emit('character_name')
            def choose():
                try:
                    self.assertIsNotNone(self.window.capture)
                    overlay = self.window.capture
                    QTest.mousePress(overlay, Qt.LeftButton, pos=QPoint(30, 40))
                    QTest.mouseMove(overlay, QPoint(160, 100))
                    QTest.mouseRelease(overlay, Qt.LeftButton, pos=QPoint(160, 100))
                    self.assertTrue(dialog.isVisible())
                    self.assertEqual(dialog.config['ocr_regions']['character_name']['width'], 131)
                    self.assertEqual(dialog.config['ocr_regions']['dialogue_text'], original['dialogue_text'])
                    dialog.select_region.emit('dialogue_text')
                    QTimer.singleShot(400, cancel)
                except Exception as exc:
                    errors.append(str(exc))
                    dialog.reject()
            def cancel():
                try:
                    QTest.keyClick(self.window.capture, Qt.Key_Escape)
                    self.assertTrue(dialog.isVisible())
                    self.assertEqual(dialog.config['ocr_regions']['dialogue_text'], original['dialogue_text'])
                    self.window.engine.recognize.assert_not_called()
                    outcomes.append(True)
                except Exception as exc:
                    errors.append(str(exc))
                finally:
                    dialog.reject()
            QTimer.singleShot(400, choose)
        QTimer.singleShot(0, interact)
        self.window.open_audiobook_settings()
        spin(lambda: bool(outcomes or errors), timeout=4)
        self.assertEqual(errors, [])
        self.assertEqual(outcomes, [True])

    def test_repeated_ocr_errors_stop_monitor_and_drop_results(self):
        self.monitor.set_enabled(True)
        self.monitor.timer.stop()
        epoch = self.monitor.epoch
        for _ in range(3):
            self.monitor.observed(dict(epoch=epoch, error='OCR failed'))
        self.assertFalse(self.monitor.enabled)
        self.assertIn('OCR 錯誤', self.window.monitor_status.text())
        self.monitor.observed(dict(epoch=epoch, error=None,
            character=config()['characters'][0], text='Late result', seconds=.1))
        self.assertFalse(self.monitor.queue)

    def test_disabled_mode_f11_explains_master_switch_and_does_not_start(self):
        self.monitor.config['enabled'] = False
        self.window.notify_error = Mock()
        with self.assertLogs('euphonia.audiobook', level='INFO') as logs:
            self.window.monitor_hotkey.pressed.emit()
        self.assertFalse(self.monitor.enabled)
        self.assertIsNone(self.monitor.thread)
        self.assertFalse(self.monitor.overlays.boxes)
        self.assertIn('總開關', self.window.notify_error.call_args.args[0])
        self.assertTrue(any('toggle received' in line for line in logs.output))
        self.assertTrue(any('start blocked' in line for line in logs.output))


if __name__ == '__main__':
    unittest.main()

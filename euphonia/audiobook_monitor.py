"""One OCR worker, GUI-thread screenshots, and the existing TTS/player pipeline."""
from collections import deque
import logging
import threading
import time
import numpy as np
from PySide6.QtCore import QObject, QRect, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication
from .audiobook import DialogueState, StreamingDialogueState, REGIONS, match_character
from .audiobook_overlay import AudiobookOverlays

log = logging.getLogger('euphonia.audiobook')


def region_screen(region, screens):
    rect = QRect(region['x'], region['y'], region['width'], region['height'])
    return next((screen for screen in screens if screen.geometry().contains(rect)), None)


def capture_regions(regions, screens, keys=REGIONS):
    """Capture each screen once, then crop independently in physical pixels."""
    snapshots, images = {}, {}
    for key in keys:
        region = regions[key]
        screen = region_screen(region, screens)
        if screen is None:
            raise ValueError('OCR 區域已不在可用螢幕內，請重新框選兩個區域。')
        index = screens.index(screen)
        if index not in snapshots:
            snapshots[index] = screen.grabWindow(0)
        snapshot = snapshots[index]
        if snapshot.isNull():
            raise RuntimeError('背景截圖失敗。')
        ratio = snapshot.devicePixelRatio()
        rect = QRect(region['x'], region['y'], region['width'], region['height'])
        rect.translate(-screen.geometry().topLeft())
        crop = snapshot.copy(QRect(round(rect.x() * ratio), round(rect.y() * ratio),
                                   round(rect.width() * ratio), round(rect.height() * ratio)))
        image = crop.toImage().convertToFormat(QImage.Format_RGB888)
        array = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
        images[key] = array[:, :image.width() * 3].reshape(image.height(), image.width(), 3)[:, :, ::-1].copy()
    return images


class OCRWorker(QObject):
    result = Signal(object)

    def __init__(self, engine, cancelled):
        super().__init__()
        self.engine, self.cancelled = engine, cancelled

    @Slot(object)
    def scan(self, request):
        epoch, config, images, *options = request
        manual = bool(options[0]) if options else False
        manual_character = options[1] if len(options) > 1 else None
        result = dict(epoch=epoch, character=None, text='', error=None, manual=manual,
                      recognized_name='')
        started = time.monotonic()
        try:
            if not self.cancelled.is_set():
                if manual:
                    result['character'] = manual_character
                    result['text'] = self.engine.recognize(images['dialogue_text'])
                else:
                    name = self.engine.recognize(images['character_name'])
                    result['recognized_name'] = (name or '').strip()
                    character = match_character(name, config)
                    if character and not self.cancelled.is_set():
                        result['character'] = character
                        result['text'] = self.engine.recognize(images['dialogue_text'])
        except Exception as exc:
            log.exception('Background OCR failed')
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['seconds'] = time.monotonic() - started
        if not self.cancelled.is_set():
            self.result.emit(result)


class AudiobookMonitor(QObject):
    scan_requested = Signal(object)
    changed = Signal(str)

    def __init__(self, window, config):
        super().__init__(window)
        self.window, self.config = window, config
        self.enabled = False
        self.overlays = AudiobookOverlays(self.read_now)
        self.epoch = 0
        self.thread = self.worker = None
        self.cancelled = None
        self.inflight = False
        self.manual_requested = False
        self.queue = deque()
        self.ready_audio = deque()
        self.playing = False
        self.tts_epoch = None
        self.failures = 0
        self.state = DialogueState(config['stable_duration_ms'])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        window.player.mediaStatusChanged.connect(self.media_status)
        window.player.errorOccurred.connect(self.playback_error)

    def status(self, message=''):
        text = '背景監控 ON · F11 關閉' if self.enabled else '背景監控 OFF · F11 開啟'
        if message:
            text += ' · ' + message
        self.changed.emit(text)

    def toggle(self):
        log.info('Monitoring toggle received; running=%s mode_enabled=%s', self.enabled, self.config['enabled'])
        self.set_enabled(not self.enabled)

    def blocked(self, message):
        self.status(message)
        self.window.notify_error('有聲小說監控未啟動：' + message)
        log.warning('Monitoring start blocked: %s', message)

    def set_enabled(self, enabled):
        if enabled == self.enabled:
            return
        if enabled:
            if QApplication.activeModalWidget() or self.window.capture_pending or self.window.capture:
                self.blocked('請先完成設定或框選')
                return
            if not self.config['enabled']:
                self.blocked('請勾選設定最上方的「啟用有聲小說模式（總開關）」並儲存；角色的啟用勾選是獨立設定。')
                return
            if not all(self.config['ocr_regions'][key] for key in REGIONS):
                self.blocked('請先設定角色名稱與對話區域')
                return
            self.voices = {voice['id']: voice for voice in self.window.store.list()}
            configured = [c for c in self.config['characters'] if c['enabled']]
            if not configured or any(c['voice_id'] not in self.voices for c in configured):
                self.blocked('請設定有效且已啟用的角色 Voice')
                return
            if any(region_screen(r, QApplication.screens()) is None for r in self.config['ocr_regions'].values()):
                self.blocked('螢幕配置已變更，請重新框選區域')
                return
            # Hand over the existing playback pipeline, draining any manual job.
            self.window.stop()
        self.enabled = enabled
        self.epoch += 1
        self.queue.clear()
        self.ready_audio.clear()
        self.manual_requested = False
        self.timer.stop()
        if enabled:
            state_type = StreamingDialogueState if self.window.engine.backend == 'qwen_streaming' else DialogueState
            self.state = state_type(self.config['stable_duration_ms'])
            self.window.engine.streaming_silence_threshold_db = self.config['streaming_silence_threshold_db']
            self.window.engine.streaming_interval_ms = self.config['streaming_interval_ms']
            self.overlays.show(self.config['ocr_regions'])
            self.failures = 0
            # A rapid OFF/ON waits for the old OCR worker to finish, never overlaps.
            if self.thread is None:
                self.start_worker()
        else:
            self.overlays.close()
            if self.cancelled:
                self.cancelled.set()
            if self.thread:
                self.thread.quit()
            if self.tts_epoch is not None and self.window.worker:
                self.window.worker.cancelled.set()
            if self.playing:
                self.window.player.stop()
            self.playing = False
        self.window.busy(self.window.thread is not None)
        self.status()
        log.info('monitor=%s epoch=%d', enabled, self.epoch)

    def start_worker(self):
        if not self.enabled or self.thread is not None:
            return
        self.cancelled = threading.Event()
        self.inflight = False
        self.thread = QThread(self)
        self.worker = OCRWorker(self.window.engine, self.cancelled)
        self.worker.moveToThread(self.thread)
        self.scan_requested.connect(self.worker.scan)
        self.worker.result.connect(self.observed)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.timer.start(self.config['ocr_interval_ms'])
        log.info('OCR worker started epoch=%d', self.epoch)

    def worker_finished(self):
        self.thread = self.worker = None
        self.inflight = False
        if self.enabled:
            self.start_worker()

    def tick(self):
        self.request_scan(False)

    def read_now(self):
        """Immediately OCR and enqueue the current dialogue, bypassing stability/deduping."""
        if not self.enabled:
            return
        if self.inflight:
            self.manual_requested = True
            self.overlays.set_action_busy(True)
            self.status('目前辨識完成後立即 OCR ＋ 播放')
            return
        self.request_scan(True)

    def request_scan(self, manual=False):
        if not self.enabled:
            return
        self.pump()
        if self.inflight or self.window.capture_pending or self.window.capture or QApplication.activeModalWidget():
            return
        try:
            with self.overlays.clean_capture():
                keys = ('dialogue_text',) if manual else REGIONS
                images = capture_regions(self.config['ocr_regions'], QApplication.screens(), keys)
        except Exception as exc:
            log.exception('Background capture failed')
            self.set_enabled(False)
            self.status(str(exc))
            return
        self.inflight = True
        if manual:
            self.overlays.set_action_busy(True)
            self.status('正在立即辨識角色與對話…')
        character = self.manual_character() if manual else None
        self.scan_requested.emit((self.epoch, self.config, images, manual, character))

    def manual_character(self):
        """Choose a Voice for the action button without consulting the name OCR region."""
        selected = self.window.voices.currentData()
        voice_id = selected.get('id') if isinstance(selected, dict) else None
        if voice_id not in self.voices:
            configured = next((row for row in self.config['characters']
                               if row['enabled'] and row['voice_id'] in self.voices), None)
            return configured
        return next((row for row in self.config['characters']
                     if row['enabled'] and row['voice_id'] == voice_id),
                    dict(id='manual-' + voice_id, name=selected['name'], voice_id=voice_id, enabled=True))

    @Slot(object)
    def observed(self, result):
        if not self.enabled or result['epoch'] != self.epoch:
            return
        self.inflight = False
        manual = result.get('manual', False)
        if result['error']:
            self.overlays.set_action_busy(False)
            self.failures += 1
            self.state.observe(None, '', time.monotonic())
            if self.failures >= 3:
                self.set_enabled(False)
            self.status('OCR 錯誤：' + result['error'])
            self.run_pending_manual()
            return
        self.failures = 0
        feedback = None
        try:
            if manual:
                character = result['character']
                text = result['text'].strip()
                utterance = self.state.commit_immediately(character, text, time.monotonic())
                if character is None:
                    feedback = '立即辨識：沒有可用的 Voice'
                elif not text:
                    feedback = '立即辨識：對話區域沒有文字'
                elif len(text) > 5000:
                    feedback = '立即辨識：文字超過 5000 字'
            else:
                observed = self.state.observe(
                    result['character'], result['text'], time.monotonic())
                utterances = observed if isinstance(observed, list) else ([observed] if observed else [])
        except Exception as exc:
            log.exception('Failed to process OCR result')
            self.overlays.set_action_busy(False)
            self.status(f'文字處理錯誤：{type(exc).__name__}: {exc}')
            self.run_pending_manual()
            return
        log.debug('OCR %.3fs state=%s matched=%s', result['seconds'], self.state.state,
                  result['character']['id'] if result['character'] else None)
        if manual:
            utterances = [utterance] if utterance else []
        for utterance in utterances:
            if len(self.queue) >= 8:
                dropped = self.queue.popleft()
                log.warning('Dialogue queue full; dropped oldest character=%s', dropped['character_id'])
            if manual:
                self.queue.appendleft(utterance)
            else:
                self.queue.append(utterance)
            log.info('Dialogue committed character=%s voice=%s chars=%d queued=%d',
                     utterance['character_id'], utterance['voice_id'], len(utterance['text']), len(self.queue))
        if feedback:
            self.status(feedback)
        else:
            phase = 'TTS_PLAYING' if self.playing else ('正在生成語音' if self.tts_epoch is not None else self.state.state)
            self.status(f'{phase} · 等待 {len(self.queue)} 段')
        if manual:
            self.overlays.set_action_busy(False)
        self.pump()
        self.run_pending_manual()

    def run_pending_manual(self):
        if self.manual_requested and self.enabled and not self.inflight:
            self.manual_requested = False
            QTimer.singleShot(0, self.read_now)

    def pump(self):
        w = self.window
        prefetch = w.engine.backend == 'qwen_streaming'
        if (not self.enabled or not self.queue or w.thread is not None or w.capture_pending
                or (self.playing and not prefetch)
                or (w.player.playbackState() != QMediaPlayer.StoppedState and not prefetch)
                or (prefetch and len(self.ready_audio) >= 2)):
            return
        utterance = self.queue.popleft()
        voice = self.voices[utterance['voice_id']]
        epoch = self.epoch
        self.tts_epoch = epoch
        self.overlays.set_action_busy(True, '播放中…' if self.playing else '生成中…')
        language = w.language.currentData()
        w.editor.setPlainText(utterance['text'])
        self.overlays.set_emotion(None)
        self.status('正在生成：' + utterance['character_name'])
        def done(path):
            # BackendProcess already turns a cancelled request into a None
            # result. Do not consult Window.worker here: queued Qt signals may
            # have advanced its lifecycle by the time this callback executes.
            if path and self.enabled and self.epoch == epoch:
                emotion = dict(w.engine.last_emotion or dict(
                    routed='neutral', score=1, sample_label='主要參考音檔'))
                self.ready_audio.append(dict(path=path, utterance=utterance, emotion=emotion))
                log.info('Audio prepared character=%s file=%s ready=%d',
                         utterance['character_name'], path, len(self.ready_audio))
                if not self.playing:
                    self.play_ready()
            else:
                self.overlays.set_action_busy(False)
                log.info('TTS result skipped path=%s enabled=%s request_epoch=%d current_epoch=%d',
                         path, self.enabled, epoch, self.epoch)
        def error(message):
            log.error('Audiobook TTS failed: %s', message)
            if self.epoch == epoch:
                self.set_enabled(False)
                self.status('TTS 錯誤：' + message)
        def model_progress(message):
            if self.enabled and self.epoch == epoch:
                self.status(message)
        w.run_job(lambda progress, cancelled: w.engine.synthesize(
            voice, utterance['text'], language, progress, cancelled), done,
            error_callback=error, progress_callback=model_progress)

    def play_ready(self):
        if not self.enabled or self.playing or not self.ready_audio:
            return False
        item = self.ready_audio.popleft()
        utterance, path = item['utterance'], item['path']
        self.overlays.set_emotion(item['emotion'])
        self.playing = True
        self.overlays.set_action_busy(True, '播放中…')
        self.window.output = path
        self.window.export_button.setEnabled(True)
        self.window.replay_button.setEnabled(False)
        self.status('TTS_PLAYING · ' + utterance['character_name'])
        log.info('Starting prepared playback character=%s file=%s ready=%d',
                 utterance['character_name'], path, len(self.ready_audio))
        self.window.play(path)
        return True

    def job_finished(self):
        self.tts_epoch = None
        if not self.playing and not self.ready_audio:
            self.overlays.set_action_busy(False)
        self.pump()

    def media_status(self, status):
        if self.playing:
            log.info('Playback media status=%s state=%s error=%s', status,
                     self.window.player.playbackState(), self.window.player.errorString())
        if status == QMediaPlayer.EndOfMedia:
            self.playing = False
            if self.enabled:
                if not self.play_ready():
                    # The configured interval is already present at the end of
                    # the WAV. If it elapsed while OCR kept growing, submit the
                    # currently safe tail instead of waiting for full stability.
                    if (isinstance(self.state, StreamingDialogueState)
                            and self.tts_epoch is None and not self.queue):
                        pending = self.state.flush_pending_for_interval()
                        if pending:
                            self.queue.appendleft(pending)
                            log.info('Streaming interval committed chars=%d',
                                     len(pending['text']))
                    self.overlays.set_action_busy(self.tts_epoch is not None,
                                                  '生成中…' if self.tts_epoch is not None else None)
                    self.status('正在生成下一段' if self.tts_epoch is not None
                                else ('INTERVAL_COMMITTED' if self.queue else 'WAITING_FOR_NEXT_TEXT'))
                QTimer.singleShot(0, self.pump)

    def playback_error(self, *args):
        if self.playing:
            message = self.window.player.errorString()
            self.set_enabled(False)
            self.status('播放失敗：' + message)
            log.error('Audiobook playback failed: %s', message)

    def shutdown(self):
        self.set_enabled(False)
        self.overlays.close()
        if self.thread is not None:
            self.cancelled.set()
            self.thread.quit()
            self.thread.wait()  # OCR owns no GUI objects or synchronous GUI calls.

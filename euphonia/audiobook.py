"""Audiobook settings, conservative name matching and time-based dialogue state.

No Qt, OCR or TTS services live here; the desktop controller supplies observations.
"""
from collections import deque
from difflib import SequenceMatcher
import json
import logging
import re
import unicodedata
import uuid

log = logging.getLogger('euphonia.audiobook')
REGIONS = ('character_name', 'dialogue_text')


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text or '').casefold() if c.isalnum())


def streaming_signature(text):
    """Compare OCR growth without discarding punctuation or quotation marks."""
    return ' '.join(unicodedata.normalize('NFKC', text or '').casefold().split())


def speech_text(text):
    """Keep OCR/display text intact while removing quotes from spoken content."""
    return re.sub(r'["\'＂＇“”‘’「」『』«»‹›]', '', text or '').strip()


def defaults():
    return dict(enabled=False, characters=[], ocr_regions={key: None for key in REGIONS},
                ocr_interval_ms=300, stable_duration_ms=700, match_threshold=0.88,
                fuzzy_match=True, streaming_silence_threshold_db=-38.0,
                streaming_interval_ms=75)


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError('有聲小說設定格式無效。')
    result = defaults()
    result.update({k: value[k] for k in result if k in value})
    if type(result['enabled']) is not bool or type(result['fuzzy_match']) is not bool:
        raise ValueError('啟用設定必須為布林值。')
    for key, low, high in [('ocr_interval_ms', 100, 5000), ('stable_duration_ms', 300, 5000)]:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'{key} 必須介於 {low}–{high} ms。')
    if (not isinstance(result['streaming_silence_threshold_db'], (int, float))
            or isinstance(result['streaming_silence_threshold_db'], bool)
            or not -80 <= result['streaming_silence_threshold_db'] <= -10):
        raise ValueError('streaming_silence_threshold_db 必須介於 -80～-10 dB。')
    result['streaming_silence_threshold_db'] = float(result['streaming_silence_threshold_db'])
    if (type(result['streaming_interval_ms']) is not int
            or not 0 <= result['streaming_interval_ms'] <= 1000):
        raise ValueError('streaming_interval_ms 必須介於 0～1000 ms。')
    if not isinstance(result['match_threshold'], (int, float)) or not 0.75 <= result['match_threshold'] <= 1:
        raise ValueError('角色相似度必須介於 0.75–1.00。')
    regions = result['ocr_regions']
    if not isinstance(regions, dict):
        raise ValueError('OCR 區域格式無效。')
    result['ocr_regions'] = {}
    for key in REGIONS:
        region = regions.get(key)
        if region is not None:
            if not isinstance(region, dict) or any(type(region.get(k)) is not int for k in ('x', 'y', 'width', 'height')):
                raise ValueError('OCR 區域座標無效。')
            if not 8 <= region['width'] <= 32768 or not 8 <= region['height'] <= 32768:
                raise ValueError('OCR 區域至少為 8 × 8。')
            region = {k: region[k] for k in ('x', 'y', 'width', 'height')}
        result['ocr_regions'][key] = region
    if not isinstance(result['characters'], list) or len(result['characters']) > 200:
        raise ValueError('最多可設定 200 個角色。')
    names, ids, characters = set(), set(), []
    for row in result['characters']:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str):
            raise ValueError('角色格式無效。')
        name = row['name'].strip()
        key = normalize(name)
        if not key or len(name) > 100 or key in names:
            raise ValueError('角色名稱不可空白、超過 100 字，或與其他角色正規化後相同。')
        identity = row.get('id') or uuid.uuid4().hex
        if not isinstance(identity, str) or identity in ids:
            raise ValueError('角色識別碼重複或無效。')
        if not isinstance(row.get('voice_id'), str) or not row['voice_id']:
            raise ValueError(f'{name} 尚未選擇 Voice。')
        if type(row.get('enabled', True)) is not bool:
            raise ValueError('角色啟用設定無效。')
        names.add(key)
        ids.add(identity)
        characters.append(dict(id=identity, name=name, voice_id=row['voice_id'], enabled=row.get('enabled', True)))
    result['characters'] = characters
    return result


def load_config(settings):
    raw = settings.value('audiobook_mode', '')
    if not raw:
        return defaults()
    try:
        return validate_config(json.loads(raw))
    except (ValueError, TypeError, KeyError):
        log.exception('Invalid audiobook settings; monitoring remains off; original value retained')
        return defaults()


def save_config(settings, config):
    config = validate_config(config)
    settings.setValue('audiobook_mode', json.dumps(config, ensure_ascii=False))
    settings.sync()
    if int(settings.status().value) != 0:
        raise OSError('無法儲存有聲小說設定。')
    return config


def match_character(text, config):
    key = normalize(text)
    if not key:
        return None
    # Disabled exact matches must not fall through to another enabled character.
    for row in config['characters']:
        if normalize(row['name']) == key:
            return row if row['enabled'] else None
    if not config['fuzzy_match'] or len(key) < 4:
        return None
    ranked = sorted(((SequenceMatcher(None, key, normalize(row['name']), autojunk=False).ratio(), row)
                     for row in config['characters'] if len(normalize(row['name'])) >= 4),
                    key=lambda pair: pair[0], reverse=True)
    if not ranked or ranked[0][0] < config['match_threshold']:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
        return None  # Ambiguous names are safer left silent.
    return ranked[0][1] if ranked[0][1]['enabled'] else None


class DialogueState:
    """Stable >= configured duration AND >=2 observations; bounded recent history.

    Current committed text stays suppressed even after TTL. Small OCR edits and
    trailing typewriter growth cannot replay it. A changed scene can reuse text
    after 30 seconds or eight intervening committed lines.
    """
    def __init__(self, stable_ms=700, recent_seconds=30, recent_count=8):
        self.stable_seconds = stable_ms / 1000
        self.recent_seconds = recent_seconds
        self.recent = deque(maxlen=recent_count)
        self.current = None
        self.candidate = None
        self.since = 0
        self.samples = 0
        self.absent_since = None
        self.state = 'IDLE'

    @staticmethod
    def same_line(a, b):
        if a[0] != b[0]:
            return False
        x, y = a[1], b[1]
        if x == y:
            return True
        # A late typewriter suffix is part of the committed dialogue, not a new line.
        if len(x) >= 4 and y.startswith(x):
            return True
        return (min(len(x), len(y)) >= 8 and min(len(x), len(y)) / max(len(x), len(y)) >= .85
                and SequenceMatcher(None, x, y, autojunk=False).ratio() >= .92)

    def observe(self, character, text, now):
        key = normalize(text)
        if character is None or not key or len(text) > 5000:
            self.candidate = None
            self.samples = 0
            self.state = 'IDLE'
            if self.absent_since is None:
                self.absent_since = now
            elif now - self.absent_since >= self.stable_seconds:
                self.current = None
            return None
        self.absent_since = None
        identity = (character['id'], key)
        if self.current and self.same_line(self.current, identity):
            self.candidate = None
            self.state = 'WAITING_FOR_NEXT_TEXT'
            return None
        if identity != self.candidate:
            self.candidate, self.since, self.samples = identity, now, 1
            self.state = 'TEXT_GROWING'
            return None
        self.samples += 1
        if self.samples < 2 or now - self.since < self.stable_seconds:
            return None
        self.state = 'TEXT_STABLE'
        while self.recent and now - self.recent[0][1] >= self.recent_seconds:
            self.recent.popleft()
        self.current = identity
        if any(self.same_line(old, identity) for old, _ in self.recent):
            self.state = 'WAITING_FOR_NEXT_TEXT'
            return None
        self.recent.append((identity, now))
        self.state = 'TEXT_COMMITTED'
        return dict(character_id=character['id'], character_name=character['name'],
                    voice_id=character['voice_id'], text=speech_text(text))

    def commit_immediately(self, character, text, now):
        """Commit a user-requested OCR result now and suppress its automatic replay."""
        text = (text or '').strip()
        key = normalize(text)
        if character is None or not key or len(text) > 5000:
            return None
        identity = (character['id'], key)
        self.current = identity
        self.candidate = None
        self.samples = 0
        self.absent_since = None
        self.state = 'TEXT_COMMITTED'
        self.recent.append((identity, now))
        return dict(character_id=character['id'], character_name=character['name'],
                    voice_id=character['voice_id'], text=speech_text(text))


class StreamingDialogueState:
    """Commit completed punctuation-delimited text while OCR text is growing.

    Text after the last punctuation is held until the OCR result stops changing
    for ``stable_ms``.  Already committed prefixes are never emitted twice.
    """
    # Streaming uses an explicit allowlist of half/full-width sentence marks.
    # Quotes and newlines do not split. A period between digits is a decimal dot.
    _BOUNDARY = re.compile(r'[,，。．!！?？;；]+|(?<!\d)\.|\.(?!\d)')
    _CLOSERS = '”’"\'」』】）》〕〉'
    _ABBREVIATIONS = {
        # Personal and professional titles.
        'mr', 'mrs', 'ms', 'mx', 'dr', 'prof', 'rev', 'hon', 'pres', 'gov',
        'capt', 'cpt', 'cmdr', 'lt', 'col', 'gen', 'sgt', 'adm', 'sr', 'jr',
        # Common prose abbreviations that can appear before more OCR text.
        'eg', 'ie', 'etc', 'vs', 'viz', 'approx', 'dept', 'est', 'fig', 'no',
        'inc', 'ltd', 'co', 'corp', 'st', 'mt', 'ft', 'a.m', 'p.m', 'u.s',
        'u.k', 'ph.d', 'm.d', 'b.c', 'a.d',
        # Month names commonly written with a trailing period.
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct',
        'nov', 'dec',
    }

    def __init__(self, stable_ms=700):
        self.stable_seconds = stable_ms / 1000
        self.character_id = None
        self.character = None
        self.latest_text = ''
        self.candidate_key = None
        self.since = 0
        self.samples = 0
        self.emitted_end = 0
        self.finished = None
        self.absent_since = None
        self.state = 'IDLE'

    def _reset(self, character_id=None):
        self.character_id = character_id
        self.character = None
        self.latest_text = ''
        self.candidate_key = None
        self.since = 0
        self.samples = 0
        self.emitted_end = 0
        self.finished = None
        self.state = 'IDLE'

    @staticmethod
    def _utterance(character, text):
        return dict(character_id=character['id'], character_name=character['name'],
                    voice_id=character['voice_id'], text=speech_text(text))

    @classmethod
    def _abbreviation_period(cls, text, match):
        punctuation = text[match.start():match.end()]
        if punctuation not in ('.', '．'):
            return False
        prefix = text[:match.start()]
        token_match = re.search(r'([A-Za-z]+(?:\.[A-Za-z]+)*)$', prefix)
        if not token_match:
            return False
        token = token_match.group(1).casefold()
        if token in cls._ABBREVIATIONS:
            return True
        # Initials and dotted initialisms: A., J.R., U.S.
        pieces = token.split('.')
        return all(len(piece) == 1 for piece in pieces)

    def _punctuated(self, character, text):
        items = []
        cursor = self.emitted_end
        for match in self._BOUNDARY.finditer(text, cursor):
            if self._abbreviation_period(text, match):
                continue
            end = match.end()
            segment = text[cursor:end].strip()
            if normalize(segment):
                items.append(self._utterance(character, segment))
            cursor = end
        self.emitted_end = cursor
        return items

    def observe(self, character, text, now):
        text = (text or '').strip()
        content_key = normalize(text)
        key = streaming_signature(text)
        if character is None or not content_key or len(text) > 5000:
            self.candidate_key = None
            self.samples = 0
            self.state = 'IDLE'
            if self.absent_since is None:
                self.absent_since = now
            elif now - self.absent_since >= self.stable_seconds:
                self._reset()
            return []
        self.absent_since = None

        if character['id'] != self.character_id:
            self._reset(character['id'])
        self.character = character
        if self.finished == (character['id'], key):
            self.state = 'WAITING_FOR_NEXT_TEXT'
            return []
        if self.finished is not None:
            # A typewriter can pause longer than the stability threshold and
            # then append more text. Continue from the spoken prefix in that
            # case; only unrelated text starts a new dialogue session.
            if text.startswith(self.latest_text) and len(text) > len(self.latest_text):
                self.finished = None
            else:
                self._reset(character['id'])
        self.character = character

        # If OCR revises an already spoken prefix, it cannot be taken back. Keep
        # the spoken offset and wait for stability instead of replaying the line.
        prefix = self.latest_text[:self.emitted_end]
        prefix_intact = not prefix or text.startswith(prefix)
        if not prefix_intact:
            self.state = 'OCR_REVISION'

        if key != self.candidate_key:
            self.candidate_key, self.since, self.samples = key, now, 1
            self.state = 'TEXT_GROWING'
        else:
            self.samples += 1
        self.latest_text = text

        items = self._punctuated(character, text) if prefix_intact else []
        if items:
            self.state = 'PUNCTUATION_COMMITTED'

        stable = self.samples >= 2 and now - self.since >= self.stable_seconds
        if stable:
            tail = text[self.emitted_end:].strip() if prefix_intact else ''
            if normalize(tail):
                items.append(self._utterance(character, tail))
            self.emitted_end = len(text)
            self.finished = (character['id'], key)
            self.state = 'TEXT_COMMITTED'
        return items

    def flush_pending_for_interval(self):
        """Commit safe pending text after the previous audio's gap has elapsed."""
        if self.character is None or self.emitted_end >= len(self.latest_text):
            return None
        pending = self.latest_text[self.emitted_end:]
        cut = len(pending)
        # For whitespace-delimited languages, retain the currently growing word.
        # CJK text has no reliable word boundary, so all visible characters are safe.
        if not any('\u3400' <= char <= '\u9fff' for char in pending):
            trailing_word = re.search(r'\S+$', pending)
            if trailing_word and trailing_word.start() > 0:
                cut = trailing_word.start()
            elif trailing_word:
                return None
        segment = pending[:cut]
        if not normalize(segment):
            return None
        self.emitted_end += cut
        self.state = 'INTERVAL_COMMITTED'
        return self._utterance(self.character, segment)

    def commit_immediately(self, character, text, now):
        """Commit the overlay button's OCR result and suppress auto replay."""
        text = (text or '').strip()
        content_key = normalize(text)
        key = streaming_signature(text)
        if character is None or not content_key or len(text) > 5000:
            return None
        self._reset(character['id'])
        self.character = character
        self.latest_text = text
        self.candidate_key = key
        self.since = now
        self.samples = 1
        self.emitted_end = len(text)
        self.finished = (character['id'], key)
        self.absent_since = None
        self.state = 'TEXT_COMMITTED'
        return self._utterance(character, text)

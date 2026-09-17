import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

diagnostics = Path(__file__).resolve().parent / 'data/logs/hotkey.log'
diagnostics.parent.mkdir(parents=True, exist_ok=True)
handler = RotatingFileHandler(diagnostics, maxBytes=1_000_000, backupCount=2, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
logger = logging.getLogger('euphonia')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
audiobook_handler = RotatingFileHandler(diagnostics.parent / 'audiobook.log',
    maxBytes=1_000_000, backupCount=2, encoding='utf-8')
audiobook_handler.setFormatter(handler.formatter)
audiobook_logger = logging.getLogger('euphonia.audiobook')
audiobook_logger.addHandler(audiobook_handler)
audiobook_logger.propagate = False

# Keep startup errors visible on disk when launched without a console.
if sys.stdout is None or sys.stderr is None:
    log_path = Path(__file__).resolve().parent / 'data/logs/app.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open('a', encoding='utf-8', buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log

from euphonia.app import main

if __name__ == '__main__':
    main()

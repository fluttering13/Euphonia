"""Compatibility wrapper for the retained F5-TTS and ZipVoice downloads."""
import sys

from download_models import download_selected


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'f5', 'zipvoice'}:
        raise SystemExit('Usage: prepare_extra_models.py {f5|zipvoice}')
    download_selected(sys.argv[1])

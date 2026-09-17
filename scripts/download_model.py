"""Download public model weights with resumable HTTP ranges and SHA-256 checks."""
import concurrent.futures
import hashlib
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from euphonia.core import DATA, MODEL

CHUNK = 16 * 1024 * 1024


def download(model=MODEL, root=None, filenames=None, token=False, revision=None):
    info = HfApi(token=token).model_info(model, revision=revision, files_metadata=True)
    root = Path(root) if root is not None else DATA / 'model'
    root.mkdir(parents=True, exist_ok=True)
    for item in info.siblings:
        if filenames is not None and item.rfilename not in filenames:
            continue
        if item.rfilename.endswith('.md') or item.rfilename.startswith('.'):
            continue
        target = root / item.rfilename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not item.lfs:
            source = hf_hub_download(model, item.rfilename, revision=info.sha, token=token)
            target.write_bytes(Path(source).read_bytes())
            continue
        expected = item.lfs.sha256
        if target.exists():
            with target.open('rb') as stream:
                valid = hashlib.file_digest(stream, 'sha256').hexdigest() == expected
            if valid:
                print('Already downloaded:', item.rfilename, flush=True)
                continue
        chunks = target.parent / (target.name + '.parts')
        chunks.mkdir(exist_ok=True)
        count = (item.size + CHUNK - 1) // CHUNK

        def fetch(index):
            start = index * CHUNK
            end = min(item.size, start + CHUNK) - 1
            part = chunks / str(index)
            if part.exists() and part.stat().st_size == end - start + 1:
                return
            url = f'https://huggingface.co/{model}/resolve/{info.sha}/{item.rfilename}?part={index}'
            for attempt in range(3):
                try:
                    headers = {'Range': f'bytes={start}-{end}'}
                    if token:
                        headers['Authorization'] = 'Bearer ' + token
                    with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{item.size}':
                            raise RuntimeError('Server did not honor byte range')
                        temp = part.with_suffix('.tmp')
                        with temp.open('wb') as stream:
                            for data in response.iter_content(1024 * 1024):
                                stream.write(data)
                    if temp.stat().st_size != end - start + 1:
                        raise RuntimeError('Incomplete download')
                    temp.replace(part)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)

        workers = max(1, min(32, int(os.environ.get('EUPHONIA_DOWNLOAD_WORKERS', '8'))))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for completed, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
                future.result()
                print(f'{item.rfilename}: {completed}/{count} parts', flush=True)
        temp = target.with_suffix('.download')
        with temp.open('wb') as stream:
            for i in range(count):
                with (chunks / str(i)).open('rb') as part:
                    while data := part.read(1024 * 1024):
                        stream.write(data)
        with temp.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise RuntimeError('Model checksum mismatch')
        temp.replace(target)
        for i in range(count):
            (chunks / str(i)).unlink()
        for leftover in chunks.glob('*.tmp'):
            leftover.unlink()
        chunks.rmdir()
        print('Verified:', item.rfilename, flush=True)
    (root / 'ready').write_text(info.sha, encoding='utf-8')
    print('Model ready:', root, flush=True)


if __name__ == '__main__':
    download()

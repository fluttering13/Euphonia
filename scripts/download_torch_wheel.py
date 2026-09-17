"""Resumable parallel download of the official Windows PyTorch wheel."""
import concurrent.futures
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import time
from urllib.parse import unquote, urljoin, urlsplit
import requests


def main():
    class Links(HTMLParser):
        links = []
        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.links.extend(v for k, v in attrs if k == 'href')
    index = 'https://download.pytorch.org/whl/cu128/torch/'
    parser = Links()
    response = requests.get(index, timeout=60)
    response.raise_for_status()
    parser.feed(response.text)
    link = next(urljoin(index, x) for x in parser.links
                if 'torch-2.7.1+cu128-cp312-cp312-win_amd64.whl' in unquote(x))
    url, expected = link.split('#sha256=')
    root = Path(__file__).resolve().parents[1] / 'data/wheels'
    root.mkdir(parents=True, exist_ok=True)
    target = root / unquote(Path(urlsplit(url).path).name)
    if target.is_file():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == expected:
                print('Already verified:', target, flush=True)
                return
    response = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=60)
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError('Official server does not support byte ranges.')
    size = int(response.headers['Content-Range'].split('/')[-1])
    block = 16 * 1024**2
    count = (size + block - 1) // block
    parts = root / 'torch-cu128.parts'
    parts.mkdir(exist_ok=True)
    def fetch(index):
        start, end = index * block, min(size, (index + 1) * block) - 1
        part = parts / str(index)
        if part.is_file() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(30, 90)) as r:
                    r.raise_for_status()
                    if r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Unexpected byte range')
                    temp = part.with_suffix('.tmp')
                    with temp.open('wb') as f:
                        for chunk in r.iter_content(1024**2):
                            f.write(chunk)
                    if temp.stat().st_size != end - start + 1:
                        raise RuntimeError('Incomplete part')
                    temp.replace(part)
                    return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    print(f'Downloading {target.name}: {size / 1024**3:.2f} GiB', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, n) for n in range(count)]), 1):
            future.result()
            print(f'PyTorch: {i}/{count}', flush=True)
    temporary = target.with_suffix('.download')
    with temporary.open('wb') as f:
        for i in range(count):
            with (parts / str(i)).open('rb') as p:
                while chunk := p.read(1024**2):
                    f.write(chunk)
    with temporary.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
            raise RuntimeError('PyTorch checksum mismatch')
    temporary.replace(target)
    for i in range(count):
        (parts / str(i)).unlink()
    parts.rmdir()
    print('Verified:', target, flush=True)


if __name__ == '__main__':
    main()

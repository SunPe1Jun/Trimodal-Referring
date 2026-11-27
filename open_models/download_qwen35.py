"""Download official ModelScope files and verify against a pinned HF file tree."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

BASE = Path('/workspace/usr3/models/baseline_expansion_20261116')
OUT = BASE / 'Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(record):
    name = record['path']
    target = OUT / name
    url = 'https://www.modelscope.cn/models/Qwen/Qwen3.5-9B/resolve/master/' + name
    lfs = record.get('lfs')
    def valid(path):
        if not path.exists() or path.stat().st_size != record['size']:
            return False
        if lfs:
            return checksum(path) == lfs['oid']
        data = path.read_bytes()
        return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == record['oid']
    if valid(target):
        return name
    temp = target.with_suffix(target.suffix + '.part')
    if not lfs:
        data = urllib.request.urlopen(url, timeout=25).read()
        temp.write_bytes(data)
        if not valid(temp):
            # The official mirrors can have different README/license revisions.
            # Use the pinned HF copy when a small-file Git blob differs.
            fallback = 'https://hf-mirror.com/Qwen/Qwen3.5-9B/resolve/' + REVISION + '/' + name
            temp.write_bytes(urllib.request.urlopen(fallback, timeout=25).read())
    else:
        fd = os.open(temp, os.O_CREAT | os.O_RDWR)
        os.ftruncate(fd, record['size'])
        size = 32 * 2**20
        def chunk(start):
            end = min(record['size'], start + size) - 1
            for attempt in range(5):
                try:
                    response = urllib.request.urlopen(urllib.request.Request(url,
                        headers={'Range': f'bytes={start}-{end}'}), timeout=40)
                    assert response.status == 206
                    assert response.headers['Content-Range'].startswith(f'bytes {start}-')
                    pos = start
                    while True:
                        data = response.read(2**20)
                        if not data:
                            break
                        os.pwrite(fd, data, pos)
                        pos += len(data)
                    assert pos == end + 1
                    return
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(3)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(chunk, range(0, record['size'], size)))
        finally:
            os.close(fd)
    if not valid(temp):
        raise RuntimeError('Pinned checkpoint checksum mismatch: ' + name)
    temp.replace(target)
    print('VERIFIED', name, record['size'], flush=True)
    return name


def main():
    OUT.mkdir(exist_ok=True)
    records = json.loads((BASE / 'Qwen3.5-9B.tree.json').read_text())
    records = [r for r in records if r['path'] not in ('.gitattributes', 'LICENSE', 'README.md')]
    # Complete processor files first so CPU validation can run during weight downloads.
    small = [r for r in records if not r.get('lfs')]
    weights = [r for r in records if r.get('lfs')]
    for record in small:
        download(record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, weights))
    (OUT / 'download_verification.json').write_text(json.dumps({
        'source': 'Official Qwen ModelScope repository', 'hf_revision': REVISION,
        'verification': 'Git blob SHA-1 for small files; LFS SHA-256 for all weights',
        'files': records}, indent=2) + '\n')
    (OUT / 'DOWNLOAD_COMPLETE.json').write_text(json.dumps({
        'repo': 'Qwen/Qwen3.5-9B', 'revision': REVISION}) + '\n')
    print('COMPLETE Qwen/Qwen3.5-9B', flush=True)


if __name__ == '__main__':
    main()

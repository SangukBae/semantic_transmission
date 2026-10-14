"""Pinned, isolated TimeSformer setup for the LGVSC Kinetics evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

from .artifacts import sha256, write_json
from .etri_60s_check import lock

REPO = Path(__file__).resolve().parents[2]
COMMIT = 'a5ef29a7b7264baff199a30b3306ac27de901133'
AUTHOR_COMMIT = '50c9ff98fb1aaa8136aef2e41e02bbb0d1a2d1c3'
MANIFEST_SHA = '8c10e3bc1ff80a083472d4b5ea9dbeaecd7a52a973f0b111a47fce0cf83e5f0e'
SOURCE_URL = f'https://codeload.github.com/facebookresearch/TimeSformer/tar.gz/{COMMIT}'
CHECKPOINT_URL = 'https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=1'
CHECKPOINT_SHA = '45c4eeccc4956447a4fca765b665b443c701d8d2e8ae600510565a649651d973'
SOURCE_SHA = 'a4043a7eefdc9321592cc0f0d4a1ed435c95c289fa58bb68b245feb2d9e36497'
ROOT = REPO/'.local/bsu_kinetics_runtime'
PACKAGES = ['fvcore==0.1.5.post20221221','iopath==0.1.10','yacs==0.1.8',
            'simplejson==3.19.3','timm==0.4.12']


def event(**values):
    print('COMPARE_PROGRESS '+json.dumps(values,ensure_ascii=False),flush=True)


def fetch(url, path, expected=None):
    path = Path(path)
    if path.exists():
        if expected and sha256(path) != expected:
            raise ValueError(f'cached asset changed: {path}')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.part')
    if temporary.exists() and expected and sha256(temporary)==expected:
        temporary.replace(path)
        return path
    # Range resumes only when the server confirms the exact byte offset.
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request,timeout=60) as response:
        resumed = offset and response.status == 206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
        if not resumed: offset = 0
        length = response.headers.get('Content-Length')
        total = offset+int(length) if length else None
        done = offset; last = 0
        with temporary.open('ab' if resumed else 'wb') as stream:
            while True:
                chunk = response.read(1024*1024)
                if not chunk: break
                stream.write(chunk); done += len(chunk)
                now = time.monotonic()
                if now-last >= 1 or done == total:
                    label = f'{done}/{total} ({100*done/total:.1f}%)' if total else f'{done} bytes (전체 크기 미확정)'
                    event(stage='다운로드 '+path.name+' '+label)
                    last = now
        if total is not None and done != total:
            raise ValueError('incomplete download: '+str(path))
    if expected and sha256(temporary) != expected:
        raise ValueError('download SHA256 mismatch: '+str(path))
    temporary.replace(path)
    return path


def source_tree(root):
    archive = fetch(SOURCE_URL,root/'TimeSformer.tar.gz',SOURCE_SHA)
    target = root/'TimeSformer'
    receipt = root/'source.json'
    if receipt.exists():
        data = json.loads(receipt.read_text())
        if data['commit'] != COMMIT or data['patch_sha256'] != sha256(REPO/'07_downstream/timesformer/patches/timesformer_modifications.patch'):
            raise ValueError('TimeSformer source policy changed')
        for name,digest in data['files'].items():
            if sha256(target/name) != digest: raise ValueError('TimeSformer source changed: '+name)
        return target
    if target.exists():
        raise ValueError('unverified TimeSformer directory; preserve it and choose a new --runtime')
    with tempfile.TemporaryDirectory(prefix='timesformer-',dir=root) as temp:
        tree = Path(temp)/'tree'; tree.mkdir()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts: continue
                if '..' in parts or member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise ValueError('unsafe source archive member')
                dest = tree.joinpath(*parts)
                if member.isdir(): dest.mkdir(parents=True,exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(bundle.extractfile(member).read())
        patch = REPO/'07_downstream/timesformer/patches/timesformer_modifications.patch'
        # git apply filters paths when invoked inside our parent checkout.
        # POSIX patch explicitly applies relative to this isolated source tree.
        subprocess.run(['patch','-p1','--batch','--forward','--dry-run','-i',str(patch)],cwd=tree,check=True)
        subprocess.run(['patch','-p1','--batch','--forward','-i',str(patch)],cwd=tree,check=True)
        tree.replace(target)
    write_json(receipt,dict(commit=COMMIT,archive_sha256=sha256(archive),patch_sha256=sha256(patch),
        files={str(p.relative_to(target)):sha256(p) for p in target.rglob('*') if p.is_file()}))
    return target


def setup(root=ROOT):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with lock(root/'setup.lock'):
        return _setup(root)


def _setup(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    python=root/'env/bin/python';stamp=root/'environment.json'
    spec=dict(base_python=sys.executable,packages=PACKAGES)
    if stamp.exists() and json.loads(stamp.read_text()) != spec:
        raise ValueError('TimeSformer environment differs; choose a new runtime directory')
    if not stamp.exists():
        event(stage='TimeSformer 독립 환경 준비 (진행률 미확정)')
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(root/'env')],check=True)
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',*PACKAGES],check=True)
        write_json(stamp,spec)
    source=source_tree(root)
    checkpoint=fetch(CHECKPOINT_URL,root/'TimeSformer_divST_8x32_224_K400.pyth',CHECKPOINT_SHA)
    return dict(python=str(python),source=str(source),source_receipt=str(root/'source.json'),
        source_receipt_sha256=sha256(root/'source.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        upstream_commit=COMMIT,checkpoint_url=CHECKPOINT_URL)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();print(json.dumps(setup(args.root),indent=2))

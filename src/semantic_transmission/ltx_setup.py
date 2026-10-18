"""Download only the registered official LTX weights; retain SHA256 evidence."""
import argparse
from pathlib import Path

from .artifacts import sha256, write_json
from .ltx_receiver import profile


def main():
    from huggingface_hub import HfApi, get_token, hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--check-access', action='store_true')
    args = p.parse_args(); cfg = profile(); repo = 'Lightricks/LTX-2.5'
    api = HfApi(token=get_token())
    # A public model-info response does not prove gated weight access.
    api.auth_check(repo_id=repo, repo_type='model')
    print('OFFICIAL_LTX_WEIGHT_ACCESS_AUTHORIZED', flush=True)
    if args.check_access:
        return
    info = api.model_info(repo, revision=cfg['model_revision'], files_metadata=True)
    expected = {f.rfilename: f for f in info.siblings}
    files = {}
    for i, name in enumerate(cfg['weights'].values(), 1):
        print(f'DOWNLOAD/VERIFY {i}/{len(cfg["weights"])} {name}', flush=True)
        path = Path(hf_hub_download(repo, name, revision=cfg['model_revision'],
                                   local_dir=args.model_dir, token=get_token()))
        digest = sha256(path); lfs = expected[name].lfs
        if lfs is None or digest != lfs.sha256 or path.stat().st_size != expected[name].size:
            raise ValueError(f'official LFS hash/size mismatch: {name}')
        stat = path.stat()
        files[name] = dict(sha256=digest, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_json(args.model_dir/'manifest.json', dict(repo_id=repo,
        revision=cfg['model_revision'], files=files, full_weight_hashes_verified=True))
    print('LTX_WEIGHTS_DOWNLOADED_AND_HASH_VERIFIED', flush=True)


if __name__ == '__main__':
    main()

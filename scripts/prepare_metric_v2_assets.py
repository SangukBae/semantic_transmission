#!/usr/bin/env python3
"""Download the pinned public corpus and metric weights; no datasets deleted."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import requests

ASSETS = {
    "DAVIS-2017-trainval-480p.zip": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "dinov2_vits14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
}
COMMITS = {"sam2": "2b90b9f5ceec907a1c18123530e92e794ad901a4", "dinov2": "7764ea0f912e53c92e82eb78a2a1631e92725fc8"}
EXPECTED_SHA256 = {
    "DAVIS-2017-trainval-480p.zip": "e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492",
    "sam2.1_hiera_tiny.pt": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("data/metric_v2_public"));p.add_argument("--models",type=Path,default=Path(".local/metric_v2_models"))
    args=p.parse_args();records=[]
    for name,url in ASSETS.items():
        directory=args.data if name.endswith(".zip") else args.models
        directory.mkdir(parents=True,exist_ok=True);dest=directory/name
        if not dest.exists():
            part=dest.with_suffix(dest.suffix+".part")
            with requests.get(url,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                with part.open("wb") as out:
                    for block in response.iter_content(1024*1024):out.write(block)
            part.replace(dest)
        digest=sha(dest)
        if digest!=EXPECTED_SHA256[name]:raise ValueError("asset hash differs from frozen pilot: "+str(dest))
        records.append({"file":str(dest.resolve()),"url":url,"sha256":digest,"bytes":dest.stat().st_size})
        if name.endswith(".zip") and not (args.data/"DAVIS/ImageSets/2017/val.txt").exists():
            with zipfile.ZipFile(dest) as archive:
                for member in archive.namelist():
                    if not (args.data/member).resolve().is_relative_to(args.data.resolve()):raise ValueError("unsafe archive path")
                archive.extractall(args.data)
    for name,commit in COMMITS.items():
        dest=args.models/name
        if not dest.exists():
            subprocess.run(["git","clone","https://github.com/facebookresearch/"+name+".git",str(dest)],check=True)
            subprocess.run(["git","checkout",commit],cwd=dest,check=True)
        actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=dest,text=True).strip()
        if actual!=commit:raise ValueError("existing model checkout differs; preserve and inspect: "+str(dest))
    (args.data/"metric_v2_assets.json").write_text(json.dumps({"assets":records,"commits":COMMITS},indent=2)+"\n")
    print("assets verified; install SAM2 separately in the isolated evaluation environment")


if __name__=="__main__":main()

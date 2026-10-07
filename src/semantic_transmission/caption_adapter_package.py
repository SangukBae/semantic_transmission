"""Export/install the fixed inference adapter without training or model loading."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from . import qwen35_caption_adapter as adapter
from .artifacts import sha256


def verify_directory(directory):
    directory = Path(directory)
    for name, expected in adapter.FILES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"adapter hash mismatch: {name}")
    return directory


def export_package(output, source=None):
    source = verify_directory(source or adapter.ADAPTER)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, kind="qwen35_caption_repair_inference_adapter",
        files=dict(adapter.FILES), rank=32, alpha=64,
        contents="LoRA only; no base model, optimizer, datasets or private annotations")
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            for name in adapter.FILES:
                z.write(source / name, name)
        # Exclusive publication: never replace an existing handoff archive.
        output.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(output), sha256=sha256(output), bytes=output.stat().st_size)


def install_package(archive, destination=None):
    destination = Path(destination or adapter.ADAPTER)
    expected_names = {"manifest.json", *adapter.FILES}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=destination.parent) as work:
        if len(z.namelist()) != len(expected_names) or set(z.namelist()) != expected_names:
            raise ValueError("unexpected or duplicate adapter package member")
        if z.getinfo("manifest.json").file_size > 16384:
            raise ValueError("oversized adapter manifest")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("schema") != 1 or manifest.get("files") != adapter.FILES:
            raise ValueError("package differs from pinned adapter")
        staged = Path(work) / "adapter"
        staged.mkdir()
        for name in adapter.FILES:
            if z.getinfo(name).file_size > 512 * 1024 * 1024:
                raise ValueError("oversized adapter member")
            with z.open(name) as source, (staged / name).open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        verify_directory(staged)
        if destination.exists():
            verify_directory(destination)  # Refuse to overwrite changed/partial weights.
        else:
            staged.rename(destination)
    return dict(path=str(destination), status="VERIFIED_PINNED_ADAPTER")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source", type=Path)
    install = commands.add_parser("install")
    install.add_argument("archive", type=Path)
    install.add_argument("--destination", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--directory", type=Path, default=adapter.ADAPTER)
    args = p.parse_args(argv)
    if args.action == "export":
        result = export_package(args.output, args.source)
    elif args.action == "install":
        result = install_package(args.archive, args.destination)
    else:
        result = dict(path=str(verify_directory(args.directory)), status="VERIFIED_PINNED_ADAPTER")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

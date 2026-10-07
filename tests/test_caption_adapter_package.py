import zipfile

import pytest

from semantic_transmission import caption_adapter_package as package
from semantic_transmission.artifacts import sha256


@pytest.fixture
def source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name, value in {"adapter_config.json": b'{"r":32}', "adapter_model.safetensors": b'test weights'}.items():
        (source / name).write_bytes(value)
    monkeypatch.setattr(package.adapter, "FILES", {p.name: sha256(p) for p in source.iterdir()})
    return source


def test_handoff_roundtrip_idempotent_install_and_no_overwrite(source, tmp_path):
    output = tmp_path / "adapter.zip"
    receipt = package.export_package(output, source)
    assert receipt["sha256"] == sha256(output)
    dest = tmp_path / "clone/adapter"
    package.install_package(output, dest)
    assert package.install_package(output, dest)["status"] == "VERIFIED_PINNED_ADAPTER"
    for p in source.iterdir():
        assert sha256(dest / p.name) == sha256(p)
    with pytest.raises(FileExistsError):
        package.export_package(output, source)
    (dest / "adapter_config.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        package.install_package(output, dest)
    assert (dest / "adapter_config.json").read_text() == "changed"


@pytest.mark.parametrize("extra", ["../escaped", "optimizer.pt", "manifest.json"])
def test_unexpected_members_cannot_be_installed(source, tmp_path, extra):
    archive = tmp_path / "bad.zip"
    package.export_package(archive, source)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr(extra, "bad")
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        package.install_package(archive, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_changed_weights_rejected_before_export(source, tmp_path):
    (source / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        package.export_package(tmp_path / "bad.zip", source)

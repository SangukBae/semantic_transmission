import hashlib
import json
from pathlib import Path

import pytest

from semantic_transmission import protocol_definitions as definitions


def test_all_historical_definition_bytes_and_frozen_copy_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(definitions, "CACHE", tmp_path)
    records = json.loads(definitions.BUNDLE.read_text())["records"]
    assert len(records) == 7
    for name, record in records.items():
        path = definitions.protocol_path(name)
        assert path.read_text() == record["utf8"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert definitions.protocol_path(name) == path


def test_changed_cached_definition_and_unknown_names_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(definitions, "CACHE", tmp_path)
    path = definitions.protocol_path("ERE_STA_PROTOCOL")
    path.write_text("changed")
    with pytest.raises(ValueError, match="cached protocol"):
        definitions.protocol_path("ERE_STA_PROTOCOL")
    with pytest.raises(ValueError, match="unknown protocol"):
        definitions.protocol_path("../ERE_STA_PROTOCOL")


def test_bundle_text_cannot_change_without_hash(tmp_path, monkeypatch):
    bundle = json.loads(definitions.BUNDLE.read_text())
    bundle["records"]["ERE_STA_PROTOCOL"]["utf8"] += "changed"
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(definitions, "BUNDLE", path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        definitions.protocol_path("ERE_STA_PROTOCOL")

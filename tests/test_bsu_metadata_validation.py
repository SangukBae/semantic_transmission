import json

import pytest

from semantic_transmission import bsu_metadata_validation as validation
from semantic_transmission import metadata_channel
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack


def fixture_sources(monkeypatch):
    packet=pack(dict(visual_codec=dict(name='NTSCC+',rate_index_encoding='uint8'),
        keyframes=[dict(rate_count=864,rate_offset=0,rate_bytes=864)]),bytes([7])*864)
    row=dict(cohort='remote14',id='fixture',packet=packet,trajectory=b'',visual_uses=100,
        recorded_base_uses=100+accounting(len(packet))['complex_channel_uses'],
        recorded_trajectory_uses=0,provenance=dict(test=True))
    monkeypatch.setattr(validation,'sources',lambda:([row],{}))
    return packet


def test_completed_exact_transport_and_expected_low_snr_rejection(monkeypatch,tmp_path):
    fixture_sources(monkeypatch);calls=[]
    def transport(packet,snr,seed):
        calls.append((packet,snr,seed))
        received=packet if snr==10 else packet[:-1]+bytes([packet[-1]^1])
        return received,accounting(len(packet))|dict(bit_errors=int(snr!=10))
    monkeypatch.setattr(metadata_channel,'transmit',transport)
    validation.run(tmp_path)
    result=validation.read(tmp_path/'RESULT.json')
    assert result['status']==validation.COMPLETE and result['channel_cases']==10
    assert sum(r['rejected'] for r in result['channel'])==4
    assert sum(r['undetected_errors'] for r in result['channel'])==0
    assert all(r['restored_exact']==r['cases'] for r in result['channel'] if r['snr_db']==10)
    before=(tmp_path/'RESULT.json').read_bytes()
    validation.run(tmp_path)
    assert len(calls)==10 and (tmp_path/'RESULT.json').read_bytes()==before
    target=tmp_path/'channel/0000.bin';target.write_bytes(target.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='completed artifact changed'): validation.run(tmp_path)


def test_interruption_resume_verifies_saved_channels(monkeypatch,tmp_path):
    fixture_sources(monkeypatch);calls=[]
    def transport(packet,snr,seed):
        calls.append((packet,snr,seed))
        if len(calls)==3: raise KeyboardInterrupt()
        return packet,accounting(len(packet))|dict(bit_errors=0)
    monkeypatch.setattr(metadata_channel,'transmit',transport)
    with pytest.raises(KeyboardInterrupt):validation.run(tmp_path)
    assert validation.read(tmp_path/'status.json')['status']=='INTERRUPTED'
    assert len(list((tmp_path/'channel').glob('*.json')))==2
    validation.run(tmp_path)
    assert validation.read(tmp_path/'RESULT.json')['status']==validation.COMPLETE
    assert len(calls)==11  # Two completed cases are not transmitted again.


def test_primary_failure_never_marked_complete(monkeypatch,tmp_path):
    fixture_sources(monkeypatch)
    def transport(packet,snr,seed):
        return packet[:-1]+bytes([packet[-1]^1]),accounting(len(packet))|dict(bit_errors=1)
    monkeypatch.setattr(metadata_channel,'transmit',transport)
    with pytest.raises(ValueError,match='primary or undetected failures'):validation.run(tmp_path)
    assert not (tmp_path/'RESULT.json').exists()
    assert validation.read(tmp_path/'status.json')['status']=='FAILED'

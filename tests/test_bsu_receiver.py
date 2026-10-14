from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_receiver as receiver
from semantic_transmission import bsu_sequence_refinement as legacy
from semantic_transmission.artifacts import write_json
from semantic_transmission.wire import pack


@pytest.fixture
def sample(tmp_path):
    base=tmp_path/'base';run=base/'reconstruction/event_pairs/run'
    movie=run/'receiver/reconstruction/sample_0000.mp4';movie.parent.mkdir(parents=True)
    movie.write_bytes(b'not read during receiver refinement')
    frames=movie.with_name('sample_0000_frames');frames.mkdir()
    keys=run/'receiver/frames/sample/key_frames_received';keys.mkdir(parents=True)
    rng=np.random.default_rng(4);texture=rng.integers(0,255,(32,48,3),dtype=np.uint8)
    arrays=[np.roll(texture,i,axis=1) for i in range(13)]
    paths=[]
    for i,frame in enumerate(arrays):
        path=frames/f'{i:05d}.png';Image.fromarray(frame).save(path);paths.append(path)
        if i in (0,6,12):Image.fromarray(255-frame).save(keys/f'{i}.png')
    packet=pack(dict(video=dict(frames=13,fps=24,width=48,height=32),keyframes=[dict(index=i) for i in (0,6,12)]))
    for folder in ('received','transmitter'):
        (run/folder).mkdir();(run/folder/'metadata.bin').write_bytes(packet)
    write_json(run/'output_audit.json',dict(video=str(movie)))
    source=tmp_path/'source';source.mkdir()
    for i,a in enumerate(arrays):Image.fromarray(a).save(source/f'{i}.png')
    return dict(base=base,run=run,paths=paths,keys=keys,source=source,arrays=arrays)


def test_default_preserves_actual_received_keys_across_shared_boundaries(sample,tmp_path):
    out=tmp_path/'context';audits=receiver.refine_windows(sample['paths'],[0,6,12],sample['keys'],{},out)
    assert [p.name for p in sorted(out.glob('*.png'))]==[f'{i:05d}.png' for i in range(13)]
    for i in (0,6,12):assert np.array_equal(np.asarray(Image.open(out/f'{i:05d}.png')),255-sample['arrays'][i])
    assert len(audits)==2 and all(a['anchor']['received_keys_exact'] for a in audits)
    assert not list(tmp_path.glob('bsu-combined-*'))


def test_explicit_combined_retains_original_pixels_and_operator(sample,tmp_path):
    before=tmp_path/'old';after=tmp_path/'new'
    legacy.refine_windows(sample['paths'],[0,6,12],sample['keys'],{},before,'combined')
    receiver.refine_windows(sample['paths'],[0,6,12],sample['keys'],{},after,'combined')
    assert all((before/f'{i:05d}.png').read_bytes()==(after/f'{i:05d}.png').read_bytes() for i in range(13))


def test_full_receiver_never_needs_source_and_delivers_complete_timeline(sample,tmp_path):
    received=tmp_path/'received';received.mkdir()
    output=tmp_path/'refined'
    receiver.refine(dict(base=str(sample['base']),source_frames='/nonexistent/source',mode=receiver.DEFAULT_MODE),received,output)
    info=legacy.read(output/'refinement.json')
    assert info['received_key_endpoints_exact'] and info['timeline']['all_pts_checked']
    assert info['timeline']['frames']==13 and info['timeline']['fps']==24
    assert info['additional_context_transmission']==0 and not info['receiver_reads_source']


def test_default_sender_packet_and_cost_equal_existing_combined(sample,tmp_path):
    job=dict(base=str(sample['base']),source_frames=str(sample['source']),base_channel_uses=200000,
        digital_bits_per_symbol=4,extra_rate_ratio=.05,mode='combined')
    old=tmp_path/'old';new=tmp_path/'new'
    legacy.prepare(job,old);receiver.prepare({**job,'mode':receiver.DEFAULT_MODE},new)
    assert legacy.read(old/'sender.json')==legacy.read(new/'sender.json')
    assert legacy.read(old/'sender.json')['packet_present']
    assert (old/'trajectory.bin').read_bytes()==(new/'trajectory.bin').read_bytes()


def test_missing_received_key_fails_without_fallback_to_generated(sample,tmp_path):
    (sample['keys']/'6.png').unlink()
    with pytest.raises(FileNotFoundError):receiver.refine_windows(sample['paths'],[0,6,12],sample['keys'],{},tmp_path/'out')


def test_standalone_default_and_resume_use_received_endpoints(tmp_path):
    from semantic_transmission import bsu_refine
    received=tmp_path/'received';received.mkdir();generation=tmp_path/'generation';(generation/'frames').mkdir(parents=True)
    keys=[np.full((32,48,3),v,np.uint8) for v in (70,150)]
    for i,k in enumerate(keys):Image.fromarray(k).save(received/f'key{i}.png')
    (received/'metadata.bin').write_bytes(pack(dict(video=dict(frames=5,fps=24,width=48,height=32),keyframes=[dict(index=i) for i in (0,4)])))
    for i in range(5):Image.fromarray(np.full((32,48,3),100,np.uint8)).save(generation/f'frames/{i:05d}.png')
    out=tmp_path/'out';r=bsu_refine.execute(received,generation,out)
    assert r['mode']=='combined_context' and r['endpoint_reference']=='received_rgb'
    assert np.array_equal(np.asarray(Image.open(out/'frames/00000.png')),keys[0])
    before={p:p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
    assert bsu_refine.execute(received,generation,out)==r
    assert before=={p:p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}

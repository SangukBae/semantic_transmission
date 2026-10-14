import io
import json
import shutil
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_compare_transfer as tx
from semantic_transmission import bsu_receiver_compare as compare
from semantic_transmission.artifacts import write_json
from semantic_transmission.receiver_key_guard import preserve_keys
from semantic_transmission.terminal_progress import Progress, Display
from semantic_transmission.wire import pack


@pytest.fixture
def remote(tmp_path):
    repo = tmp_path/'remote'; root = repo/'outputs/bench'
    vid = 'webvid_test'; source = repo/'outputs/prepared'
    count = 9; keys = [0, 4, 8]

    def png(path, index):
        path.parent.mkdir(parents=True, exist_ok=True)
        y, x = np.mgrid[:320, :576]
        a = np.stack([(x+index*3)%256, (y+index)%256, (x+y)%256], axis=-1).astype(np.uint8)
        Image.fromarray(a).save(path)
        return tx.digest(path)

    hashes = {str(i):png(source/f'frames/{i}.png', i) for i in range(count)}
    (source/'normalized.mp4').write_bytes(b'normalized source movie')
    prepared = dict(id=vid, source_sha256='1'*64, frames=count, fps=24, width=576, height=320,
                    normalized_video=str(source/'normalized.mp4'), normalized_sha256=tx.digest(source/'normalized.mp4'),
                    source_frames=str(source/'frames'), source_frame_hashes=hashes)
    write_json(source/'source_prepared.json', prepared)
    plan = dict(mode='combined', signature='benchmark', videos=[dict(id=vid, prepared=str(source/'source_prepared.json'),
           prepared_sha256=tx.digest(source/'source_prepared.json'), source_sha256='1'*64, frames=count)])
    write_json(root/'protocol.json', plan)
    job = root/'snr_6'/vid; base = job/'base'; run = base/'reconstruction/event_pairs/run'
    movie = run/'receiver/reconstruction/sample_0000.mp4'
    movie.parent.mkdir(parents=True); movie.write_bytes(b'base movie')
    base_artifacts = {str(movie.relative_to(run.parent)):tx.digest(movie)}
    for i in range(count):
        path = movie.with_name('sample_0000_frames')/f'{i:05d}.png'
        base_artifacts[str(path.relative_to(run.parent))] = png(path, i+1)
    for i in keys:
        path = run/f'receiver/frames/sample/key_frames_received/{i}.png'
        base_artifacts[str(path.relative_to(run.parent))] = png(path, i)
    packet = run/'received/metadata.bin'; packet.parent.mkdir()
    packet.write_bytes(pack(dict(video=dict(frames=count, fps=24, width=576, height=320),
                                 keyframes=[dict(index=i) for i in keys], segments=[])))
    base_artifacts[str(packet.relative_to(run.parent))] = tx.digest(packet)
    write_json(run/'channel_accounting.json', dict(metadata_exact_match=True, bit_errors=0))
    base_artifacts['run/channel_accounting.json'] = tx.digest(run/'channel_accounting.json')
    quality = dict(frames=count, shape=[count, 320, 576, 3], psnr_db=20., ssim=.7, lpips_vgg=.3, clip=.9, dists=.2)
    bq = dict(video_sha256=tx.digest(movie), lossless_frames=quality, delivered_mp4=quality)
    write_json(run/'quality.json', bq); base_artifacts['run/quality.json'] = tx.digest(run/'quality.json')
    for stage in ('reconstruct', 'receive', 'channel', 'evaluate', 'output-audit'):
        write_json(base/f'reconstruction/event_pairs/stages/{stage}.json', dict(status='PASSED', artifacts=base_artifacts))
    write_json(base/'selection.json', dict(keys=keys))
    write_json(base/'protocol.json', {})
    write_json(base/'captions/event_pairs.json', dict(captions=[dict(caption='A moving pattern.')]))
    current_movie = job/'refined/reconstruction.mp4'; current_movie.parent.mkdir(); current_movie.write_bytes(b'current movie')
    artifacts = {'refined/reconstruction.mp4':tx.digest(current_movie)}
    for i in range(count):
        path = job/f'refined/frames/{i:05d}.png'; artifacts[str(path.relative_to(job))] = png(path, i+2)
    fq = dict(video_sha256=tx.digest(current_movie), lossless_frames=quality, delivered_mp4=quality)
    write_json(job/'refined/quality.json', fq); artifacts['refined/quality.json'] = tx.digest(job/'refined/quality.json')
    result = dict(status='COMPLETE_RECONSTRUCTION_METRICS_REVIEW_PENDING', signature='benchmark', mode='combined',
                  id=vid, snr_db=6, frames=count, source_sha256='1'*64, base=str(base), base_video=str(movie),
                  base_quality=bq, refined_quality=fq, base_channel_uses=100, trajectory_channel_uses=0,
                  total_channel_uses=100, trajectory_audit=dict(packet_present=False), artifacts=artifacts)
    write_json(job/'RESULT.json', result)
    return dict(repo=repo, root=root, job=job, source=source, vid=vid)


def manifest(remote):
    value = tx.inventory(str(remote['repo']), 'outputs/bench', [remote['vid']], [6])
    value['host'] = 'example'; value['signature'] = tx.fingerprint(value)
    return value


def test_inventory_minimal_and_read_only(remote):
    before = {str(p):tx.digest(p) for p in remote['repo'].rglob('*') if p.is_file()}
    m = manifest(remote)
    assert len(m['cases']) == 1 and len(m['sources']) == 1
    assert len(m['sources'][0]['frames_paths']) == 9
    assert not any('visual.c64' in p or 'model' in p or 'cache' in p for p in m['files'])
    assert before == {str(p):tx.digest(p) for p in remote['repo'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('field,value', [('status', 'RUNNING'), ('signature', 'other'), ('snr_db', 8),
                                       ('frames', 8), ('source_sha256', '0'*64), ('mode', 'cadence')])
def test_incomplete_or_mismatched_remote_never_enters_transfer(remote, field, value):
    p = remote['job']/'RESULT.json'; r = compare.read(p); r[field] = value; write_json(p, r)
    with pytest.raises(ValueError): manifest(remote)


@pytest.mark.parametrize('path', ['../outside', '/absolute', 'a/../b', 'a\nb', 'a\\b', 'a//b'])
def test_transfer_path_traversal_rejected(path):
    with pytest.raises(ValueError): tx.safe_relative(path)


def test_remote_symlink_cannot_export_unrelated_files(remote, tmp_path):
    p = remote['source']/'frames/0.png'; outside = tmp_path/'outside'; outside.write_bytes(p.read_bytes())
    p.unlink(); p.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'): manifest(remote)


def test_manifest_tamper_and_local_symlink_rejected(remote, tmp_path):
    m = manifest(remote); tx.validate_manifest(m)
    first = next(iter(m['files'])); m['files'][first]['size'] += 1
    with pytest.raises(ValueError, match='manifest changed'): tx.validate_manifest(m)
    mirror = tmp_path/'mirror'; mirror.mkdir(); (mirror/'outputs').symlink_to(remote['repo']/'outputs', target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe local'): tx.local_file(mirror, first)


def test_offline_resume_verifies_hashes_without_network(remote, monkeypatch):
    m = manifest(remote)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('offline tried network'))
    assert tx.fetch(m, remote['repo'], offline=True)['reused']
    p = remote['source']/'normalized.mp4'; p.write_bytes(b'x'*p.stat().st_size)
    with pytest.raises(ValueError, match='offline mirror'): tx.fetch(m, remote['repo'], offline=True)


def test_receiver_job_has_no_source_and_cadence_needs_no_key_pixels(remote):
    m = manifest(remote); job = compare.receiver_inputs(remote['repo'], m['cases'][0])
    assert not any('source' in k for k in job)
    assert not any(str(remote['source']) in str(v) for v in job.values())


def test_rgb_anchor_exactness_and_bounded_support():
    rng = np.random.default_rng(23)
    frames = rng.integers(0, 256, size=(17, 32, 48, 3), dtype=np.uint8)
    keys = [frames[0]//2, frames[-1]//2]
    before = frames.copy(); output, audit = preserve_keys(frames, keys)
    assert np.array_equal(frames, before)
    assert np.array_equal(output[0], keys[0]) and np.array_equal(output[-1], keys[-1])
    assert np.array_equal(output[4:-4], frames[4:-4])
    assert audit['receiver_reads_source'] is False


@pytest.mark.parametrize('arm', ['cadence_only', 'key_guarded'])
def test_multisegment_refinement_failure_resume_and_shared_keys(remote, tmp_path, monkeypatch, arm):
    from semantic_transmission import etri_60s_check
    m = manifest(remote); case = m['cases'][0]; job = compare.receiver_inputs(remote['repo'], case)
    output = tmp_path/'comparison'
    calls = []

    def encode(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(9, command)
        Path(command[-1]).write_bytes(b'encoded video')
    monkeypatch.setattr(subprocess, 'run', encode)
    monkeypatch.setattr(etri_60s_check, 'pts_audit', lambda *args: dict(frames=9, fps=24))
    if arm == 'cadence_only':
        job['key_paths'] = ['/not/read']*3; job['trajectory'] = '/not/read'
    with pytest.raises(subprocess.CalledProcessError): compare.refine(job, output, 'frozen', arm)
    assert not (output/'generation.json').exists()
    before = {p.name:tx.digest(p) for p in (output/'frames').glob('*.png')}
    compare.refine(job, output, 'frozen', arm)
    assert compare.verify_receipt(output, 'generation', 'frozen')
    assert before == {p.name:tx.digest(p) for p in (output/'frames').glob('*.png')}
    assert len(before) == 9
    for i, k in enumerate(case['keys']):
        expected = case['key_paths'][i] if arm=='key_guarded' else case['raw_frames'][k]
        assert np.array_equal(np.asarray(Image.open(output/f'frames/{k:05d}.png')),
                              np.asarray(Image.open(remote['repo']/expected)))
    (output/'frames/00000.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'): compare.verify_receipt(output, 'generation', 'frozen')


def test_fixed_display_json_distinguishes_overall_and_inner_passes():
    p = Progress()
    p.feed('COMPARE_PROGRESS '+json.dumps(dict(overall='7/36 (19.4%) 완료', current='video / 6dB',
                                              stage='구간 2/10: 100/100 (100%)'))+'\n')
    assert p.overall == '7/36 (19.4%) 완료'
    out = io.StringIO(); d = Display(out)
    p.feed('Loading model from: local cache\n')
    d.draw(p.rows(1, 'log')); p.feed('COMPARE_PROGRESS '+json.dumps(dict(stage='평가 1/200 (0.5%)'))+'\n')
    d.draw(p.rows(2, 'log'))
    assert '\x1b[6F' in out.getvalue() and p.overall.startswith('7/36')
    assert p.state == '실행 중'
    p.feed('REPORT: /tmp/comparison/report.html\n')
    assert p.summary == ['REPORT: /tmp/comparison/report.html']


def test_partial_report_aggregates_only_matched_complete_cohorts(remote, tmp_path):
    m = manifest(remote); root = tmp_path/'result'; root.mkdir()
    case = m['cases'][0]; folder = root/'comparison'/compare.case_name(case)/'current_combined'; folder.mkdir(parents=True)
    q = dict(quality=dict(psnr_db=20, ssim=.7, lpips_vgg=.3, clip=.9, dists=.2),
             temporal=dict(flow_error_mean=.1, cadence_error_mean=.1, false_freeze_rate=.1), key_fidelity=[])
    path = folder/'quality.json'; write_json(path, dict(lossless_frames=q, delivered_mp4=q))
    compare.receipt(folder, 'evaluation', 'frozen', [path])
    result = compare.report(root, m, dict(signature='frozen'))
    assert result['status']=='PARTIAL' and result['completed']==1 and result['total']==3
    assert result['matched_cohort_aggregates']==[] and len(result['pending'])==2
    assert result['performance_improvement_claimed'] is False


def test_options_cannot_relabel_a_frozen_comparison(remote):
    m = manifest(remote); args = compare.parser().parse_args(['--host', 'example', '--remote-repo', str(remote['repo']),
        '--remote-root', 'outputs/bench', '--ids', remote['vid'], '--snr', '6'])
    compare.request_matches(m, args)
    args.snr = [10]
    with pytest.raises(ValueError, match='selection/options changed'): compare.request_matches(m, args)


def test_transfer_retries_only_missing_or_corrupt_files_and_checks_receipts(remote, tmp_path, monkeypatch):
    m = manifest(remote); mirror = tmp_path/'mirror'
    shutil.copytree(remote['repo'], mirror)
    names = m['sources'][0]['frames_paths'][:2]
    (mirror/names[0]).unlink()
    corrupt = mirror/names[1]; corrupt.write_bytes(b'x'*corrupt.stat().st_size)
    requested = []

    class Rsync:
        def __init__(self, command, **kwargs):
            listing = next(a.split('=', 1)[1] for a in command if a.startswith('--files-from='))
            requested.extend(Path(listing).read_bytes().decode().strip('\0').split('\0'))
            for name in requested:
                shutil.copyfile(remote['repo']/name, mirror/name)
            self.stdout = io.StringIO(' 1,000 100% 1.00MB/s\n')
            self.returncode = 0

        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self): return 0

    monkeypatch.setattr(subprocess, 'Popen', Rsync)
    result = tx.fetch(m, mirror)
    assert set(requested) == set(names) and result['transferred_files'] == 2
    assert not tx.inspect_files(m, mirror)
    assert tx.fetch(m, mirror, offline=True)['transferred_files'] == 0


def test_eta_resets_between_hash_passes(monkeypatch, capsys):
    now = [0.]
    monkeypatch.setattr(compare.time, 'monotonic', lambda: now[0])
    cb = compare.tick('verify')
    cb(1, 100)
    now[0] = 10.; cb(100, 100)
    now[0] = 100.; cb(1, 100)
    now[0] = 104.; cb(50, 100)
    stages = [json.loads(line.split(' ', 1)[1])['stage'] for line in capsys.readouterr().out.splitlines()]
    assert stages[0].endswith('ETA 미확정') and stages[1].endswith('ETA 0초')
    assert stages[2].endswith('ETA 미확정') and stages[3].endswith('ETA 4초')
